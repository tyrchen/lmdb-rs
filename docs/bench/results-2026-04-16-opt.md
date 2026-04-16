# Benchmark Results — 2026-04-16 (post-optimization)

Second run of the `bench-compare` suite after landing three targeted
optimizations aimed at the gaps identified in
[results-2026-04-16.md](./results-2026-04-16.md).

## Environment

Same machine / toolchain as the baseline run:
Apple Silicon M-series, macOS 15.x (Darwin 24.6.0), Rust stable 2024.

## Summary vs C LMDB

| Group                              | Rust (before → after) | C (ns)     | Ratio (before → after) | Budget | Status |
|------------------------------------|----------------------:|-----------:|-----------------------:|-------:|:------:|
| compare/read/point_random          |      353 → 391 ns     |        514 |   0.70× → **0.76×**    |  1.05  |  ✓    |
| compare/read/point_zipf            |      200 → 209 ns     |        321 |   0.62× → **0.65×**    |  1.05  |  ✓    |
| compare/read/seq_scan              |  1.91 ms → **1.01 ms** |    1.26 ms |   1.62× → **0.80×**    |  1.05  |  ✓    |
| compare/read/range_scan            |   10.1 µs → **5.48 µs** |    6.71 µs |   1.52× → **0.82×**    |  1.05  |  ✓    |
| compare/read/cursor_set_range      |      353 → 409 ns     |        556 |   0.76× → **0.74×**    |  1.05  |  ✓    |
| compare/write/put_random           |  29.1 ms → 28.9 ms    |    22.8 ms |   1.30× → **1.27×**    |  1.10  |  ✗    |
| compare/write/put_seq              |  22.6 ms → 22.2 ms    |    14.5 ms |   1.55× → **1.53×**    |  1.10  |  ✗    |

## What changed

Three optimizations landed together in `97b04f1` + `3b703a8`:

1. **RoCursor caches cmp + is_dupsort at open_cursor.** The old hot path
   inside `cursor.get(None, Next)` did `env.get_cmp(dbi)` per call — a
   `RwLock::read` + `Arc::clone` + guard drop + `Arc::drop`, ~20–30 ns
   fired 200K times per `seq_scan` sample. Now populated once per
   cursor, hot path dereferences a local `Arc`.

2. **RoCursor skips dup bookkeeping on non-DUPSORT.** `sync_dup_count()`
   used to reconstruct the current node just to conclude `count = 1`.
   Cached `is_dupsort` flag lets us skip it entirely; a specialised
   `current_kv_nondup()` also saves a node reconstruction on every
   `Next` return.

3. **In-place branch child pgno update.** The COW descent in
   `walk_and_touch` used to call `node_del` + `node_add` on the branch —
   O(n) pointer-array shift + data-area memmove — just to rewrite 6
   bytes of encoded pgno. Now writes those 6 bytes directly.

Also, `cmp_cache` was added to both `RoTransaction` and `RwTransaction`
so that `txn.get` / `cursor_put` / `cursor_del` / `reserve` pay one Vec
clone at transaction construction instead of an `env.get_cmp` RwLock
per op.

## Impact

Cursor iteration — the biggest offender in the baseline — is now
**faster than C LMDB** by ~20% on both `seq_scan` and `range_scan`.
Absolute numbers on the Rust side:

- `seq_scan`:  1.91 ms → **1.01 ms** (1.89× speedup)
- `range_scan`: 10.1 µs → **5.48 µs** (1.84× speedup)

The cached hot path is doing the work the cursor was always supposed
to be doing — pure B+ tree traversal without lock overhead.

Point reads regressed slightly in ratio (0.70 → 0.76 on point_random)
because opening a fresh transaction per iteration now pays the cost of
cloning the cmp Vec. Still comfortably inside the 1.05 budget.

## Still open — writes

Writes barely moved: `put_random` 1.30× → 1.27×, `put_seq` 1.55× →
1.53×. The in-place branch pgno fix helped a few ns/put; the cmp cache
saved an `Arc::clone` per put. Neither touches the dominant cost.

Candidates for a follow-up:

- **Dirty-page list** is a `Vec<(pgno, PageBuf)>` kept sorted by pgno;
  `DirtyPages::insert` calls `Vec::insert` which is O(n). For a 50K-put
  txn with ~1.25K new pages, that's quadratic work. A `BTreeMap` or
  bucketed structure would be O(log n).
- **Page-buf allocation.** Each new page allocates a fresh
  `vec![0u8; page_size]`. A free-list / slab of 4 KiB buffers would
  cut the malloc traffic dramatically.
- **`PathLevel` stack in `walk_and_touch`** — allocated per put via
  `TreePath::new()`. Could be a `SmallVec` on the stack.

None of these were attempted in this PR.

## How to reproduce

```
make bench-compare
```

The ratio summary is printed at the end of the run; exit code is 1
when any group is over-budget so CI can gate on it.
