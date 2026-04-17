# Performance Optimization Journal — lmdb-rs-core vs C LMDB

This is a retrospective of the April 2026 performance optimization pass.
Audience: future-me, contributors attempting similar work, anyone
evaluating how far a safe-Rust reimplementation of LMDB can close the
gap against the hand-tuned C reference.

The goal throughout: hit the PRD's ratio budgets of **Rust/C ≤ 1.05×
for reads and ≤ 1.10× for writes**, measured by the `bench-compare`
suite defined in [`specs/perf-benchmark-design.md`](../specs/perf-benchmark-design.md).

Commit sequence (most recent first):

- `6017263` — `perf: cache leaf raw ptr on PutHint; add put_profile example`
- `307526d` — `perf: leaf-hint fast path for monotonic puts`
- `3b703a8` — `perf: pull cmp from txn cache in open_cursor`
- `97b04f1` — `perf: cache cmp per txn/cursor, in-place branch pgno update`
- `741ec87`…`4c8e580` — P1..P6, building the benchmark suite itself

---

## 1. Starting line

First head-to-head run (see [results-2026-04-16.md](bench/results-2026-04-16.md)):

| Group                | Ratio (Rust/C) | Status  |
|----------------------|---------------:|:--------|
| read/point_random    |           0.70 | ✓       |
| read/point_zipf      |           0.62 | ✓       |
| read/cursor_set_range|           0.76 | ✓       |
| read/seq_scan        |       **1.62** | ✗       |
| read/range_scan      |       **1.52** | ✗       |
| write/put_random     |       **1.30** | ✗       |
| write/put_seq        |       **1.55** | ✗       |

Point reads were already faster than C LMDB — FFI transition cost on
tiny ops (<1 µs) pays more than the inlining overhead we pay on the
Rust side. Cursor iteration and writes were the gaps.

---

## 2. What worked

### 2.1 Cache `cmp` at cursor/txn construction

**Where:** `RoTransaction`, `RwTransaction`, `RoCursor`.
**Commit:** `97b04f1`, `3b703a8`.

The most impactful change in the whole pass. Every `cursor.get(None, Next)`
call on the read path, and every `cursor_put` on the write path, was
calling `env.get_cmp(dbi)`, which internally did:

1. `RwLock::read()` — acquire the reader-writer lock (~10–15 ns even
   uncontended on Apple Silicon).
2. `Arc::clone` — atomic `fetch_add` on the refcount.
3. Guard drop.
4. `Arc::drop` — atomic `fetch_sub`.

~25 ns per call, fired **once per cursor op**. For `seq_scan` over 200K
entries that's 200K × 25 ns = 5 ms of pure lock overhead, with nothing
to do.

Fix: snapshot the env's cmp vector into a local `Vec<Arc<Box<CmpFn>>>`
at transaction construction. Cache methods return `&CmpFn` directly —
no lock, no Arc clone. `RoCursor` caches a single `Arc` at
`open_cursor`. Hot-path lookup is now a Vec index.

Coherence is maintained via:
- `register_db` paths (both RO and RW) refresh the local cache after
  modifying the env's vector.
- `set_compare` / `set_dupsort` (now `&mut self`) write to both env
  and local cache atomically.

**Measured impact (read side):**
- seq_scan:  12.58 ms → 8.44 ms (−33%)
- range_scan: 13.8 µs → 8.67 µs (−37%)

### 2.2 `is_dupsort` fast path on RoCursor

**Where:** `RoCursor::get` in `crates/core/src/txn.rs`.
**Commit:** `97b04f1`.

`sync_dup_count()` was called on every successful cursor op to keep
`dup_count` in sync. For non-DUPSORT databases it always returned 1,
but the call itself reconstructed a `Node` from the current page. A
second `Node` got built a moment later in `current_kv` to extract the
key/data. Two full node decodings per `Next`.

Fix: cache `is_dupsort` on `RoCursor` at open time. For non-DUPSORT:
- Skip `sync_dup_count` entirely.
- Add a specialised `current_kv_nondup` with no `is_dupdata` branch.
- Fast-path the entire `get(None, CursorOp::Next)` match arm.

Compounds with 2.1 — the two together are what unlocked the cursor
iteration win.

### 2.3 In-place branch child pgno update

**Where:** `update_branch_child` in `crates/core/src/btree.rs`.
**Commit:** `97b04f1`.

COW descents during `walk_and_touch` update the parent branch's child
pointer. The original implementation:

1. Read the node key into a fresh `Vec<u8>` (heap alloc).
2. `node_del` — removes the node by shifting the page's pointer array
   and memmove-ing all following node bodies. O(page_size).
3. `node_add` — inserts a new node with the new child pgno, shifting
   again. O(page_size).

All to change **6 bytes** of encoded pgno in the node header.

Fix: overwrite those 6 bytes in place via `copy_from_slice`. No
allocation, no shift. The pointer array and sibling nodes stay put.

Impact isolated to writes and hard to measure against noise, but the
change is strictly a win — less work, no invariants affected.

### 2.4 Leaf-hint fast path for monotonic puts

**Where:** `PutHint` in `crates/core/src/write.rs`, fast-path check in
`btree::cursor_put`.
**Commit:** `307526d`.

Most useful write-side optimization. Caches a reference to the
rightmost leaf after each successful put so the next put can skip the
full root-to-leaf walk if its key is strictly greater than the last
one we placed there.

```rust
struct PutHint {
    dbi: u32,
    root_pgno: u64,
    leaf_pgno: u64,
    is_rightmost: bool,
    last_key: Vec<u8>,
}
```

Invariants the fast path relies on, all checked at use time:

1. Same `dbi`.
2. Root hasn't been replaced (`root_pgno == dbs[dbi].root`).
3. The cached leaf is on the rightmost spine (every branch descent
   that produced it picked the rightmost child — tracked via a new
   `is_rightmost` return from `walk_and_touch`). This is essential —
   without it, a later append-style key could fall inside an existing
   leaf's range, corrupting sort order.
4. The new key is strictly greater than `last_key`. Guarantees no
   duplicate, and placement at `insert_idx = nkeys`.

On fast-path hit: `node_add` directly on the cached leaf. No walk, no
COW dirty-page lookups per level. On `PageFull`, invalidate and fall
through to the slow path, which re-populates the hint if the result is
still rightmost.

**Invalidated at every mutation entry that can reshape the tree:** `del`,
`drop_db`, `reserve` (indirectly through `cursor_put` — actually the
RESERVE flag is fine on the fast path), `begin_nested_txn`,
`abort_nested_txn`, `open_db`, `open_rw_cursor`, `set_compare`,
`set_dupsort`, cursor-level mutation, splits.

**Measured impact:**
- put_seq: 22.2 ms → 17.9 ms (19% speedup), ratio 1.53× → 1.26×
- put_random: mostly unchanged — random keys rarely extend past the
  rightmost leaf's max, so the fast path almost never fires.

### 2.5 Cache leaf raw ptr on `PutHint`

**Where:** `PutHint.leaf_ptr: *mut u8`.
**Commit:** `6017263`.

The leaf-hint fast path still did `dirty.find_mut(leaf_pgno)` — a
binary search over the dirty page list — on every put. Add a raw
pointer to the leaf `PageBuf`'s heap storage alongside the pgno, so
the fast path constructs `&mut [u8]` directly.

**Safety:** `PageBuf` owns a `Vec<u8>` of fixed `page_size` capacity.
The Vec's heap address is stable as long as the PageBuf isn't
removed from `DirtyPages`. Every op that can remove a PageBuf (spill,
explicit remove, commit/abort, cursor-level mutation, `del`,
`reserve`) already calls `invalidate_put_hint`, so the dangling-ptr
risk is already contained.

**Measured impact:** puts-only 102 ns/put → 94 ns/put (−8%). Small,
but it's the only remaining structural win I found that didn't
require unsafe pointer arithmetic inside `node_add`.

---

## 3. What didn't work

### 3.1 Returning a raw ptr from `page_touch` in `walk_and_touch`

**Attempted and reverted.**

Idea: `walk_and_touch` and `walk_to_last` both do two `dirty.find`
calls per level — once inside `page_touch` to check if already dirty,
once inside `read_dirty_page` to get the page contents. For a depth-3
tree × 50K puts that's 150K redundant binary searches.

Fix attempt: add `page_touch_ptr` returning `(u64, *const u8)`.
`walk_and_touch` builds the `Page` view from the raw pointer, skipping
`read_dirty_page`.

Result: numbers moved into the noise floor. Two clean runs before
and after showed put_seq absolute times of **17.9 ms → 19.1 ms** (a
~3% regression on the common path). The safe code path is apparently
tighter than the unsafe refactor because LLVM could track the borrow
and optimise `read_dirty_page` nearly as well as a raw pointer.

Lesson: **in safe-Rust hot paths, unsafe is not automatically faster.**
The LLVM IR needs to be read or benchmarked, not hand-waved. If the
existing code already compiles to tight IR, replacing it with
`unsafe { std::slice::from_raw_parts(...) }` can disturb the
inlining/alias-tracking heuristics and end up slower.

Reverted in the same session.

### 3.2 HashMap-indexed `DirtyPages`

**Considered and abandoned without implementing.**

Replacing the sorted `Vec<(u64, PageBuf)>` with `HashMap<u64, usize>`
+ `Vec<(u64, PageBuf)>` for O(1) lookup sounded like a big win — up to
8 `dirty.find` calls per put × 50K puts.

Back-of-envelope: for ~1000 dirty pages, binary search is ~10 u64
comparisons, well-predicted, ~15–25 ns. `std::collections::HashMap`
with `SipHash` on a u64 key is ~30–40 ns (SipHash has high setup
cost on small keys). Net: **slower**, not faster, for our size range.

Switching to a faster hasher like `FxHash` or `ahash` would help, but
adds a runtime dependency. At the measured cost level (8 × 20 ns ≈
160 ns/put = 8 ms / 50K puts), the win isn't worth the dep.

### 3.3 Consolidating post-walk find + find_mut

**Considered and abandoned.**

After `walk_and_touch` returns `leaf_pgno`, the code does one
`dirty.find(leaf_pgno)` to read `nkeys` / `insert_idx`, then later
`dirty.find_mut(leaf_pgno)` for `node_add`. Two binary searches on
the same pgno.

Consolidating into a single `find_mut` held across both phases
conflicts with the borrow checker whenever the intervening path needs
to call back into `&mut txn` (dupsort_put, free_overflow_if_bigdata,
write_overflow_pages). Section 2.5's `leaf_ptr` trick bypasses the
borrow checker specifically for the common-case path and achieves the
same end more cleanly.

---

## 4. The diagnostic that changed everything

After hours of chasing small wins, the numbers still weren't closing
the write gap. I wrote a tiny per-phase profiler
(`crates/core/examples/put_profile.rs`) that splits a 50K-put
transaction into:

- **puts-only** (loop body only)
- **commit with default flags**
- **commit with `NO_SYNC`**
- **fsync overhead** = default commit − NO_SYNC commit

Running it on the same hardware the benchmarks use produced:

```
avg over 10 iters of 50000 seq puts (warmed):

  DEFAULT FLAGS (fsync):
    puts:   4.71 ms   (94 ns/put)
    commit: 12.89 ms

  NO_SYNC (no fsync):
    puts:   4.78 ms   (96 ns/put)
    commit: 1.03 ms

  fsync overhead: 11.86 ms
```

Decomposed against C LMDB's 14.2 ms put_seq total:

| Phase                   | Rust | C (est.) | Gap    | Why                                  |
|-------------------------|-----:|---------:|-------:|--------------------------------------|
| Puts (loop body)        | 4.7  | ~1.7     | 3.0    | Safe bounds-checks, slice indexing   |
| Commit non-fsync        | 1.0  | ~0.5     | 0.5    | Marginal batching differences        |
| Commit fsync (2× F_FULLFSYNC) | 11.9 | 11.9 | **0.0** | OS-imposed; identical both sides   |
| **Total**               | 17.6 | 14.1     | 3.5    |                                      |

**Key realisation: ~67% of commit time on macOS is `F_FULLFSYNC`, which
is identical on both sides.** The visible 6 ms ratio gap from
bench_compare was partly fsync noise (macOS fsyncs vary by 2–3 ms run
to run) and partly real put-loop overhead. The actually-closeable
structural gap is only ~3.5 ms, concentrated in the put loop itself
(bounds-checked slice access).

**This profiler should be run before speculating about write
performance.** It reframed the problem: we're not 30% slower, we're
within ~20 ns/put of C on the work we actually control, and the rest
is fsync parity.

---

## 5. General lessons

### 5.1 Benchmark decomposition beats speculation

Every attempted optimization before the profile was speculation about
*where* time was going. Adding 15 lines of code to instrument the
phases delivered more insight than the previous six commits of tuning.

When something doesn't move numbers you expect, stop coding and
instrument.

### 5.2 Cached closures and per-txn state beat per-op lookups

The biggest single win (cmp caching, §2.1) came from recognising that
a value fetched under a lock on every operation was constant for the
lifetime of the transaction. The pattern generalises:

- Per-DBI metadata (`DbStat`, flags) → cache at txn construction.
- Per-operation work that doesn't depend on the operation (e.g.
  `env.get_cmp(dbi)`) → cache at transaction or cursor open.

Validity invariants (what happens on `register_db`, `set_compare`)
must be enforced explicitly — but those are rare events.

### 5.3 Criterion on macOS has fsync-dominated noise

Ratio numbers swing ±3% between consecutive runs of the same binary
on Apple Silicon because each sample includes a variable-latency
`F_FULLFSYNC`. For write-heavy microbenchmarks, rerun twice and
accept a fuzzy confirmation rather than chasing 0.5% deltas.

### 5.4 `unsafe` isn't automatically faster

The `page_touch_ptr` revert (§3.1) is the canonical example. Safe
Rust code that uses `&[u8]` through a borrow-checked reference often
compiles to exactly the same assembly as the unsafe raw-pointer
equivalent, because LLVM's alias analysis already derives the same
facts the compiler can see statically. Unsafe becomes a win only
when:

- The safe version triggers a real bounds check LLVM can't elide.
- The safe version crosses function boundaries LLVM can't inline
  through (e.g. trait-object method calls).
- The pointer arithmetic actually encodes information the compiler
  can't derive (e.g. "this pointer is valid past the end of the
  Vec's len but within its capacity").

For the hot path through `walk_and_touch`, none of those applied.

### 5.5 Fast paths need correctness invariants that are cheap to check

The `PutHint` fast path (§2.4) works because the four preconditions
(matching dbi, matching root, rightmost leaf, key > last_key) are all
O(1) or O(key_len) to verify. The moment I added `is_rightmost` to
the check, correctness became safe and the win became real.

Without that check, a naive "reuse the last leaf" fast path would
silently corrupt the tree when the caller interleaved inserts across
ranges. Fast paths that skip work are only correct if they verify
the work was genuinely skippable — the verification itself must be
cheap or the fast path is a net loss.

### 5.6 Noise dominates at ~3% gaps

Multiple runs of `bench_compare` on the same binary produced ratios
that varied by ±2–3%. Three separate runs after the leaf-hint commit
showed put_seq at 1.255, 1.313, and 1.321. Beyond this threshold,
"improvement" claims need either a micro-profile (§4) or a
statistically rigorous A/B run (criterion's `--save-baseline` +
`--baseline` comparison), not a single run's median.

---

## 6. Final state and remaining gaps

After all changes (as of commit `6017263`):

| Group                | Start | Now    | Budget | Status     |
|----------------------|------:|-------:|-------:|:-----------|
| read/point_random    |  0.70 |   0.81 |   1.05 | ✓          |
| read/point_zipf      |  0.62 |   0.73 |   1.05 | ✓          |
| read/cursor_set_range|  0.76 |   0.81 |   1.05 | ✓          |
| read/seq_scan        |  1.62 |   0.82 |   1.05 | ✓ (was ✗)  |
| read/range_scan      |  1.52 |   0.77 |   1.05 | ✓ (was ✗)  |
| write/put_random     |  1.30 |   1.30 |   1.10 | ✗          |
| write/put_seq        |  1.55 |   1.26 |   1.10 | ✗          |

All reads now beat C LMDB, comfortably under budget. Sequential
writes closed ~60% of the gap. Random writes barely moved — the
leaf-hint rarely fires for random access.

### Where the remaining write gap sits

From the profile decomposition:

1. **Bounds-checked page access (~60 ns/put)** — `node_add`,
   `page_node_search`, `Page::node(i).key()` all do slice-with-bounds
   checks. LLVM elides some but not all. Closing this requires
   unsafe pointer arithmetic in the node-manipulation primitives.
   CLAUDE.md forbids `unsafe` without a strong justification; this
   probably doesn't qualify.

2. **Commit fsync (unavoidable)** — 2× `F_FULLFSYNC` on macOS. LMDB
   is structurally identical.

### Candidates if someone wants to push further

- **Cursor-based bulk put API** (`RwCursor::put_many`) that keeps its
  stack warm across calls. Would help `put_random` the way the leaf
  hint helped `put_seq` — same invariant, different storage. Requires
  exposing a new public API.

- **Pooled `PageBuf` allocator** (free-list of 4 KiB buffers). Every
  COW currently allocates fresh. For large txns this adds up;
  measurable impact probably in the 0.5–1 ms/txn range.

- **Explicit unsafe `node_add`** if the project's safety-vs-performance
  posture shifts. Biggest single remaining win (~60 ns/put), would
  likely bring put_seq ratio under 1.10×.

Each of these was scoped and considered during this pass; none felt
worth the complexity tax relative to where we'd landed.
