# Performance Benchmark Design

**Status:** Draft · 2026-04-16
**Target crate:** `lmdb-rs-core` (path: `crates/core`)
**Harness:** `criterion = 0.8.2`
**FFI comparison baseline:** `lmdb-master-sys = 0.2.5` (raw bindings to vendored C LMDB `mdb.master`)

---

## 1. Motivation & Scope

The PRD ([lmdb-rs-prd.md §2.1 & §6.2](./lmdb-rs-prd.md)) commits us to:

- Read performance within **5%** of C LMDB.
- Write performance within **10%** of C LMDB.

Recent history (commits `0e4cbc0 Fix all MINOR gaps + performance parity with C LMDB`, `e55a085 Close all remaining audit gaps`) indicates the implementation has reached feature + functional parity. This spec defines the benchmark suite that **measures** that parity, guards it against regressions, and publishes comparable numbers against the vendored C LMDB.

### 1.1 Goals

1. **Self-benchmarks (G1):** Microbenchmarks of every hot public API path in `lmdb-rs-core`, reported in `ns/op`, `ops/s`, and `B/s` where meaningful, with statistical confidence intervals.
2. **Head-to-head (G2):** Apples-to-apples comparison between `lmdb-rs-core` and C LMDB for identical workloads, producing a ratio ("Rust/C") per metric. Budget: reads ≤1.05×, writes ≤1.10×.
3. **Regression tripwire (G3):** Baseline JSON checked into `docs/bench/baselines/` so a nightly CI job can detect >5% regressions between commits.
4. **Scaling curves (G4):** Not just a single point; sweep value size, key size, and DB size so we understand performance as a function of shape.
5. **Reproducibility (G5):** One `make bench` invocation reproduces the full suite; results are self-describing (hardware, page size, filesystem, flags).

### 1.2 Non-Goals

- Not reproducing Symas's historical microbench site verbatim. We compare within our own harness using the currently vendored C source, not published numbers from a different machine/year.
- Not benchmarking against redb, sled, RocksDB, etc. (Future work; see [§9](#9-future-work).)
- Not optimizing based on benchmark results inside this spec. This spec builds the instrument; tuning happens in follow-up tasks driven by results.
- Not running benchmarks in normal `cargo test`. They are opt-in via `cargo bench` / `make bench`.

---

## 2. Workload Taxonomy

Every scenario is a 3-tuple **(operation, shape, access pattern)**. The matrix below is the canonical list of benchmark groups. Each row corresponds to one `criterion::BenchmarkGroup`.

### 2.1 Read workloads

| ID      | Group                    | Operation                                        | Shape                                    | Access          |
|---------|--------------------------|--------------------------------------------------|------------------------------------------|-----------------|
| R-01    | `read/point_random`      | `RoTransaction::get`                             | k=16B, v=100B, N=1M                      | uniform random  |
| R-02    | `read/point_zipf`        | `RoTransaction::get`                             | k=16B, v=100B, N=1M                      | zipfian θ=0.99  |
| R-03    | `read/seq_scan`          | `RoCursor::iter`                                 | k=16B, v=100B, N=1M                      | full scan       |
| R-04    | `read/range_scan`        | `RoCursor::iter_from` + 1000 nexts               | k=16B, v=100B, N=1M                      | from random key |
| R-05    | `read/overflow`          | `RoTransaction::get`                             | k=16B, v=64KB (overflow), N=10K          | uniform random  |
| R-06    | `read/cursor_set`        | `RoCursor::get(key, Set)`                        | k=16B, v=100B, N=1M                      | uniform random  |
| R-07    | `read/cursor_set_range`  | `RoCursor::get(key, SetRange)`                   | k=16B, v=100B, N=1M                      | uniform random  |
| R-08    | `read/first_last`        | `RoCursor::get(_, First/Last)`                   | k=16B, v=100B, N=1M                      | n/a             |
| R-09    | `read/cold_mmap`         | `RoTransaction::get` after purge of FS cache     | k=16B, v=100B, N=1M                      | uniform random  |

### 2.2 Write workloads

| ID      | Group                    | Operation                                    | Shape                                       | Commit pattern                      |
|---------|--------------------------|----------------------------------------------|---------------------------------------------|-------------------------------------|
| W-01    | `write/put_random`       | `RwTransaction::put`                         | k=16B, v=100B, N=100K                       | 1 commit per txn (batch=all)        |
| W-02    | `write/put_random_sync`  | `RwTransaction::put` (default flags, fsync)  | k=16B, v=100B, N=10K                        | 1 commit per key (synchronous)      |
| W-03    | `write/put_seq`          | `RwTransaction::put`                         | sorted k=16B, v=100B, N=100K                | 1 commit per txn                    |
| W-04    | `write/append`           | `RwTransaction::put` w/ `APPEND`             | sorted k=16B, v=100B, N=100K                | 1 commit per txn                    |
| W-05    | `write/reserve`          | `RwTransaction::reserve` + in-place fill     | k=16B, v=1KB, N=100K                        | 1 commit per txn                    |
| W-06    | `write/overwrite`        | `RwTransaction::put` of existing key         | k=16B, v=100B, N=100K prepop                | 1 commit per txn                    |
| W-07    | `write/delete`           | `RwTransaction::del`                         | k=16B, N=100K prepop                        | 1 commit per txn                    |
| W-08    | `write/overflow_put`     | `RwTransaction::put`                         | k=16B, v=64KB (overflow), N=10K             | 1 commit per txn                    |
| W-09    | `write/writemap`         | `put_random` with `EnvFlags::WRITE_MAP`      | k=16B, v=100B, N=100K                       | 1 commit per txn                    |
| W-10    | `write/nosync`           | `put_random` with `EnvFlags::NO_SYNC`        | k=16B, v=100B, N=100K                       | 1 commit per key                    |

### 2.3 DUPSORT workloads

| ID      | Group                | Operation                              | Shape                                |
|---------|----------------------|----------------------------------------|--------------------------------------|
| D-01    | `dupsort/put`        | `put` with `DUP_SORT`                  | 10K keys × 100 dups × 16B value      |
| D-02    | `dupsort/iter`       | cursor iter over all dups              | as D-01                              |
| D-03    | `dupfixed_multiple`  | `put` with `MULTIPLE` / `DUP_FIXED`    | 1K keys × 1K dups × 8B value         |

### 2.4 Transaction lifecycle

| ID      | Group                      | Operation                            |
|---------|----------------------------|--------------------------------------|
| T-01    | `txn/ro_begin_commit`      | `begin_ro_txn` + drop                |
| T-02    | `txn/rw_empty_commit`      | `begin_rw_txn` + `commit()`          |
| T-03    | `txn/rw_single_put_commit` | `begin_rw_txn` + 1 put + `commit()`  |
| T-04    | `txn/nested`               | nested child txn open + commit       |

### 2.5 Concurrency

| ID      | Group                      | Operation                                      |
|---------|----------------------------|------------------------------------------------|
| C-01    | `concurrent/readers`       | M reader threads × random point reads          |
| C-02    | `concurrent/1w_Mr`         | 1 writer (put loop) + M readers (point reads)  |

Reported as **aggregate throughput** across threads (ops/s). Threads are pinned if `core_affinity` is available.

### 2.6 Scaling sweeps

| ID      | Group                      | X axis                                  | Fixed                 |
|---------|----------------------------|-----------------------------------------|-----------------------|
| S-01    | `scale/value_size/read`    | v ∈ {32, 128, 512, 2K, 8K, 32K, 128K}   | k=16B, N=100K         |
| S-02    | `scale/value_size/write`   | same                                    | same                  |
| S-03    | `scale/key_size`           | k ∈ {8, 32, 128, 256, 511}              | v=100B, N=100K        |
| S-04    | `scale/db_size/read`       | N ∈ {1K, 10K, 100K, 1M, 10M}            | k=16B, v=100B         |

### 2.7 Admin / maintenance

| ID   | Group                  | Operation                                    |
|------|------------------------|----------------------------------------------|
| A-01 | `admin/open_warm`      | `Environment::open` on existing DB           |
| A-02 | `admin/copy`           | `env.copy`                                   |
| A-03 | `admin/copy_compact`   | `env.copy_compact`                           |
| A-04 | `admin/sync_force`     | `env.sync(true)`                             |

---

## 3. Crate Layout & Cargo Configuration

### 3.1 Directory layout

```
crates/
├── core/
│   ├── Cargo.toml             # add [[bench]] entries (harness = false)
│   └── benches/
│       ├── common/mod.rs       # shared harness: paths, RNG, workload generators
│       ├── bench_read.rs       # R-01..R-09
│       ├── bench_write.rs      # W-01..W-10
│       ├── bench_cursor.rs     # R-03, R-04, R-06..R-08 + cursor-specific
│       ├── bench_dupsort.rs    # D-01..D-03
│       ├── bench_txn.rs        # T-01..T-04
│       ├── bench_concurrent.rs # C-01, C-02
│       ├── bench_scaling.rs    # S-01..S-04
│       ├── bench_admin.rs      # A-01..A-04
│       └── bench_compare.rs    # head-to-head Rust vs C (all groups, gated)
└── bench-compat/               # NEW: FFI adapter over lmdb-master-sys
    ├── Cargo.toml
    └── src/lib.rs              # mirrors lmdb-rs-core API surface used in benches
```

Rationale for a separate `bench-compat` crate (not a dev-dependency in `core`):

- Isolates the C `build.rs` (requires `cc` + C toolchain) from the core crate's dev cycle. `cargo test` on `core` stays pure-Rust.
- Makes `bench_compare.rs` import a single unified trait (`KvBackend`) that both implementations satisfy, so the criterion groups are parameterized on backend without `#[cfg]` noise.
- License separation: LMDB is OpenLDAP-licensed; keeping the FFI in its own crate keeps `core`'s dep tree clean.

### 3.2 `crates/core/Cargo.toml` additions

```toml
[dev-dependencies]
criterion      = { version = "0.8", features = ["html_reports", "cargo_bench_support"] }
rand           = "0.9"
rand_chacha    = "0.9"     # deterministic RNG seeded per bench
rand_distr     = "0.5"     # zipfian
core_affinity  = "0.8"
bench-compat   = { path = "../bench-compat", optional = true }

[features]
# Off by default so `cargo test` doesn't pull in the C toolchain.
bench-compare  = ["dep:bench-compat"]

[[bench]]
name = "bench_read"
harness = false

[[bench]]
name = "bench_write"
harness = false

# ... one entry per benches/*.rs file

[[bench]]
name = "bench_compare"
harness = false
required-features = ["bench-compare"]
```

`harness = false` is mandatory for criterion. `required-features` makes the comparison bench compile only when the user opts in, per CLAUDE.md "minimize dependencies".

### 3.3 `crates/bench-compat/Cargo.toml`

```toml
[package]
name = "bench-compat"
edition = "2024"
publish = false

[dependencies]
lmdb-master-sys = "0.2"     # raw FFI, no wrapper overhead
libc.workspace  = true
```

The crate exposes a `CBackend` struct whose method signatures are byte-for-byte compatible with what the benches need. It handles `MDB_env` lifecycle, `MDB_txn`, `MDB_cursor`, and `MDB_val` conversions. No `unsafe` leaks out; all FFI is contained in this crate.

### 3.4 Shared trait (`bench_compare.rs`)

```rust
pub trait KvBackend {
    type Env;
    type RoTxn<'e> where Self: 'e;
    type RwTxn<'e> where Self: 'e;

    fn open(path: &Path, map_size: usize, flags: BenchFlags) -> Self::Env;
    fn begin_ro(env: &Self::Env) -> Self::RoTxn<'_>;
    fn begin_rw(env: &Self::Env) -> Self::RwTxn<'_>;

    fn get<'t>(txn: &'t Self::RoTxn<'_>, k: &[u8]) -> Option<&'t [u8]>;
    fn put(txn: &mut Self::RwTxn<'_>, k: &[u8], v: &[u8]);
    fn commit(txn: Self::RwTxn<'_>);
    // ...cursor methods, etc.
}

struct RustBackend; impl KvBackend for RustBackend { /* ... */ }
struct CBackend;    impl KvBackend for CBackend    { /* ... */ }
```

Every criterion group in `bench_compare.rs` is generic over `B: KvBackend`. We run each group twice — once with `RustBackend`, once with `CBackend` — reported under sibling IDs like `compare/read/point_random/rust` and `compare/read/point_random/c`. Criterion's built-in grouping lets us plot them side-by-side in the HTML report.

---

## 4. Harness Conventions

### 4.1 Setup / teardown patterns

Use `iter_batched` with `BatchSize::PerIteration` for mutating ops so each sample starts from a clean state:

```rust
c.benchmark_group("write/put_random")
 .throughput(Throughput::Elements(N as u64))
 .bench_function("lmdb-rs-core", |b| {
     b.iter_batched(
         || setup_empty_env(),                    // per-sample setup (untimed)
         |env| {
             let mut txn = env.begin_rw_txn().unwrap();
             for (k, v) in workload.iter() {
                 txn.put(MAIN_DBI, k, v, WriteFlags::empty()).unwrap();
             }
             txn.commit().unwrap();
         },
         BatchSize::PerIteration,
     );
 });
```

For read benches, populate **once** outside the `iter`:

```rust
let env = setup_and_fill(N);
group.bench_function("lmdb-rs-core", |b| {
    let keys = &workload.keys;
    let mut idx = 0usize;
    b.iter(|| {
        let txn = env.begin_ro_txn().unwrap();
        let v = txn.get(MAIN_DBI, &keys[idx % keys.len()]).unwrap();
        idx = idx.wrapping_add(1);
        black_box(v);
    });
});
```

### 4.2 Deterministic workloads

All keys / values / access sequences come from `rand_chacha::ChaCha20Rng::seed_from_u64(BENCH_SEED)` where `BENCH_SEED = 0x_C0FFEE_u64`. This means two runs on the same machine produce identical data distributions, so variance is dominated by system noise, not input.

Workload generators live in `benches/common/mod.rs`:

```rust
pub struct Workload {
    pub keys: Vec<Vec<u8>>,
    pub values: Vec<Vec<u8>>,
    pub access: Vec<u32>,     // index into keys, pre-shuffled
}

pub fn gen(seed: u64, n: usize, key_sz: usize, val_sz: usize, dist: Dist) -> Workload { ... }

pub enum Dist { Uniform, Zipf(f64), Sequential }
```

### 4.3 Filesystem isolation

- Benchmarks create temp directories in `$BENCH_TMP` (default `/tmp/lmdb-bench`). The env var lets CI point to `tmpfs` or a dedicated SSD mount.
- `cold_mmap` bench (R-09) uses a helper that flushes OS page cache between samples: `purge` on macOS (via `std::process::Command`), `echo 3 > /proc/sys/vm/drop_caches` on Linux. Requires elevated privileges; bench skips with a clear diagnostic if unavailable.

### 4.4 Criterion settings

Per-group tuning in `common/mod.rs`:

```rust
pub fn configure(name: &str) -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
        .with_plots()
        .noise_threshold(0.02)           // 2% is our target run-to-run noise floor
        .significance_level(0.05)
        .confidence_level(0.99)
}
```

Slow groups (`scale/db_size/read` with N=10M, admin copy) override with `sample_size(20)` and `measurement_time(30s)` to keep total suite runtime manageable.

### 4.5 Use `black_box` correctly

Every hot-path read must `black_box(result)` to defeat LLVM's dead-code elimination; every key/value slice read from `workload` must pass through `black_box(&workload.keys[idx])` to prevent loop-invariant hoisting. This is especially important for the C backend, whose FFI boundary the compiler might optimize around.

### 4.6 Throughput reporting

- `Throughput::Elements(N)` for ops/sec.
- `Throughput::Bytes((k_len + v_len) * N)` for byte throughput on sequential scans and bulk writes.
- Per-operation latency is emitted automatically in `ns/op` at the bench_function level.

---

## 5. C LMDB Comparison Methodology

### 5.1 Apples-to-apples rules

| Variable                  | Rule                                                                            |
|---------------------------|---------------------------------------------------------------------------------|
| Page size                 | Both use OS native (4K Linux / 16K macOS). Don't force a non-default.           |
| Map size                  | Identical; sized to 2× largest dataset to avoid grow during benchmark.          |
| Env flags                 | Identical (`empty()` by default; `NO_SYNC` / `WRITE_MAP` explicit per-group).  |
| Key / value buffers       | Same `Vec<u8>` from the shared workload — no hash / copy differences.           |
| Commit granularity        | Identical number of commits per group.                                          |
| Reader slot count         | Default (126) on both.                                                          |
| Transaction drop          | Rust `RoTransaction` dropped; C `mdb_txn_abort` called. No reuse.               |
| Cursor reuse              | Neither side reuses cursors across samples unless the operation inherently does. |
| Filesystem                | Same directory root; `bench-compare` creates sibling `rust/` and `c/` subdirs. |
| Warmup                    | Identical warmup count.                                                         |

### 5.2 Ratio output

The `bench_compare.rs` harness emits a post-run summary:

```
compare/read/point_random        rust: 612 ns/op   c: 608 ns/op   ratio: 1.007   ok (≤1.05)
compare/write/put_random         rust: 8.4 µs/op   c: 7.9 µs/op   ratio: 1.063   ok (≤1.10)
compare/read/seq_scan            rust: 41 MB/s     c: 44 MB/s     ratio: 1.073   FAIL (>1.05)
```

Thresholds from the PRD. A single FAIL exits the harness with code 1; CI uses this to gate merges.

### 5.3 Fairness audit

Each comparison group carries a `// FAIRNESS:` comment in source documenting any behavioral difference between backends (e.g., "C LMDB uses `mdb_cursor_get` with `MDB_NEXT` whereas our `iter()` yields a tuple; both do a single page access per iteration"). This file is reviewed on every PR that touches the comparison suite so drift stays visible.

---

## 6. Reporting & Regression Detection

### 6.1 HTML reports

Criterion writes to `target/criterion/`. After `make bench`, the Makefile copies a timestamped snapshot to `docs/bench/reports/YYYY-MM-DD/` so historical runs are browsable without re-running.

### 6.2 Baselines

```bash
make bench-baseline NAME=v0.1.0    # runs suite with --save-baseline v0.1.0
make bench-compare-to NAME=v0.1.0  # runs suite with --baseline v0.1.0
```

Criterion's built-in baseline feature persists raw samples in `target/criterion/*/v0.1.0/`. We copy the interesting ones (one per release) to `docs/bench/baselines/` and commit them. Files are small (hundreds of KB); retention is fine.

### 6.3 Regression gate

`scripts/bench-diff.rs` (small Rust binary in `xtask/` if we ever add one; for now a plain `cargo run --bin bench-diff`) parses `target/criterion/.../estimates.json` and compares `mean.point_estimate` against the baseline. >5% regression on any group → nonzero exit code.

### 6.4 Results doc template

`docs/bench/results-{date}.md`:

```markdown
# Benchmark Results — 2026-04-16

## Environment
- CPU: Apple M3 Max, 14 cores (10 P + 4 E)
- OS: macOS 15.2, page size 16384
- FS: APFS on NVMe
- Rust: 1.84.0
- C LMDB: vendored `mdb.master` @ <sha>, compiled `-O2 -DNDEBUG`

## Summary vs C LMDB
| Group            | Rust       | C          | Ratio | Budget | Pass |
| ---------------- | ---------- | ---------- | ----- | ------ | ---- |
| read/point_random| 612 ns/op  | 608 ns/op  | 1.007 | 1.05   | ✓    |
| write/put_random | 8.4 µs/op  | 7.9 µs/op  | 1.063 | 1.10   | ✓    |
...
```

Referenced from `docs/index.md`.

---

## 7. Makefile Integration

Add to `Makefile` (per CLAUDE.md "Makefile-based automation"):

```make
bench:                          ## run full bench suite (no comparison)
	@cargo bench --bench bench_read --bench bench_write --bench bench_cursor \
	    --bench bench_dupsort --bench bench_txn --bench bench_scaling --bench bench_admin

bench-compare:                  ## run head-to-head Rust vs C LMDB
	@cargo bench --features bench-compare --bench bench_compare

bench-quick:                    ## CI smoke test — 1s per bench
	@cargo bench -- --quick

bench-baseline:
	@test -n "$(NAME)" || (echo "usage: make bench-baseline NAME=<tag>" && exit 1)
	@cargo bench -- --save-baseline $(NAME)

bench-regress:
	@test -n "$(NAME)" || (echo "usage: make bench-regress NAME=<tag>" && exit 1)
	@cargo bench -- --baseline $(NAME)

.PHONY: bench bench-compare bench-quick bench-baseline bench-regress
```

---

## 8. Implementation Phasing

Phases are strictly sequential; each lands in its own PR with its own green `cargo build && cargo test && cargo +nightly fmt && cargo clippy -- -D warnings`.

**Phase P1 — Harness skeleton.** (1 PR)
Adds `benches/common/mod.rs`, `bench_read.rs` (R-01 only), Cargo config, Makefile targets. Establishes conventions. Success = `make bench-quick` runs green.

**Phase P2 — Self-benchmark coverage.** (1 PR)
Fills out read, write, cursor, dupsort, txn, admin, scaling groups (§2.1–§2.4, §2.6, §2.7). No comparison yet. Success = all self-benches run on CI in <10 minutes.

**Phase P3 — FFI adapter.** (1 PR)
Introduces `crates/bench-compat/` and the `KvBackend` trait. No criterion groups yet; just the adapter with its own unit tests verifying byte-equal behavior against `lmdb-rs-core` on a small dataset.

**Phase P4 — Head-to-head suite.** (1 PR)
Adds `bench_compare.rs`, the ratio reporter, and the fairness-audit comments. Success = `make bench-compare` produces the summary table; thresholds hold.

**Phase P5 — Concurrency + cold cache.** (1 PR)
Adds C-01, C-02, R-09. These are finicky (thread scheduling, cache drops); they land last so earlier signal isn't blocked on platform quirks.

**Phase P6 — Baseline + regression CI.** (1 PR)
Records the first baseline under `docs/bench/baselines/v0.1.0/`, adds the diff tool, wires the nightly CI job.

---

## 9. Future Work

- Cross-engine comparison (redb, sled, fjall).
- Long-running soak + RSS/mmap residency measurements (criterion is latency-focused; tracking peak RSS needs a separate harness).
- Flamegraph integration (`cargo flamegraph --bench`).
- `dhat`/`bytehound` allocation counts per operation (should be ~zero for reads).
- JMH-style percentiles (p50/p99/p99.9) — criterion reports mean/stddev; for tail latency we may need HDR-based sampling.
- Windows numbers once Windows support lands (see README §Platform Support).

---

## 10. Risks & Open Questions

| Risk                                                  | Mitigation                                                                 |
|-------------------------------------------------------|----------------------------------------------------------------------------|
| macOS `purge` requires `sudo` → R-09 skipped on dev   | Document; run R-09 only in CI Linux container with `drop_caches`.          |
| C LMDB requires `cc` toolchain → harder to build      | `bench-compare` is feature-gated; default `make bench` stays pure-Rust.   |
| Benchmark noise on shared dev laptops                 | `noise_threshold(0.02)`; encourage users to run with `bench-quick`; CI uses a dedicated runner. |
| `lmdb-master-sys` API drift between 0.2.x releases    | Pin to `= 0.2.5` with a comment; re-audit on each bump.                    |
| DUPSORT semantics subtly differ between impls         | FAIRNESS comments + round-trip equality assertion in D-01 setup.           |

---

## References

- [criterion 0.8 docs](https://docs.rs/criterion/latest/criterion/) — `0.8.2` is current as of 2026-04.
- [lmdb-master-sys 0.2.5](https://crates.io/crates/lmdb-master-sys) — raw FFI we use for the C baseline.
- [LMDB/dbbench](https://github.com/LMDB/dbbench) — workload inspiration (LevelDB-derived).
- [lmdbjava/benchmarks](https://github.com/lmdbjava/benchmarks) — JMH-based comparison suite we mirror structurally.
- [LMDB microbench page](http://www.lmdb.tech/bench/microbench/) — Symas-published metrics for historical context.
- Internal: [lmdb-rs-prd.md §6.2](./lmdb-rs-prd.md), [docs/research/lmdb-architecture.md](../docs/research/lmdb-architecture.md).
