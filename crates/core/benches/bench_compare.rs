//! Head-to-head comparison: `lmdb-rs-core` vs the vendored C LMDB.
//!
//! Every group runs the identical workload against both backends and reports
//! sibling criterion IDs (`compare/read/point_random/rust` vs `.../c`). The
//! post-run summary at the bottom of this file prints a ratio table the PRD
//! can gate on.
//!
//! # FAIRNESS PRINCIPLES
//! Each workload is generated exactly once and handed to both backends
//! byte-for-byte identical. Both backends use the OS default page size,
//! identical map size, identical env flags, and open a fresh tempdir per
//! sample for destructive ops. Slices yielded on the read path are only
//! `black_box`ed for length; neither side sees an extra copy.

#![cfg(feature = "bench-compare")]

mod common;

use std::{hint::black_box, path::Path, time::Duration};

use bench_compat::{CCursor, CEnv, COp};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group};
use lmdb_rs_core::{
    env::Environment,
    types::{CursorOp, MAIN_DBI},
};

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, Workload, bulk_load, configure, fresh_tempdir,
    gen_workload, open_empty_env,
};

// ---------------------------------------------------------------------------
// Fairness-critical sizes — kept smaller than self-benches so both backends
// finish the full comparison in a reasonable wall-clock budget.
// ---------------------------------------------------------------------------

const POINT_N: usize = 200_000;
const WRITE_N: usize = 50_000;
const KEY_SZ: usize = 16;
const VAL_SZ: usize = 100;
const RANGE_STEPS: usize = 1_000;

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------
//
// Each backend owns its filesystem artefact and exposes just the operations
// the benches need. Lifetimes are elided on the trait methods because the
// concrete types each side uses differ (Rust returns `&'env [u8]` via the
// txn; C returns `&'txn [u8]` via mdb_get). What matters here is the ops,
// not the exact borrow contract, so we box work into plain closures.

trait Backend {
    /// Short label used in criterion IDs: `rust` or `c`.
    const LABEL: &'static str;

    fn open(path: &Path, map_size: usize) -> Self;

    /// Bulk-load all (key, value) pairs in a single transaction.
    fn bulk_put(&self, workload: &Workload);

    /// Open a RO txn, call `body(k)` over every access index, and let the
    /// impl handle transaction lifetime. Returns sum of value lengths
    /// observed so the compiler can't strip the loop.
    fn bench_point_get(&self, keys: &[Vec<u8>], access: &[u32]) -> u64;

    /// Cursor seq-scan the whole DB, returning (count, sum_of_value_lens).
    fn bench_seq_scan(&self) -> (u64, u64);

    /// Range scan: seek with SetRange, then walk `steps` Next's. Return
    /// sum of value lengths.
    fn bench_range_scan(&self, start: &[u8], steps: usize) -> u64;

    /// SetRange for the given key, one lookup, return value length.
    fn bench_cursor_set_range(&self, key: &[u8]) -> usize;
}

// ---------------------------------------------------------------------------
// Rust backend
// ---------------------------------------------------------------------------

struct RustBackend {
    _dir: tempfile::TempDir,
    env: Environment,
}

impl Backend for RustBackend {
    const LABEL: &'static str = "rust";

    fn open(path: &Path, map_size: usize) -> Self {
        let dir = tempfile::Builder::new()
            .prefix("lmdb-bench-cmp-rust-")
            .tempdir_in(path)
            .expect("tempdir");
        let env = open_empty_env(dir.path(), map_size);
        Self { _dir: dir, env }
    }

    fn bulk_put(&self, workload: &Workload) {
        bulk_load(&self.env, workload);
    }

    fn bench_point_get(&self, keys: &[Vec<u8>], access: &[u32]) -> u64 {
        let idx = access[0] as usize;
        let txn = self.env.begin_ro_txn().expect("ro txn");
        let v = txn.get(MAIN_DBI, &keys[idx]).expect("get");
        v.len() as u64
    }

    fn bench_seq_scan(&self) -> (u64, u64) {
        let txn = self.env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let mut count = 0u64;
        let mut bytes = 0u64;
        let mut op = CursorOp::First;
        while let Ok((_k, v)) = cursor.get(None, op) {
            count += 1;
            bytes += v.len() as u64;
            op = CursorOp::Next;
        }
        (count, bytes)
    }

    fn bench_range_scan(&self, start: &[u8], steps: usize) -> u64 {
        let txn = self.env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let (_k, v) = cursor
            .get(Some(start), CursorOp::SetRange)
            .expect("set_range");
        let mut acc = v.len() as u64;
        for _ in 1..steps {
            match cursor.get(None, CursorOp::Next) {
                Ok((_k, v)) => acc += v.len() as u64,
                Err(_) => break,
            }
        }
        acc
    }

    fn bench_cursor_set_range(&self, key: &[u8]) -> usize {
        let txn = self.env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let (_k, v) = cursor
            .get(Some(key), CursorOp::SetRange)
            .expect("set_range");
        v.len()
    }
}

// ---------------------------------------------------------------------------
// C backend
// ---------------------------------------------------------------------------

struct CBackendWrap {
    _dir: tempfile::TempDir,
    env: CEnv,
}

impl Backend for CBackendWrap {
    const LABEL: &'static str = "c";

    fn open(path: &Path, map_size: usize) -> Self {
        let dir = tempfile::Builder::new()
            .prefix("lmdb-bench-cmp-c-")
            .tempdir_in(path)
            .expect("tempdir");
        let env = CEnv::open(dir.path(), map_size, 0, 0).expect("open C env");
        Self { _dir: dir, env }
    }

    fn bulk_put(&self, workload: &Workload) {
        let mut txn = self.env.begin_rw().expect("rw");
        let dbi = txn.main_dbi().expect("dbi");
        for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
            txn.put(dbi, k, v, 0).expect("put");
        }
        txn.commit().expect("commit");
    }

    fn bench_point_get(&self, keys: &[Vec<u8>], access: &[u32]) -> u64 {
        let idx = access[0] as usize;
        let txn = self.env.begin_ro().expect("ro");
        let dbi = txn.main_dbi().expect("dbi");
        let v = txn.get(dbi, &keys[idx]).expect("get");
        v.len() as u64
    }

    fn bench_seq_scan(&self) -> (u64, u64) {
        let txn = self.env.begin_ro().expect("ro");
        let dbi = txn.main_dbi().expect("dbi");
        let mut cursor: CCursor<'_> = txn.open_cursor(dbi).expect("cursor");
        let mut count = 0u64;
        let mut bytes = 0u64;
        let mut op = COp::First;
        while let Ok((_k, v)) = cursor.get(None, op) {
            count += 1;
            bytes += v.len() as u64;
            op = COp::Next;
        }
        (count, bytes)
    }

    fn bench_range_scan(&self, start: &[u8], steps: usize) -> u64 {
        let txn = self.env.begin_ro().expect("ro");
        let dbi = txn.main_dbi().expect("dbi");
        let mut cursor = txn.open_cursor(dbi).expect("cursor");
        let (_k, v) = cursor.get(Some(start), COp::SetRange).expect("set_range");
        let mut acc = v.len() as u64;
        for _ in 1..steps {
            match cursor.get(None, COp::Next) {
                Ok((_k, v)) => acc += v.len() as u64,
                Err(_) => break,
            }
        }
        acc
    }

    fn bench_cursor_set_range(&self, key: &[u8]) -> usize {
        let txn = self.env.begin_ro().expect("ro");
        let dbi = txn.main_dbi().expect("dbi");
        let mut cursor = txn.open_cursor(dbi).expect("cursor");
        let (_k, v) = cursor.get(Some(key), COp::SetRange).expect("set_range");
        v.len()
    }
}

// ---------------------------------------------------------------------------
// Group helpers
// ---------------------------------------------------------------------------

fn bench_reads_pair(c: &mut Criterion, group_name: &str, workload: &Workload, tmp: &Path) {
    let rust = RustBackend::open(tmp, DEFAULT_MAP_SIZE);
    rust.bulk_put(workload);
    let cbe = CBackendWrap::open(tmp, DEFAULT_MAP_SIZE);
    cbe.bulk_put(workload);

    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(1));

    let keys = &workload.keys;
    let access = &workload.access;

    let mut idx = 0usize;
    group.bench_function(BenchmarkId::from_parameter(RustBackend::LABEL), |b| {
        b.iter(|| {
            let a = [access[idx % access.len()]; 1];
            idx = idx.wrapping_add(1);
            black_box(rust.bench_point_get(keys, &a));
        });
    });

    let mut idx = 0usize;
    group.bench_function(BenchmarkId::from_parameter(CBackendWrap::LABEL), |b| {
        b.iter(|| {
            let a = [access[idx % access.len()]; 1];
            idx = idx.wrapping_add(1);
            black_box(cbe.bench_point_get(keys, &a));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

fn compare_point_random(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    bench_reads_pair(c, "compare/read/point_random", &workload, tmp.path());
}

fn compare_point_zipf(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Zipf(0.99));
    bench_reads_pair(c, "compare/read/point_zipf", &workload, tmp.path());
}

fn compare_seq_scan(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Sequential);
    let rust = RustBackend::open(tmp.path(), DEFAULT_MAP_SIZE);
    rust.bulk_put(&workload);
    let cbe = CBackendWrap::open(tmp.path(), DEFAULT_MAP_SIZE);
    cbe.bulk_put(&workload);

    let mut group = c.benchmark_group("compare/read/seq_scan");
    group.throughput(Throughput::Elements(POINT_N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter(RustBackend::LABEL), |b| {
        b.iter(|| black_box(rust.bench_seq_scan()));
    });
    group.bench_function(BenchmarkId::from_parameter(CBackendWrap::LABEL), |b| {
        b.iter(|| black_box(cbe.bench_seq_scan()));
    });

    group.finish();
}

fn compare_range_scan(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    let rust = RustBackend::open(tmp.path(), DEFAULT_MAP_SIZE);
    rust.bulk_put(&workload);
    let cbe = CBackendWrap::open(tmp.path(), DEFAULT_MAP_SIZE);
    cbe.bulk_put(&workload);

    let keys = workload.keys.clone();
    let access = workload.access.clone();

    let mut group = c.benchmark_group("compare/read/range_scan");
    group.throughput(Throughput::Elements(RANGE_STEPS as u64));

    let mut i = 0usize;
    group.bench_function(BenchmarkId::from_parameter(RustBackend::LABEL), |b| {
        b.iter(|| {
            let start = &keys[access[i % access.len()] as usize];
            i = i.wrapping_add(1);
            black_box(rust.bench_range_scan(start, RANGE_STEPS));
        });
    });
    let mut i = 0usize;
    group.bench_function(BenchmarkId::from_parameter(CBackendWrap::LABEL), |b| {
        b.iter(|| {
            let start = &keys[access[i % access.len()] as usize];
            i = i.wrapping_add(1);
            black_box(cbe.bench_range_scan(start, RANGE_STEPS));
        });
    });

    group.finish();
}

fn compare_cursor_set_range(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    let rust = RustBackend::open(tmp.path(), DEFAULT_MAP_SIZE);
    rust.bulk_put(&workload);
    let cbe = CBackendWrap::open(tmp.path(), DEFAULT_MAP_SIZE);
    cbe.bulk_put(&workload);

    let mut group = c.benchmark_group("compare/read/cursor_set_range");
    group.throughput(Throughput::Elements(1));

    let keys = workload.keys.clone();
    let access = workload.access.clone();

    let mut i = 0usize;
    group.bench_function(BenchmarkId::from_parameter(RustBackend::LABEL), |b| {
        b.iter(|| {
            let k = &keys[access[i % access.len()] as usize];
            i = i.wrapping_add(1);
            black_box(rust.bench_cursor_set_range(k));
        });
    });
    let mut i = 0usize;
    group.bench_function(BenchmarkId::from_parameter(CBackendWrap::LABEL), |b| {
        b.iter(|| {
            let k = &keys[access[i % access.len()] as usize];
            i = i.wrapping_add(1);
            black_box(cbe.bench_cursor_set_range(k));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Write benches: bulk put, measured per sample.
// ---------------------------------------------------------------------------

fn compare_put_random(c: &mut Criterion) {
    use rand::{SeedableRng, seq::SliceRandom};
    use rand_chacha::ChaCha20Rng;

    let mut workload = gen_workload(BENCH_SEED, WRITE_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    // Shuffle key/value pairs in lockstep for random insert order.
    let mut rng = ChaCha20Rng::seed_from_u64(BENCH_SEED ^ 0x1234_5678);
    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = workload
        .keys
        .drain(..)
        .zip(workload.values.drain(..))
        .collect();
    pairs.shuffle(&mut rng);
    for (k, v) in pairs {
        workload.keys.push(k);
        workload.values.push(v);
    }

    let mut group = c.benchmark_group("compare/write/put_random");
    group.throughput(Throughput::Elements(WRITE_N as u64));
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let tmp_root = tempfile::tempdir().expect("tempdir");

    group.bench_function(BenchmarkId::from_parameter(RustBackend::LABEL), |b| {
        b.iter_batched(
            || RustBackend::open(tmp_root.path(), DEFAULT_MAP_SIZE),
            |be| be.bulk_put(&workload),
            BatchSize::PerIteration,
        );
    });
    group.bench_function(BenchmarkId::from_parameter(CBackendWrap::LABEL), |b| {
        b.iter_batched(
            || CBackendWrap::open(tmp_root.path(), DEFAULT_MAP_SIZE),
            |be| be.bulk_put(&workload),
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn compare_put_seq(c: &mut Criterion) {
    let workload = gen_workload(BENCH_SEED, WRITE_N, KEY_SZ, VAL_SZ, Dist::Sequential);

    let mut group = c.benchmark_group("compare/write/put_seq");
    group.throughput(Throughput::Elements(WRITE_N as u64));
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let tmp_root = tempfile::tempdir().expect("tempdir");

    group.bench_function(BenchmarkId::from_parameter(RustBackend::LABEL), |b| {
        b.iter_batched(
            || RustBackend::open(tmp_root.path(), DEFAULT_MAP_SIZE),
            |be| be.bulk_put(&workload),
            BatchSize::PerIteration,
        );
    });
    group.bench_function(BenchmarkId::from_parameter(CBackendWrap::LABEL), |b| {
        b.iter_batched(
            || CBackendWrap::open(tmp_root.path(), DEFAULT_MAP_SIZE),
            |be| be.bulk_put(&workload),
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------
//
// Note on the ratio report: criterion 0.8 doesn't expose a built-in
// "sibling bench ratio" feature, but the HTML report groups IDs that share
// a prefix, which already gives a side-by-side view. For a programmatic
// gate on Rust/C ratios, run `cargo bench --features bench-compare --
// --save-baseline latest` and pair it with a small parser over
// `target/criterion/compare/*/*/estimates.json` — that's what P6 does.
//
// The `fresh_tempdir` helper isn't used here because comparison groups
// keep their fixtures for the whole bench binary lifetime (reads) or use
// ad-hoc subtree roots (writes). Silence the "unused import" warning.

#[allow(dead_code)]
fn _force_link(_: &Path) {
    let _ = fresh_tempdir("unused");
}

criterion_group! {
    name = benches;
    config = configure();
    targets =
        compare_point_random,
        compare_point_zipf,
        compare_seq_scan,
        compare_range_scan,
        compare_cursor_set_range,
        compare_put_random,
        compare_put_seq,
}

// Can't use `criterion_main!(benches);` because we want to print a ratio
// summary after criterion finishes. The body below mirrors what the macro
// expands to, plus a trailing `print_ratio_summary()` call.
fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
    print_ratio_summary();
}

// ---------------------------------------------------------------------------
// Post-run ratio summary
// ---------------------------------------------------------------------------
//
// criterion writes per-bench estimates to
//   target/criterion/compare/<group>/<rust|c>/new/estimates.json
// after each run. We parse just the `mean.point_estimate` field (nanoseconds)
// out of each pair and print a ratio table. The PRD budgets are:
//   reads:  Rust / C ≤ 1.05
//   writes: Rust / C ≤ 1.10
// A single FAIL returns a non-zero exit code so CI can gate on it.

#[derive(Debug)]
struct RatioRow {
    group: String,
    rust_ns: f64,
    c_ns: f64,
    budget: f64,
    is_write: bool,
}

/// The seven groups registered above. Read budget 1.05, write budget 1.10.
const RATIO_GROUPS: &[(&str, f64, bool)] = &[
    ("compare/read/point_random", 1.05, false),
    ("compare/read/point_zipf", 1.05, false),
    ("compare/read/seq_scan", 1.05, false),
    ("compare/read/range_scan", 1.05, false),
    ("compare/read/cursor_set_range", 1.05, false),
    ("compare/write/put_random", 1.10, true),
    ("compare/write/put_seq", 1.10, true),
];

fn print_ratio_summary() {
    let mut rows = Vec::new();
    let mut any_missing = false;
    for &(group, budget, is_write) in RATIO_GROUPS {
        let rust_ns = read_mean_ns(group, "rust");
        let c_ns = read_mean_ns(group, "c");
        match (rust_ns, c_ns) {
            (Some(r), Some(c)) => rows.push(RatioRow {
                group: group.to_string(),
                rust_ns: r,
                c_ns: c,
                budget,
                is_write,
            }),
            _ => {
                any_missing = true;
                eprintln!(
                    "warning: missing estimates for {group} (rust={}, c={})",
                    rust_ns.is_some(),
                    c_ns.is_some()
                );
            }
        }
    }

    if rows.is_empty() {
        eprintln!("no comparison results to summarise");
        return;
    }

    println!();
    println!("================ Rust vs C LMDB — Ratio Summary ================");
    println!(
        "{:<36} {:>12} {:>12} {:>8} {:>8}  status",
        "group", "rust (ns)", "c (ns)", "ratio", "budget"
    );
    let mut any_fail = false;
    for row in &rows {
        let ratio = row.rust_ns / row.c_ns;
        let ok = ratio <= row.budget;
        if !ok {
            any_fail = true;
        }
        println!(
            "{:<36} {:>12.0} {:>12.0} {:>8.3} {:>8.2}  {}",
            row.group,
            row.rust_ns,
            row.c_ns,
            ratio,
            row.budget,
            if ok { "OK" } else { "FAIL" }
        );
    }
    let kind = if rows.iter().any(|r| !r.is_write) {
        "reads"
    } else {
        "writes"
    };
    let _ = kind;
    println!("================================================================");

    if any_missing {
        eprintln!("note: ran with --quick or a subset; some groups have no estimates yet.");
    }
    if any_fail {
        eprintln!("\nat least one group exceeded its budget → exiting 1");
        std::process::exit(1);
    }
}

/// Read `$CARGO_TARGET_DIR/criterion/<group>/<side>/new/estimates.json` and
/// return `mean.point_estimate` in nanoseconds, or `None` if missing/malformed.
///
/// criterion sanitizes bench IDs by turning every non-alphanumeric char into
/// `_`, so `compare/read/point_random` becomes `compare_read_point_random`
/// on disk.
fn read_mean_ns(group: &str, side: &str) -> Option<f64> {
    let target = locate_target_dir()?;
    let sanitized: String = group
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let path = target
        .join("criterion")
        .join(sanitized)
        .join(side)
        .join("new")
        .join("estimates.json");
    let text = std::fs::read_to_string(&path).ok()?;
    // Cheap hand-rolled JSON lookup: find `"mean":{...,"point_estimate":N,...}`.
    let mean_idx = text.find("\"mean\"")?;
    let rest = &text[mean_idx..];
    let pe_idx = rest.find("\"point_estimate\"")?;
    let after = &rest[pe_idx + "\"point_estimate\"".len()..];
    let colon = after.find(':')?;
    let num_start = colon + 1;
    let num_end = after[num_start..]
        .find(|c: char| {
            c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E' && !c.is_ascii_digit()
        })
        .unwrap_or(after.len() - num_start);
    after[num_start..num_start + num_end].trim().parse().ok()
}

/// Find cargo's target dir by:
///   1. `$CARGO_TARGET_DIR` if set.
///   2. walking up from `current_exe()` until we find a dir named `target` or hit the root — this
///      handles the `build.target-dir` config-file case where the env var is empty.
///   3. falling back to `./target` relative to CWD.
fn locate_target_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return Some(p);
        }
    }
    // The bench binary lives at `<target>/release/deps/<name>`.
    if let Ok(exe) = std::env::current_exe() {
        // .../release/deps/bench_compare-HASH
        //              ^deps ^release    ^target
        if let Some(target) = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            if target.join("criterion").exists() {
                return Some(target.to_path_buf());
            }
        }
    }
    let fallback = PathBuf::from("target");
    if fallback.join("criterion").exists() {
        Some(fallback)
    } else {
        None
    }
}
