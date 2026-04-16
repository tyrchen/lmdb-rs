//! Read microbenchmarks (R-01..R-08; R-09 lives in P5).
//!
//! All groups use a single pre-loaded environment. The `iter` body opens a
//! fresh RO transaction per sample (mirrors the realistic usage pattern
//! where a reader spins up briefly to fetch a value).

mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::{
    env::Environment,
    types::{CursorOp, MAIN_DBI},
};

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, Workload, bulk_load, configure, fresh_tempdir,
    gen_workload, open_empty_env,
};

/// Shared corpus size for point-read benches.
const POINT_N: usize = 1_000_000;
/// Key/value shape.
const KEY_SZ: usize = 16;
const VAL_SZ: usize = 100;

/// Build the read fixture used by every R-* group. Held for the lifetime of
/// the bench binary so the OS page cache stays warm across groups.
struct Fixture {
    _dir: tempfile::TempDir,
    env: Environment,
    workload: Workload,
}

fn build_fixture(n: usize, dist: Dist) -> Fixture {
    let dir = fresh_tempdir("lmdb-bench-read-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    let workload = gen_workload(BENCH_SEED, n, KEY_SZ, VAL_SZ, dist);
    bulk_load(&env, &workload);
    Fixture {
        _dir: dir,
        env,
        workload,
    }
}

// ---------------------------------------------------------------------------
// R-01 — point reads, uniform random access
// ---------------------------------------------------------------------------

fn bench_point_random(c: &mut Criterion) {
    let fix = build_fixture(POINT_N, Dist::Uniform);

    let mut group = c.benchmark_group("read/point_random");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &fix.workload.keys;
        let access = &fix.workload.access;
        let mut idx = 0usize;
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let k = &keys[access[idx % access.len()] as usize];
            let v = txn.get(MAIN_DBI, k).expect("get");
            idx = idx.wrapping_add(1);
            black_box(v.len());
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// R-02 .. R-08 stubs (filled in P2)
// ---------------------------------------------------------------------------

fn bench_point_zipf(c: &mut Criterion) {
    let fix = build_fixture(POINT_N, Dist::Zipf(0.99));

    let mut group = c.benchmark_group("read/point_zipf");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &fix.workload.keys;
        let access = &fix.workload.access;
        let mut idx = 0usize;
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let k = &keys[access[idx % access.len()] as usize];
            let v = txn.get(MAIN_DBI, k).expect("get");
            idx = idx.wrapping_add(1);
            black_box(v.len());
        });
    });

    group.finish();
}

fn bench_seq_scan(c: &mut Criterion) {
    let fix = build_fixture(POINT_N, Dist::Sequential);

    let mut group = c.benchmark_group("read/seq_scan");
    group.throughput(Throughput::Elements(POINT_N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
            let mut count = 0u64;
            let mut bytes = 0u64;
            let mut op = CursorOp::First;
            while let Ok((k, v)) = cursor.get(None, op) {
                count += 1;
                bytes += (k.len() + v.len()) as u64;
                op = CursorOp::Next;
            }
            black_box((count, bytes));
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_point_random, bench_point_zipf, bench_seq_scan
}
criterion_main!(benches);
