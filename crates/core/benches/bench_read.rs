//! Read microbenchmarks (R-01 .. R-08). R-09 (cold mmap) lives in P5.
//!
//! All groups share the same pre-loaded environment built once per bench
//! binary. The `iter` body opens a fresh RO transaction per sample, which
//! mirrors the realistic usage pattern where a reader spins up briefly to
//! fetch a value.

mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::{
    env::Environment,
    types::{CursorOp, MAIN_DBI, WriteFlags},
};

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, Workload, bulk_load, configure, fresh_tempdir,
    gen_workload, open_empty_env,
};

const POINT_N: usize = 1_000_000;
const KEY_SZ: usize = 16;
const VAL_SZ: usize = 100;

/// For the range-scan bench: number of cursor steps per sample.
const RANGE_STEPS: usize = 1_000;

/// Overflow workload: N=10K at 64 KiB per value.
const OVERFLOW_N: usize = 10_000;
const OVERFLOW_VAL_SZ: usize = 64 * 1024;

struct Fixture {
    _dir: tempfile::TempDir,
    env: Environment,
    workload: Workload,
}

fn build_fixture(prefix: &str, n: usize, key_sz: usize, val_sz: usize, dist: Dist) -> Fixture {
    let dir = fresh_tempdir(prefix);
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    let workload = gen_workload(BENCH_SEED, n, key_sz, val_sz, dist);
    bulk_load(&env, &workload);
    Fixture {
        _dir: dir,
        env,
        workload,
    }
}

// ---------------------------------------------------------------------------
// R-01 / R-02 / R-06 / R-07 / R-08 — point reads via txn.get or cursor.get
// ---------------------------------------------------------------------------

fn bench_point_random(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-r01-", POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);

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

fn bench_point_zipf(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-r02-", POINT_N, KEY_SZ, VAL_SZ, Dist::Zipf(0.99));

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

fn bench_cursor_set(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-r06-", POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);

    let mut group = c.benchmark_group("read/cursor_set");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &fix.workload.keys;
        let access = &fix.workload.access;
        let mut idx = 0usize;
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
            let k = &keys[access[idx % access.len()] as usize];
            let (_k, v) = cursor.get(Some(k), CursorOp::Set).expect("set");
            idx = idx.wrapping_add(1);
            black_box(v.len());
        });
    });

    group.finish();
}

fn bench_cursor_set_range(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-r07-", POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);

    let mut group = c.benchmark_group("read/cursor_set_range");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &fix.workload.keys;
        let access = &fix.workload.access;
        let mut idx = 0usize;
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
            let k = &keys[access[idx % access.len()] as usize];
            let (_k, v) = cursor.get(Some(k), CursorOp::SetRange).expect("set range");
            idx = idx.wrapping_add(1);
            black_box(v.len());
        });
    });

    group.finish();
}

fn bench_first_last(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-r08-", POINT_N, KEY_SZ, VAL_SZ, Dist::Sequential);

    let mut group = c.benchmark_group("read/first_last");
    group.throughput(Throughput::Elements(2));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
            let (_k1, v1) = cursor.get(None, CursorOp::First).expect("first");
            let first_len = v1.len();
            let (_k2, v2) = cursor.get(None, CursorOp::Last).expect("last");
            black_box((first_len, v2.len()));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// R-03 — full sequential scan
// ---------------------------------------------------------------------------

fn bench_seq_scan(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-r03-", POINT_N, KEY_SZ, VAL_SZ, Dist::Sequential);

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

// ---------------------------------------------------------------------------
// R-04 — range scan (positioned seek + 1000 nexts)
// ---------------------------------------------------------------------------

fn bench_range_scan(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-r04-", POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);

    let mut group = c.benchmark_group("read/range_scan");
    group.throughput(Throughput::Elements(RANGE_STEPS as u64));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &fix.workload.keys;
        let access = &fix.workload.access;
        let mut idx = 0usize;
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
            let start = &keys[access[idx % access.len()] as usize];
            idx = idx.wrapping_add(1);
            let (_k, v) = cursor.get(Some(start), CursorOp::SetRange).expect("seek");
            let mut acc = v.len() as u64;
            for _ in 1..RANGE_STEPS {
                match cursor.get(None, CursorOp::Next) {
                    Ok((k, v)) => acc += (k.len() + v.len()) as u64,
                    Err(_) => break,
                }
            }
            black_box(acc);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// R-05 — overflow page reads (k=16B, v=64KiB)
// ---------------------------------------------------------------------------

fn bench_overflow(c: &mut Criterion) {
    // Custom fixture: bulk_load in smaller batches because 10K × 64KiB = 640MiB.
    let dir = fresh_tempdir("lmdb-bench-r05-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    let workload = gen_workload(
        BENCH_SEED,
        OVERFLOW_N,
        KEY_SZ,
        OVERFLOW_VAL_SZ,
        Dist::Uniform,
    );
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
            txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
        }
        txn.commit().expect("commit");
    }

    let mut group = c.benchmark_group("read/overflow");
    group.throughput(Throughput::Bytes(OVERFLOW_VAL_SZ as u64));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &workload.keys;
        let access = &workload.access;
        let mut idx = 0usize;
        b.iter(|| {
            let txn = env.begin_ro_txn().expect("ro txn");
            let k = &keys[access[idx % access.len()] as usize];
            let v = txn.get(MAIN_DBI, k).expect("get");
            idx = idx.wrapping_add(1);
            black_box(v.len());
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

criterion_group! {
    name = benches;
    config = configure();
    targets =
        bench_point_random,
        bench_point_zipf,
        bench_seq_scan,
        bench_range_scan,
        bench_overflow,
        bench_cursor_set,
        bench_cursor_set_range,
        bench_first_last,
}
criterion_main!(benches);
