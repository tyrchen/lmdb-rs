//! Cursor-specific microbenchmarks.
//!
//! Most cursor ops are already covered by bench_read (R-06..R-08). This file
//! zooms in on patterns that are hard to express through point reads:
//!
//! * iter / iter_from (the Rust iterator adapter over a cursor).
//! * RwCursor put + del_current (mutation via cursor).

mod common;

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::{
    env::Environment,
    types::{CursorOp, MAIN_DBI, WriteFlags},
};

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, Workload, bulk_load, configure, fresh_tempdir,
    gen_workload, open_empty_env,
};

const POINT_N: usize = 1_000_000;
const MUT_N: usize = 100_000;
const KEY_SZ: usize = 16;
const VAL_SZ: usize = 100;

struct Fixture {
    _dir: tempfile::TempDir,
    env: Environment,
    workload: Workload,
}

fn build_fixture(prefix: &str, n: usize) -> Fixture {
    let dir = fresh_tempdir(prefix);
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    let workload = gen_workload(BENCH_SEED, n, KEY_SZ, VAL_SZ, Dist::Sequential);
    bulk_load(&env, &workload);
    Fixture {
        _dir: dir,
        env,
        workload,
    }
}

// ---------------------------------------------------------------------------
// cursor/iter — the `Iterator for CursorIter` adapter
// ---------------------------------------------------------------------------

fn bench_cursor_iter(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-iter-", POINT_N);

    let mut group = c.benchmark_group("cursor/iter");
    group.throughput(Throughput::Elements(POINT_N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
            let mut sum = 0u64;
            for item in cursor.iter() {
                let (_k, v) = item.expect("iter");
                sum += v.len() as u64;
            }
            black_box(sum);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// cursor/iter_from — positioned seek then iterate
// ---------------------------------------------------------------------------

fn bench_cursor_iter_from(c: &mut Criterion) {
    let fix = build_fixture("lmdb-bench-iterfrom-", POINT_N);

    let mut group = c.benchmark_group("cursor/iter_from");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &fix.workload.keys;
        let mut idx = 0usize;
        b.iter(|| {
            let txn = fix.env.begin_ro_txn().expect("ro txn");
            let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
            // Seek to a key deterministically but spread across the keyspace.
            let pos = (idx * 7919) % keys.len();
            idx = idx.wrapping_add(1);
            let mut iter = cursor.iter_from(&keys[pos]).expect("iter_from");
            // Take exactly one element so the work is bounded.
            let item = iter.next().expect("one").expect("ok");
            black_box(item.1.len());
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// cursor/rw_put — RwCursor::put
// ---------------------------------------------------------------------------

fn bench_rw_cursor_put(c: &mut Criterion) {
    let workload = gen_workload(BENCH_SEED, MUT_N, KEY_SZ, VAL_SZ, Dist::Sequential);

    let mut group = c.benchmark_group("cursor/rw_put");
    group.throughput(Throughput::Elements(MUT_N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-rwput-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                (dir, env)
            },
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                {
                    let mut cursor = txn.open_rw_cursor(MAIN_DBI).expect("rw cursor");
                    for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                        cursor.put(k, v, WriteFlags::empty()).expect("cursor put");
                    }
                }
                txn.commit().expect("commit");
                black_box((dir, env));
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// cursor/rw_del_current — RwCursor::del_current walking forward
// ---------------------------------------------------------------------------

fn bench_rw_cursor_del_current(c: &mut Criterion) {
    let workload = gen_workload(BENCH_SEED, MUT_N, KEY_SZ, VAL_SZ, Dist::Sequential);

    let mut group = c.benchmark_group("cursor/rw_del_current");
    group.throughput(Throughput::Elements(MUT_N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-delcur-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                bulk_load(&env, &workload);
                (dir, env)
            },
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                {
                    let mut cursor = txn.open_rw_cursor(MAIN_DBI).expect("rw cursor");
                    // Re-seek to First after each del_current: LMDB's cursor
                    // position after a delete is implementation-defined, so
                    // always reposition to get deterministic traversal.
                    while cursor.get(None, CursorOp::First).is_ok() {
                        cursor.del_current().expect("del_current");
                    }
                }
                txn.commit().expect("commit");
                black_box((dir, env));
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure();
    targets =
        bench_cursor_iter,
        bench_cursor_iter_from,
        bench_rw_cursor_put,
        bench_rw_cursor_del_current,
}
criterion_main!(benches);
