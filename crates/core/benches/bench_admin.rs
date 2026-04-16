//! Admin / maintenance microbenchmarks (A-01 .. A-04).

mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::types::MAIN_DBI;

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, bulk_load, configure, configure_slow, fresh_tempdir,
    gen_workload, open_empty_env,
};

const ADMIN_N: usize = 100_000;

// ---------------------------------------------------------------------------
// A-01 — open an existing env (cache warm)
// ---------------------------------------------------------------------------

fn bench_admin_open_warm(c: &mut Criterion) {
    // Pre-create a DB, then time opening it again.
    let dir = fresh_tempdir("lmdb-bench-a01-");
    {
        let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
        let workload = gen_workload(BENCH_SEED, ADMIN_N, 16, 100, Dist::Sequential);
        bulk_load(&env, &workload);
        drop(env);
    }

    let mut group = c.benchmark_group("admin/open_warm");
    group.throughput(Throughput::Elements(1));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
            black_box(&env);
            drop(env);
        });
    });

    group.finish();
    drop(dir);
}

// ---------------------------------------------------------------------------
// A-02 — env.copy
// ---------------------------------------------------------------------------

fn bench_admin_copy(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-a02-src-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    let workload = gen_workload(BENCH_SEED, ADMIN_N, 16, 100, Dist::Sequential);
    bulk_load(&env, &workload);

    let mut group = c.benchmark_group("admin/copy");
    group.throughput(Throughput::Elements(ADMIN_N as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || fresh_tempdir("lmdb-bench-a02-dst-"),
            |dst| {
                let file = dst.path().join("data.mdb");
                env.copy(&file).expect("copy");
                black_box(dst);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.finish();
    drop(env);
    drop(dir);
}

// ---------------------------------------------------------------------------
// A-03 — env.copy_compact
// ---------------------------------------------------------------------------

fn bench_admin_copy_compact(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-a03-src-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    let workload = gen_workload(BENCH_SEED, ADMIN_N, 16, 100, Dist::Sequential);
    bulk_load(&env, &workload);

    let mut group = c.benchmark_group("admin/copy_compact");
    group.throughput(Throughput::Elements(ADMIN_N as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || fresh_tempdir("lmdb-bench-a03-dst-"),
            |dst| {
                let file = dst.path().join("data.mdb");
                env.copy_compact(&file).expect("copy_compact");
                black_box(dst);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.finish();
    drop(env);
    drop(dir);
}

// ---------------------------------------------------------------------------
// A-04 — env.sync(true)
// ---------------------------------------------------------------------------

fn bench_admin_sync_force(c: &mut Criterion) {
    // Need at least one committed txn so there's something to sync.
    let dir = fresh_tempdir("lmdb-bench-a04-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    {
        use lmdb_rs_core::types::WriteFlags;
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"k", b"v", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    let mut group = c.benchmark_group("admin/sync_force");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            env.sync(true).expect("sync");
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_admin_open_warm, bench_admin_sync_force,
}

criterion_group! {
    name = slow_benches;
    config = configure_slow();
    targets = bench_admin_copy, bench_admin_copy_compact,
}

criterion_main!(benches, slow_benches);
