//! Scaling sweeps (S-01 .. S-04): how does latency change with value size,
//! key size, and total DB size?

mod common;

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::types::{MAIN_DBI, WriteFlags};

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, bulk_load, configure_slow, fresh_tempdir, gen_workload,
    open_empty_env,
};

const VALUE_SIZES: &[usize] = &[32, 128, 512, 2048, 8192, 32768, 131_072];
const KEY_SIZES: &[usize] = &[8, 32, 128, 256, 511];
const DB_SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

const SCALE_N: usize = 100_000;

// ---------------------------------------------------------------------------
// S-01 — read latency vs value size
// ---------------------------------------------------------------------------

fn bench_scale_value_size_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale/value_size/read");
    group.throughput(Throughput::Elements(1));

    for &v in VALUE_SIZES {
        // 128KB × 100K = 12.8 GB — too big for the 8 GiB default map.
        let n = if v >= 32_768 { 10_000 } else { SCALE_N };
        let dir = fresh_tempdir("lmdb-bench-s01-");
        let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
        let workload = gen_workload(BENCH_SEED, n, 16, v, Dist::Uniform);
        bulk_load(&env, &workload);

        group.bench_with_input(BenchmarkId::from_parameter(v), &v, |b, _| {
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

        drop(env);
        drop(dir);
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// S-02 — write latency vs value size (one commit per txn)
// ---------------------------------------------------------------------------

fn bench_scale_value_size_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale/value_size/write");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(15));

    for &v in VALUE_SIZES {
        let n = if v >= 32_768 { 2_000 } else { 20_000 };
        let workload = gen_workload(BENCH_SEED, n, 16, v, Dist::Uniform);
        group.throughput(Throughput::Bytes((n * v) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(v), &v, |b, _| {
            b.iter_batched(
                || {
                    let dir = fresh_tempdir("lmdb-bench-s02-");
                    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                    (dir, env)
                },
                |(dir, env)| {
                    let mut txn = env.begin_rw_txn().expect("rw txn");
                    for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                        txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
                    }
                    txn.commit().expect("commit");
                    black_box((dir, env));
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// S-03 — latency vs key size (fixed v=100B)
// ---------------------------------------------------------------------------

fn bench_scale_key_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale/key_size");
    group.throughput(Throughput::Elements(1));

    for &k in KEY_SIZES {
        let dir = fresh_tempdir("lmdb-bench-s03-");
        let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
        let workload = gen_workload(BENCH_SEED, SCALE_N, k, 100, Dist::Uniform);
        bulk_load(&env, &workload);

        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            let keys = &workload.keys;
            let access = &workload.access;
            let mut idx = 0usize;
            b.iter(|| {
                let txn = env.begin_ro_txn().expect("ro txn");
                let key = &keys[access[idx % access.len()] as usize];
                let v = txn.get(MAIN_DBI, key).expect("get");
                idx = idx.wrapping_add(1);
                black_box(v.len());
            });
        });

        drop(env);
        drop(dir);
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// S-04 — read latency vs DB size
// ---------------------------------------------------------------------------

fn bench_scale_db_size_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale/db_size/read");
    group.throughput(Throughput::Elements(1));

    for &n in DB_SIZES {
        let dir = fresh_tempdir("lmdb-bench-s04-");
        let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
        let workload = gen_workload(BENCH_SEED, n, 16, 100, Dist::Uniform);
        bulk_load(&env, &workload);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let keys = &workload.keys;
            let access = &workload.access;
            let mut idx = 0usize;
            b.iter(|| {
                let txn = env.begin_ro_txn().expect("ro txn");
                let key = &keys[access[idx % access.len()] as usize];
                let v = txn.get(MAIN_DBI, key).expect("get");
                idx = idx.wrapping_add(1);
                black_box(v.len());
            });
        });

        drop(env);
        drop(dir);
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_slow();
    targets =
        bench_scale_value_size_read,
        bench_scale_value_size_write,
        bench_scale_key_size,
        bench_scale_db_size_read,
}
criterion_main!(benches);
