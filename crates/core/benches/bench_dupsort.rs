//! DUPSORT / DUPFIXED microbenchmarks (D-01 .. D-03).

mod common;

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::{
    env::{Environment, EnvironmentBuilder},
    types::{CursorOp, DatabaseFlags, WriteFlags},
};

use crate::common::{DEFAULT_MAP_SIZE, configure, fresh_tempdir};

const DUP_KEYS: usize = 1_000;
const DUPS_PER_KEY: usize = 100;
const DUP_VAL_SZ: usize = 16;

const DUPFIXED_KEYS: usize = 200;
const DUPFIXED_DUPS: usize = 1_000;
const DUPFIXED_VAL_SZ: usize = 8;

fn open_env_with_named_db(prefix: &str) -> (tempfile::TempDir, Environment) {
    let dir = fresh_tempdir(prefix);
    let env = EnvironmentBuilder::new()
        .map_size(DEFAULT_MAP_SIZE)
        .max_dbs(4)
        .open(dir.path())
        .expect("open env");
    (dir, env)
}

// ---------------------------------------------------------------------------
// D-01 — put with DUP_SORT (10K × 100 dups × 16B)
// ---------------------------------------------------------------------------

fn bench_dupsort_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("dupsort/put");
    let total = (DUP_KEYS * DUPS_PER_KEY) as u64;
    group.throughput(Throughput::Elements(total));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(15));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || open_env_with_named_db("lmdb-bench-d01-"),
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                let dbi = txn
                    .open_db(
                        Some("dupdb"),
                        DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                    )
                    .expect("open_db");
                for i in 0..DUP_KEYS {
                    let k = (i as u64).to_be_bytes();
                    for j in 0..DUPS_PER_KEY {
                        let mut v = [0u8; DUP_VAL_SZ];
                        v[..8].copy_from_slice(&(j as u64).to_be_bytes());
                        v[8..].copy_from_slice(&(i as u64).to_be_bytes());
                        txn.put(dbi, &k, &v, WriteFlags::empty()).expect("put");
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
// D-02 — iterate all entries via cursor (Next traverses through dups)
// ---------------------------------------------------------------------------

fn bench_dupsort_iter(c: &mut Criterion) {
    // One-time fixture: load the DB, reuse for all samples.
    let (dir, env) = open_env_with_named_db("lmdb-bench-d02-");
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        let dbi = txn
            .open_db(
                Some("dupdb"),
                DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
            )
            .expect("open_db");
        for i in 0..DUP_KEYS {
            let k = (i as u64).to_be_bytes();
            for j in 0..DUPS_PER_KEY {
                let mut v = [0u8; DUP_VAL_SZ];
                v[..8].copy_from_slice(&(j as u64).to_be_bytes());
                v[8..].copy_from_slice(&(i as u64).to_be_bytes());
                txn.put(dbi, &k, &v, WriteFlags::empty()).expect("put");
            }
        }
        txn.commit().expect("commit");
    }

    let total = (DUP_KEYS * DUPS_PER_KEY) as u64;
    let mut group = c.benchmark_group("dupsort/iter");
    group.throughput(Throughput::Elements(total));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            let mut ro = env.begin_ro_txn().expect("ro txn");
            // RO txns must register the named DB handle locally.
            let dbi = ro.open_db(Some("dupdb")).expect("open_db ro");
            let mut cursor = ro.open_cursor(dbi).expect("cursor");
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
    drop(env);
    drop(dir);
}

// ---------------------------------------------------------------------------
// D-03 — dupfixed with DUP_FIXED (1K × 1K × 8B)
// ---------------------------------------------------------------------------

fn bench_dupfixed_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("dupsort/dupfixed_put");
    let total = (DUPFIXED_KEYS * DUPFIXED_DUPS) as u64;
    group.throughput(Throughput::Elements(total));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(15));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || open_env_with_named_db("lmdb-bench-d03-"),
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                let dbi = txn
                    .open_db(
                        Some("dupfixed"),
                        DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED,
                    )
                    .expect("open_db");
                for i in 0..DUPFIXED_KEYS {
                    let k = (i as u64).to_be_bytes();
                    for j in 0..DUPFIXED_DUPS {
                        let v = (j as u64).to_be_bytes();
                        assert_eq!(v.len(), DUPFIXED_VAL_SZ);
                        txn.put(dbi, &k, &v, WriteFlags::empty()).expect("put");
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
    targets = bench_dupsort_put, bench_dupsort_iter, bench_dupfixed_put,
}
criterion_main!(benches);
