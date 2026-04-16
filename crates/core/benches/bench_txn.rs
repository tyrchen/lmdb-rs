//! Transaction-lifecycle microbenchmarks (T-01 .. T-04).

mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::types::{MAIN_DBI, WriteFlags};

use crate::common::{DEFAULT_MAP_SIZE, configure, fresh_tempdir, open_empty_env};

fn bench_ro_begin_commit(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-t01-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    // Put one key so the DB isn't empty — exercises the full begin path.
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"k", b"v", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    let mut group = c.benchmark_group("txn/ro_begin_commit");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            let txn = env.begin_ro_txn().expect("ro txn");
            black_box(&txn);
            drop(txn);
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

fn bench_rw_empty_commit(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-t02-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);

    let mut group = c.benchmark_group("txn/rw_empty_commit");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter(|| {
            let txn = env.begin_rw_txn().expect("rw txn");
            txn.commit().expect("commit");
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

fn bench_rw_single_put_commit(c: &mut Criterion) {
    // Each sample writes the same (single) key with its own txn. The commit
    // fsyncs → latency dominated by storage.
    let dir = fresh_tempdir("lmdb-bench-t03-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);

    let mut group = c.benchmark_group("txn/rw_single_put_commit");
    group.throughput(Throughput::Elements(1));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let mut i = 0u64;
        b.iter(|| {
            let mut txn = env.begin_rw_txn().expect("rw txn");
            let k = i.to_be_bytes();
            txn.put(MAIN_DBI, &k, b"v", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
            i = i.wrapping_add(1);
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

fn bench_nested(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-t04-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    // Seed a key so there's a non-trivial snapshot to take.
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"k", b"v", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    let mut group = c.benchmark_group("txn/nested");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let mut i = 0u64;
        b.iter(|| {
            let mut txn = env.begin_rw_txn().expect("rw txn");
            txn.begin_nested_txn().expect("begin nested");
            let k = i.to_be_bytes();
            txn.put(MAIN_DBI, &k, b"v", WriteFlags::empty())
                .expect("put");
            txn.commit_nested_txn().expect("commit nested");
            txn.commit().expect("commit outer");
            i = i.wrapping_add(1);
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
        bench_ro_begin_commit,
        bench_rw_empty_commit,
        bench_rw_single_put_commit,
        bench_nested,
}
criterion_main!(benches);
