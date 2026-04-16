//! Write microbenchmarks (W-01 .. W-10).
//!
//! Writes are destructive — each sample must start from the same initial
//! state. We use `iter_batched` with `PerIteration` setup for anything that
//! mutates state; setup cost is excluded from timing.

mod common;

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::types::{EnvFlags, MAIN_DBI, WriteFlags};

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, Workload, configure, configure_slow, fresh_tempdir,
    gen_workload, open_empty_env, open_env_with_flags,
};

const N: usize = 100_000;
const SYNC_N: usize = 10_000; // sync writes are much slower → smaller N.
const RESERVE_N: usize = 10_000; // reserve returns &mut [u8] into a dirty page,
// which becomes invalid once the page spills.
const KEY_SZ: usize = 16;
const VAL_SZ: usize = 100;
const RESERVE_VAL_SZ: usize = 1024;
const OVERFLOW_N: usize = 10_000;
const OVERFLOW_VAL_SZ: usize = 64 * 1024;

fn shuffle_workload(w: &mut Workload) {
    use rand::{SeedableRng, seq::SliceRandom};
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(BENCH_SEED ^ 0xABCD_EF01);
    // Zip (key, value) then shuffle in lockstep so access pattern is random.
    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> =
        w.keys.drain(..).zip(w.values.drain(..)).collect::<Vec<_>>();
    pairs.shuffle(&mut rng);
    for (k, v) in pairs {
        w.keys.push(k);
        w.values.push(v);
    }
}

// ---------------------------------------------------------------------------
// W-01 — random put, single commit per txn
// ---------------------------------------------------------------------------

fn bench_put_random(c: &mut Criterion) {
    let mut workload = gen_workload(BENCH_SEED, N, KEY_SZ, VAL_SZ, Dist::Uniform);
    shuffle_workload(&mut workload);

    let mut group = c.benchmark_group("write/put_random");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w01-");
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

    group.finish();
}

// ---------------------------------------------------------------------------
// W-02 — synchronous: one commit per key (default flags → each commit fsyncs)
// ---------------------------------------------------------------------------

fn bench_put_random_sync(c: &mut Criterion) {
    let mut workload = gen_workload(BENCH_SEED, SYNC_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    shuffle_workload(&mut workload);

    let mut group = c.benchmark_group("write/put_random_sync");
    group.throughput(Throughput::Elements(SYNC_N as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w02-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                (dir, env)
            },
            |(dir, env)| {
                for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                    let mut txn = env.begin_rw_txn().expect("rw txn");
                    txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
                    txn.commit().expect("commit");
                }
                black_box((dir, env));
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// W-03 — sequential put (sorted insertion order)
// ---------------------------------------------------------------------------

fn bench_put_seq(c: &mut Criterion) {
    // Keys generated in natural order — already sorted.
    let workload = gen_workload(BENCH_SEED, N, KEY_SZ, VAL_SZ, Dist::Sequential);

    let mut group = c.benchmark_group("write/put_seq");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w03-");
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

    group.finish();
}

// ---------------------------------------------------------------------------
// W-04 — APPEND flag (sorted input, fast-path in btree)
// ---------------------------------------------------------------------------

fn bench_append(c: &mut Criterion) {
    let workload = gen_workload(BENCH_SEED, N, KEY_SZ, VAL_SZ, Dist::Sequential);

    let mut group = c.benchmark_group("write/append");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w04-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                (dir, env)
            },
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                    txn.put(MAIN_DBI, k, v, WriteFlags::APPEND).expect("append");
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
// W-05 — reserve + in-place fill (1KiB values)
// ---------------------------------------------------------------------------

fn bench_reserve(c: &mut Criterion) {
    // Keys random, values don't matter — we reserve+fill zero bytes.
    let mut workload = gen_workload(BENCH_SEED, RESERVE_N, KEY_SZ, 0, Dist::Uniform);
    shuffle_workload(&mut workload);

    let mut group = c.benchmark_group("write/reserve");
    group.throughput(Throughput::Elements(RESERVE_N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w05-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                (dir, env)
            },
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                for k in &workload.keys {
                    let buf = txn.reserve(MAIN_DBI, k, RESERVE_VAL_SZ).expect("reserve");
                    // Touch first & last byte so the compiler can't skip the fill.
                    buf[0] = k[0];
                    buf[RESERVE_VAL_SZ - 1] = k[0];
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
// W-06 — overwrite existing keys
// ---------------------------------------------------------------------------

fn bench_overwrite(c: &mut Criterion) {
    let mut workload = gen_workload(BENCH_SEED, N, KEY_SZ, VAL_SZ, Dist::Uniform);
    shuffle_workload(&mut workload);

    let mut group = c.benchmark_group("write/overwrite");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w06-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                // Prepopulate with the same keys.
                let mut txn = env.begin_rw_txn().expect("rw txn");
                for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                    txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
                }
                txn.commit().expect("commit");
                (dir, env)
            },
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                // Overwrite with same values (still triggers CoW of leaf pages).
                for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                    txn.put(MAIN_DBI, k, v, WriteFlags::empty())
                        .expect("overwrite");
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
// W-07 — delete
// ---------------------------------------------------------------------------

fn bench_delete(c: &mut Criterion) {
    let mut workload = gen_workload(BENCH_SEED, N, KEY_SZ, VAL_SZ, Dist::Uniform);
    shuffle_workload(&mut workload);

    let mut group = c.benchmark_group("write/delete");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w07-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                let mut txn = env.begin_rw_txn().expect("rw txn");
                for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                    txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
                }
                txn.commit().expect("commit");
                (dir, env)
            },
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                for k in &workload.keys {
                    txn.del(MAIN_DBI, k, None).expect("del");
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
// W-08 — overflow-page put (64 KiB values)
// ---------------------------------------------------------------------------

fn bench_overflow_put(c: &mut Criterion) {
    let workload = gen_workload(
        BENCH_SEED,
        OVERFLOW_N,
        KEY_SZ,
        OVERFLOW_VAL_SZ,
        Dist::Sequential,
    );

    let mut group = c.benchmark_group("write/overflow_put");
    group.throughput(Throughput::Bytes((OVERFLOW_N * OVERFLOW_VAL_SZ) as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w08-");
                let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
                (dir, env)
            },
            |(dir, env)| {
                let mut txn = env.begin_rw_txn().expect("rw txn");
                for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                    txn.put(MAIN_DBI, k, v, WriteFlags::empty())
                        .expect("put overflow");
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
// W-09 — put_random with WRITE_MAP
// ---------------------------------------------------------------------------

fn bench_writemap(c: &mut Criterion) {
    let mut workload = gen_workload(BENCH_SEED, N, KEY_SZ, VAL_SZ, Dist::Uniform);
    shuffle_workload(&mut workload);

    let mut group = c.benchmark_group("write/writemap");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w09-");
                let env = open_env_with_flags(dir.path(), DEFAULT_MAP_SIZE, EnvFlags::WRITE_MAP);
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

    group.finish();
}

// ---------------------------------------------------------------------------
// W-10 — NO_SYNC: commit-per-key but no fsync
// ---------------------------------------------------------------------------

fn bench_nosync(c: &mut Criterion) {
    let mut workload = gen_workload(BENCH_SEED, N, KEY_SZ, VAL_SZ, Dist::Uniform);
    shuffle_workload(&mut workload);

    let mut group = c.benchmark_group("write/nosync");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        b.iter_batched(
            || {
                let dir = fresh_tempdir("lmdb-bench-w10-");
                let env = open_env_with_flags(dir.path(), DEFAULT_MAP_SIZE, EnvFlags::NO_SYNC);
                (dir, env)
            },
            |(dir, env)| {
                for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
                    let mut txn = env.begin_rw_txn().expect("rw txn");
                    txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
                    txn.commit().expect("commit");
                }
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
        bench_put_random,
        bench_put_seq,
        bench_append,
        bench_reserve,
        bench_overwrite,
        bench_delete,
        bench_writemap,
}

criterion_group! {
    name = slow_benches;
    config = configure_slow();
    targets =
        bench_put_random_sync,
        bench_overflow_put,
        bench_nosync,
}

criterion_main!(benches, slow_benches);
