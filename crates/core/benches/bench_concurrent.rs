//! Concurrency microbenchmarks (C-01, C-02).
//!
//! Threads share an `Arc<Environment>` and each one spins on a tight
//! read loop. We measure aggregate throughput across `THREADS` workers:
//! the `iter` body is one "batch" of `OPS_PER_BATCH` reads spread across
//! the workers. Criterion's `Throughput::Elements` converts that into
//! ops/sec.
//!
//! Threads are pinned to distinct cores when `core_affinity` can list
//! them, which reduces run-to-run variance on modern laptops with
//! asymmetric P/E cores.

mod common;

use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lmdb_rs_core::types::{MAIN_DBI, WriteFlags};

use crate::common::{
    BENCH_SEED, DEFAULT_MAP_SIZE, Dist, bulk_load, configure, fresh_tempdir, gen_workload,
    open_empty_env,
};

const POINT_N: usize = 1_000_000;
const KEY_SZ: usize = 16;
const VAL_SZ: usize = 100;

/// Number of reader threads for C-01 / C-02.
const THREADS: usize = 4;
/// Reads per batch per thread. Criterion times the whole batch.
const OPS_PER_THREAD: usize = 1_000;

// ---------------------------------------------------------------------------
// C-01 — M concurrent readers
// ---------------------------------------------------------------------------

fn bench_concurrent_readers(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-c01-");
    let env = Arc::new(open_empty_env(dir.path(), DEFAULT_MAP_SIZE));
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    bulk_load(&env, &workload);

    let keys = Arc::new(workload.keys);
    let access = Arc::new(workload.access);
    let cores = core_affinity::get_core_ids().unwrap_or_default();

    let mut group = c.benchmark_group("concurrent/readers");
    group.throughput(Throughput::Elements((THREADS * OPS_PER_THREAD) as u64));
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter(THREADS), |b| {
        b.iter(|| {
            let mut handles = Vec::with_capacity(THREADS);
            let start_offset = Arc::new(AtomicUsize::new(0));
            for t in 0..THREADS {
                let env = Arc::clone(&env);
                let keys = Arc::clone(&keys);
                let access = Arc::clone(&access);
                let core = cores.get(t).copied();
                let start_offset = Arc::clone(&start_offset);
                handles.push(thread::spawn(move || {
                    if let Some(c) = core {
                        core_affinity::set_for_current(c);
                    }
                    let base = start_offset.fetch_add(OPS_PER_THREAD, Ordering::Relaxed);
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = access[(base + i) % access.len()] as usize;
                        let txn = env.begin_ro_txn().expect("ro txn");
                        let v = txn.get(MAIN_DBI, &keys[idx]).expect("get");
                        sum += v.len() as u64;
                    }
                    sum
                }));
            }
            let mut total = 0u64;
            for h in handles {
                total = total.wrapping_add(h.join().expect("thread join"));
            }
            black_box(total);
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

// ---------------------------------------------------------------------------
// C-02 — 1 writer + M readers
// ---------------------------------------------------------------------------

fn bench_one_writer_m_readers(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-c02-");
    let env = Arc::new(open_empty_env(dir.path(), DEFAULT_MAP_SIZE));
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    bulk_load(&env, &workload);

    let keys = Arc::new(workload.keys);
    let access = Arc::new(workload.access);
    let cores = core_affinity::get_core_ids().unwrap_or_default();

    let mut group = c.benchmark_group("concurrent/1w_Mr");
    group.throughput(Throughput::Elements((THREADS * OPS_PER_THREAD) as u64));
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function(BenchmarkId::from_parameter(THREADS), |b| {
        b.iter(|| {
            let start_offset = Arc::new(AtomicUsize::new(0));
            let mut reader_handles = Vec::with_capacity(THREADS);
            for t in 0..THREADS {
                let env = Arc::clone(&env);
                let keys = Arc::clone(&keys);
                let access = Arc::clone(&access);
                let core = cores.get(t).copied();
                let start_offset = Arc::clone(&start_offset);
                reader_handles.push(thread::spawn(move || {
                    if let Some(c) = core {
                        core_affinity::set_for_current(c);
                    }
                    let base = start_offset.fetch_add(OPS_PER_THREAD, Ordering::Relaxed);
                    let mut sum = 0u64;
                    for i in 0..OPS_PER_THREAD {
                        let idx = access[(base + i) % access.len()] as usize;
                        let txn = env.begin_ro_txn().expect("ro txn");
                        let v = txn.get(MAIN_DBI, &keys[idx]).expect("get");
                        sum += v.len() as u64;
                    }
                    sum
                }));
            }

            // Writer: one txn, updating a small set of keys. Pinned to a
            // remaining core if available.
            let writer_core = cores.get(THREADS).copied();
            let env_w = Arc::clone(&env);
            let keys_w = Arc::clone(&keys);
            let writer = thread::spawn(move || {
                if let Some(c) = writer_core {
                    core_affinity::set_for_current(c);
                }
                let mut txn = env_w.begin_rw_txn().expect("rw txn");
                // Write a small sub-range so writer doesn't dominate time.
                for k in keys_w.iter().take(512) {
                    txn.put(MAIN_DBI, k, b"xx", WriteFlags::empty())
                        .expect("put");
                }
                txn.commit().expect("commit");
            });

            let mut total = 0u64;
            for h in reader_handles {
                total = total.wrapping_add(h.join().expect("reader join"));
            }
            writer.join().expect("writer join");
            black_box(total);
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

// ---------------------------------------------------------------------------
// R-09 — cold mmap (point reads after flushing OS page cache)
// ---------------------------------------------------------------------------
//
// On macOS `sudo purge` flushes the cache. On Linux `echo 3 >
// /proc/sys/vm/drop_caches` works. Both need elevated privileges — if we
// can't drop caches, the bench reports that cleanly and skips. This keeps
// developers from seeing spurious numbers; CI runners that have the right
// sudoers config get the real cold-read measurement.

fn bench_cold_mmap(c: &mut Criterion) {
    let dir = fresh_tempdir("lmdb-bench-r09-");
    let env = open_empty_env(dir.path(), DEFAULT_MAP_SIZE);
    let workload = gen_workload(BENCH_SEED, POINT_N, KEY_SZ, VAL_SZ, Dist::Uniform);
    bulk_load(&env, &workload);

    if !can_drop_caches() {
        eprintln!(
            "skipping read/cold_mmap — cache-drop helper unavailable on this platform / user"
        );
        return;
    }

    let mut group = c.benchmark_group("read/cold_mmap");
    group.throughput(Throughput::Elements(1));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function(BenchmarkId::from_parameter("lmdb-rs-core"), |b| {
        let keys = &workload.keys;
        let access = &workload.access;
        let mut idx = 0usize;
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                drop_caches();
                let k = &keys[access[idx % access.len()] as usize];
                idx = idx.wrapping_add(1);
                let start = std::time::Instant::now();
                let txn = env.begin_ro_txn().expect("ro txn");
                let v = txn.get(MAIN_DBI, k).expect("get");
                total += start.elapsed();
                black_box(v.len());
            }
            total
        });
    });

    group.finish();
    drop(env);
    drop(dir);
}

// ---------------------------------------------------------------------------
// Cache-drop helpers — platform-specific, best-effort.
// ---------------------------------------------------------------------------

fn can_drop_caches() -> bool {
    #[cfg(target_os = "linux")]
    {
        // We can only drop caches if /proc/sys/vm/drop_caches is writable.
        std::fs::OpenOptions::new()
            .write(true)
            .open("/proc/sys/vm/drop_caches")
            .is_ok()
    }
    #[cfg(target_os = "macos")]
    {
        // On macOS, `purge` exists but only works as root. Try a no-op
        // invocation; if it exits 0 we can use it in the loop.
        match std::process::Command::new("/usr/sbin/purge").status() {
            Ok(s) => s.success(),
            Err(_) => false,
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        false
    }
}

fn drop_caches() {
    #[cfg(target_os = "linux")]
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .write(true)
            .open("/proc/sys/vm/drop_caches")
        {
            let _ = f.write_all(b"3\n");
        }
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("/usr/sbin/purge").status();
    }
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_concurrent_readers, bench_one_writer_m_readers, bench_cold_mmap,
}
criterion_main!(benches);
