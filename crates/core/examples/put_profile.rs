//! Micro-profile: put hot-path + commit breakdown.

use std::time::{Duration, Instant};

use lmdb_rs_core::{
    env::Environment,
    types::{EnvFlags, MAIN_DBI, WriteFlags},
};

const N: usize = 50_000;
const KEY_SZ: usize = 16;
const VAL_SZ: usize = 100;
const ITERS: usize = 12;
const WARM: usize = 2;

fn run_once(no_sync: bool) -> (Duration, Duration) {
    let keys: Vec<Vec<u8>> = (0..N)
        .map(|i| {
            let mut k = vec![0u8; KEY_SZ];
            k[..8].copy_from_slice(&(i as u64).to_be_bytes());
            k
        })
        .collect();
    let value = vec![0xABu8; VAL_SZ];

    let d = tempfile::Builder::new()
        .prefix("put-profile-iter-")
        .tempdir()
        .expect("d");
    let mut builder = Environment::builder();
    builder = builder.map_size(1 << 30);
    if no_sync {
        builder = builder.flags(EnvFlags::NO_SYNC);
    }
    let e = builder.open(d.path()).expect("env");

    let mut txn = e.begin_rw_txn().expect("txn");
    let t0 = Instant::now();
    for (k, v) in keys.iter().zip(std::iter::repeat(&value)) {
        txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
    }
    let put_elapsed = t0.elapsed();

    let t1 = Instant::now();
    txn.commit().expect("commit");
    let commit_elapsed = t1.elapsed();

    drop(e);
    drop(d);
    (put_elapsed, commit_elapsed)
}

fn main() {
    let mut sync_put = Duration::ZERO;
    let mut sync_commit = Duration::ZERO;
    let mut nosync_put = Duration::ZERO;
    let mut nosync_commit = Duration::ZERO;
    let mut samples = 0u32;

    for i in 0..ITERS {
        let (p, c) = run_once(false);
        let (np, nc) = run_once(true);
        if i >= WARM {
            sync_put += p;
            sync_commit += c;
            nosync_put += np;
            nosync_commit += nc;
            samples += 1;
        }
    }

    println!("avg over {samples} iters of {N} seq puts:");
    println!();
    println!("  DEFAULT FLAGS (fsync):");
    println!(
        "    puts:   {:?}  ({:.0} ns/put)",
        sync_put / samples,
        (sync_put / samples).as_nanos() as f64 / N as f64
    );
    println!("    commit: {:?}", sync_commit / samples);
    println!();
    println!("  NO_SYNC (no fsync):");
    println!(
        "    puts:   {:?}  ({:.0} ns/put)",
        nosync_put / samples,
        (nosync_put / samples).as_nanos() as f64 / N as f64
    );
    println!("    commit: {:?}", nosync_commit / samples);
    println!();
    println!(
        "  fsync overhead: {:?}",
        sync_commit / samples - nosync_commit / samples
    );
}
