//! Shared benchmark harness: workload generators, env setup, config.
//!
//! Everything here is deterministic — seeded with `BENCH_SEED` so that two
//! runs on the same machine produce identical workloads. Variance in results
//! should therefore be dominated by system noise, not input.

#![allow(dead_code)]

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use criterion::Criterion;
use lmdb_rs_core::{
    env::{Environment, EnvironmentBuilder},
    types::{EnvFlags, MAIN_DBI, WriteFlags},
};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Zipf};
use tempfile::TempDir;

/// Master RNG seed. Change only if you explicitly want to invalidate baselines.
pub const BENCH_SEED: u64 = 0x_C0FFEE_u64;

/// Default map size: large enough for the biggest bench workload (10M × 128 B).
pub const DEFAULT_MAP_SIZE: usize = 8 * 1024 * 1024 * 1024; // 8 GiB

// ---------------------------------------------------------------------------
// Criterion configuration
// ---------------------------------------------------------------------------

/// Build a Criterion instance with bench-suite defaults.
///
/// Fast groups can use this as-is. Slow groups should override with smaller
/// sample sizes and longer measurement times (see `configure_slow`).
pub fn configure() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
        .noise_threshold(0.02)
        .significance_level(0.05)
        .confidence_level(0.99)
}

/// Config for slow benches (large N or expensive setup).
pub fn configure_slow() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(10))
        .sample_size(20)
        .noise_threshold(0.05)
        .significance_level(0.05)
        .confidence_level(0.95)
}

// ---------------------------------------------------------------------------
// Workload generation
// ---------------------------------------------------------------------------

/// Access-pattern distribution for the `access` index sequence.
#[derive(Debug, Clone, Copy)]
pub enum Dist {
    Uniform,
    Zipf(f64),
    Sequential,
}

/// A deterministic synthetic workload.
///
/// * `keys` — all unique keys, sorted lexicographically (good for APPEND).
/// * `values` — one value per key slot, same length for every entry.
/// * `access` — indices into `keys`, drawn from the chosen distribution.
#[derive(Debug)]
pub struct Workload {
    pub keys: Vec<Vec<u8>>,
    pub values: Vec<Vec<u8>>,
    pub access: Vec<u32>,
}

impl Workload {
    /// Return the byte span of a single (key, value) pair. Useful for
    /// `Throughput::Bytes` reporting.
    pub fn pair_bytes(&self) -> u64 {
        let k = self.keys.first().map_or(0, Vec::len);
        let v = self.values.first().map_or(0, Vec::len);
        (k + v) as u64
    }
}

/// Generate a workload.
///
/// Keys are fixed-width, lexicographically sortable (big-endian encoded index
/// padded to `key_sz` bytes). Values are pseudo-random bytes of exactly
/// `val_sz` bytes each.
///
/// `access` is a length-`n` sequence of indices in `[0, n)` drawn from `dist`.
pub fn gen_workload(seed: u64, n: usize, key_sz: usize, val_sz: usize, dist: Dist) -> Workload {
    assert!(key_sz >= 8, "key_sz must be ≥ 8 to hold a u64 index");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let keys = (0..n)
        .map(|i| {
            let mut k = vec![0u8; key_sz];
            k[..8].copy_from_slice(&(i as u64).to_be_bytes());
            k
        })
        .collect::<Vec<_>>();

    let values = (0..n)
        .map(|_| {
            let mut v = vec![0u8; val_sz];
            rng.fill(v.as_mut_slice());
            v
        })
        .collect::<Vec<_>>();

    let access: Vec<u32> = match dist {
        Dist::Uniform => {
            let mut v: Vec<u32> = (0..n as u32).collect();
            v.shuffle(&mut rng);
            v
        }
        Dist::Zipf(theta) => {
            let zipf = Zipf::new(n as f64, theta).expect("valid zipf params");
            (0..n)
                .map(|_| {
                    (zipf.sample(&mut rng) as u32)
                        .saturating_sub(1)
                        .min(n as u32 - 1)
                })
                .collect()
        }
        Dist::Sequential => (0..n as u32).collect(),
    };

    Workload {
        keys,
        values,
        access,
    }
}

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------

/// Root for bench temp directories. Overridable via `$BENCH_TMP`.
pub fn bench_tmp_root() -> PathBuf {
    std::env::var_os("BENCH_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

/// Create a fresh tempdir under `$BENCH_TMP`.
pub fn fresh_tempdir(prefix: &str) -> TempDir {
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(bench_tmp_root())
        .expect("tempdir")
}

/// Open an empty environment with default flags.
pub fn open_empty_env(dir: &Path, map_size: usize) -> Environment {
    EnvironmentBuilder::new()
        .map_size(map_size)
        .open(dir)
        .expect("open env")
}

/// Open an environment with custom flags (for `NO_SYNC`, `WRITE_MAP`, etc.).
pub fn open_env_with_flags(dir: &Path, map_size: usize, flags: EnvFlags) -> Environment {
    EnvironmentBuilder::new()
        .map_size(map_size)
        .flags(flags)
        .open(dir)
        .expect("open env")
}

/// Bulk-load a workload into `MAIN_DBI` using a single write transaction.
///
/// Keys are inserted in their natural (`workload.keys`) order — which is
/// sorted, so this is effectively sequential insert.
pub fn bulk_load(env: &Environment, workload: &Workload) {
    let mut txn = env.begin_rw_txn().expect("rw txn");
    for (k, v) in workload.keys.iter().zip(workload.values.iter()) {
        txn.put(MAIN_DBI, k, v, WriteFlags::empty()).expect("put");
    }
    txn.commit().expect("commit");
}
