# Documentation

## Guides

- [Developer Guide](dev-guide.md) — Architecture, internals, build instructions, and contributor reference for lmdb-rs
- [Benchmarks](bench/README.md) — How to run, save baselines, and gate CI on regressions
- [Benchmark Results — 2026-04-16](bench/results-2026-04-16.md) — First head-to-head numbers vs C LMDB (point reads win; seq/range/write gaps identified)

## Research

- [LMDB Architecture Deep Dive](research/lmdb-architecture.md) — Comprehensive analysis of the C LMDB internals: data structures, B+ tree, MVCC, COW, page management, locking, and sync
- [Rust LMDB Ecosystem](research/rust-ecosystem.md) — Survey of existing Rust LMDB bindings and alternatives (heed, redb, sled, libmdbx, jammdb)
