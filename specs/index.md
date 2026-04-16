# Specifications

## lmdb-rs: Pure Rust LMDB Reimplementation

- [PRD](lmdb-rs-prd.md) — Product requirements: goals, non-goals, API surface, crate structure, quality requirements
- [Design](lmdb-rs-design.md) — Technical design: on-disk format, memory architecture, B+ tree, transactions, concurrency, I/O, DUPSORT
- [Implementation Plan](lmdb-rs-impl-plan.md) — Phased implementation plan with milestones, dependencies, risk assessment
- [Remaining Features](remaining-features-design.md) — Design for overflow pages, free page reuse, nested transactions, DUPSORT/DUPFIXED
- [Performance Benchmarks](perf-benchmark-design.md) — Criterion-based self-benchmarks and head-to-head comparison vs C LMDB (workload taxonomy, harness, methodology, phasing)
