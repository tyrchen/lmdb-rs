# Rust LMDB Ecosystem Research

## Existing Projects

### 1. heed (Meilisearch)

**Approach:** Type-safe Rust bindings over C LMDB via `lmdb-sys` FFI.

- The leading maintained LMDB binding in Rust
- Uses serde for typed key/value encoding
- Provides a safe API with lifetime-bound transactions and zero-copy reads
- **Limitations:** Still FFI to C LMDB — inherits all constraints (max 1 writer, fixed map size pre-allocation, no built-in compaction, C dependency)

### 2. lmdb-rkv / lmdb-rs (Mozilla / danburkert)

**Approach:** Thin FFI wrappers over C LMDB.

- `lmdb-rkv` was Mozilla's fork (used in Firefox/rkv), now largely unmaintained
- `lmdb-rs` (danburkert) was the original binding, also unmaintained
- Community has consolidated around `heed` as the maintained option

### 3. sled

**Approach:** Pure Rust embedded DB using Bw-tree (lock-free B+ tree variant).

- Very different architecture from LMDB — no mmap for data pages, log-structured storage
- Had performance issues, high write amplification, known corruption bugs
- Author (spacejam) effectively paused development for "sled 1.0" rewrite (in progress for years)
- **Not recommended for production** by the community

### 4. redb

**Approach:** Pure Rust, B+ tree based, inspired by LMDB's MVCC/COW design.

- Uses mmap for reads, explicit writes
- ACID with single-writer/multi-reader like LMDB
- Actively maintained, most credible pure-Rust LMDB alternative
- Performance close to LMDB (within ~1.5-2x on most benchmarks, sometimes matching)
- No C dependencies
- **Limitations:** Younger project, smaller ecosystem, no multi-map (DupSort) equivalent

### 5. libmdbx (MDBX) + Rust bindings

**Approach:** C fork of LMDB by Leonid Yuriev, with Rust bindings (`libmdbx-rs`).

Key improvements over LMDB:
- Auto-resizing database (no fixed mapsize)
- Better Windows support
- Built-in compaction/GC improvements
- More robust reader slot handling
- Additional safety checks
- **Licensing issue:** Changed to custom non-OSI license in 2024, causing migration away by projects like Reth/Paradigm

### 6. jammdb

**Approach:** Pure Rust, inspired by BoltDB (Go's LMDB-like DB).

- B+ tree with COW, mmap-based
- Small, simple, but limited maintenance and features

### 7. fjall/lsm-tree

**Approach:** Pure Rust LSM-tree based.

- Different trade-offs: better write throughput, worse read latency
- Not mmap-based, not directly comparable to LMDB

## The mmap + Borrow Checker Challenge

The fundamental tension in a Rust LMDB implementation:

1. **mmap returns raw pointers** (`*mut u8`) with lifetimes the compiler cannot track
2. Rust bindings must use `unsafe` to tie returned `&[u8]` slices to transaction lifetimes
3. The borrow checker cannot prevent use-after-unmap at the type level without careful lifetime design
4. **Solution approaches:**
   - Tie data slice lifetimes to transaction guard objects (heed, redb)
   - Use `PhantomData` and lifetime parameters to encode the relationship
   - Control the entire abstraction layer in Rust (redb, jammdb) for maximum safety
   - Accept some `unsafe` at the mmap boundary as unavoidable

## Community Consensus (as of early 2025)

| Use Case | Recommendation |
|----------|---------------|
| Production with C dep OK | **heed** (LMDB) |
| Pure Rust | **redb** is the clear leader |
| MDBX improvements needed | Watch licensing carefully |
| New projects | Avoid sled |

## What Our Reimplementation Can Offer

A pure-Rust LMDB reimplementation that:
1. Matches LMDB's performance by replicating its exact algorithms
2. Provides memory safety via Rust's type system
3. Supports the full feature set including DUPSORT (which redb lacks)
4. Maintains wire-level compatibility with LMDB data files (optional goal)
5. Incorporates libmdbx improvements without the licensing concerns
6. Leverages Rust's ecosystem (memmap2, thiserror, etc.)
