# lmdb-rs

A pure-Rust, zero-copy, memory-mapped key-value store that faithfully reimplements [LMDB](http://www.lmdb.tech/doc/) (Lightning Memory-Mapped Database). It provides Rust-native memory safety, type safety, and ergonomics while targeting performance parity with the original C implementation.

## Features

- **Zero-copy reads** via memory-mapped I/O (`memmap2`)
- **MVCC concurrency** -- single writer, multiple lock-free readers
- **Copy-on-Write B+ tree** with dual meta-page atomicity (no WAL required)
- **Named sub-databases** -- multiple B+ trees in one environment
- **DUPSORT / DUPFIXED** -- sorted duplicate values per key with compact storage
- **Nested transactions** -- child write transactions with savepoint semantics
- **Overflow pages** -- transparent handling of values larger than a page
- **Free page reclamation** -- MVCC-safe reuse of freed pages
- **Custom comparators** -- pluggable key and value comparison functions
- **MDB_APPEND** -- bulk loading optimization that skips binary search
- **MDB_RESERVE** -- zero-copy writes by reserving space and writing in place
- **Environment copy / compacting copy** -- online backup with optional defragmentation
- **WRITEMAP mode** -- direct mmap writes with `msync` instead of `pwrite`
- **Cross-process writer lock** via `flock(2)`
- **Crash recovery** -- automatic via dual meta pages, no manual recovery needed

## Quick Start

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
lmdb-rs-core = { git = "https://github.com/tyrchen/lmdb-rs" }
```

### Basic Usage

```rust
use lmdb_rs_core::env::Environment;
use lmdb_rs_core::types::{MAIN_DBI, WriteFlags};

// Open an environment
let env = Environment::builder()
    .map_size(10 * 1024 * 1024) // 10 MB
    .open("/tmp/my-lmdb")
    .expect("failed to open environment");

// Write data
let mut txn = env.begin_rw_txn().expect("begin write txn");
txn.put(MAIN_DBI, b"hello", b"world", WriteFlags::empty()).expect("put");
txn.commit().expect("commit");

// Read data
let txn = env.begin_ro_txn().expect("begin read txn");
let value = txn.get(MAIN_DBI, b"hello").expect("get");
assert_eq!(value, b"world");
```

### Named Databases

```rust
use lmdb_rs_core::env::Environment;
use lmdb_rs_core::types::{DatabaseFlags, WriteFlags};

let env = Environment::builder()
    .map_size(10 * 1024 * 1024)
    .max_dbs(4)
    .open("/tmp/my-lmdb-named")
    .expect("open");

let mut txn = env.begin_rw_txn().expect("begin write txn");
let db = txn.open_db(Some("users"), DatabaseFlags::CREATE).expect("open db");
txn.put(db, b"user:1", b"Alice", WriteFlags::empty()).expect("put");
txn.commit().expect("commit");
```

### Cursor Iteration

```rust
use lmdb_rs_core::env::Environment;
use lmdb_rs_core::types::MAIN_DBI;

let env = Environment::builder()
    .map_size(10 * 1024 * 1024)
    .open("/tmp/my-lmdb-cursor")
    .expect("open");

let txn = env.begin_ro_txn().expect("begin read txn");
let mut cursor = txn.open_cursor(MAIN_DBI).expect("open cursor");
for result in cursor.iter() {
    let (key, value) = result.expect("iterate");
    println!("{}: {}", String::from_utf8_lossy(key), String::from_utf8_lossy(value));
}
```

## Architecture

```
lmdb-rs/
├── crates/core/src/
│   ├── btree.rs    # B+ tree split, merge, rebalance, DUPSORT
│   ├── cmp.rs      # Key/value comparison functions
│   ├── cursor.rs   # Cursor traversal and positioning
│   ├── env.rs      # Environment, mmap, configuration
│   ├── error.rs    # Error types (thiserror)
│   ├── idl.rs      # ID list for sorted page tracking
│   ├── node.rs     # Node add/delete/init on pages
│   ├── page.rs     # Zero-copy page parsing
│   ├── txn.rs      # Read transactions, cursors, iterators
│   ├── types.rs    # On-disk format types (repr(C))
│   └── write.rs    # Write transactions, COW, page allocation, freelist
├── specs/          # Design specifications
├── docs/           # Documentation and research
└── vendors/lmdb/   # Reference C LMDB source
```

## Design Decisions

| Decision | Rationale |
|---|---|
| Pure Rust (no FFI) | Memory safety, no C toolchain dependency, full control over algorithms |
| Own on-disk format | Optimized for Rust's type system; not wire-compatible with C LMDB |
| `memmap2` for mmap | Maintained, safe API for memory mapping |
| `thiserror` for errors | Idiomatic Rust error types with zero-cost abstractions |
| No `unsafe` in public API | All unsafe contained in internal modules with documented invariants |
| Size-aware page splits | Matches C LMDB's algorithm for correct behavior across all page sizes |

## Platform Support

- **Linux** (x86_64, aarch64) -- 4 KB pages
- **macOS** (Apple Silicon, Intel) -- 16 KB pages

Windows support is planned for a future release.

## License

This project is distributed under the terms of MIT.

See [LICENSE](LICENSE.md) for details.

Copyright 2025 Tyr Chen
