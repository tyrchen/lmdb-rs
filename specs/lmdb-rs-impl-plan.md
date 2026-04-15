# lmdb-rs: Implementation Plan

## Phase Overview

```
Phase 1: Foundation          ──→  Read-only database access
Phase 2: Write Path          ──→  Single-key CRUD operations
Phase 3: B+ Tree Operations  ──→  Split, merge, rebalance
Phase 4: Transactions        ──→  Full MVCC with nested txns
Phase 5: Advanced Features   ──→  DUPSORT, compaction, CLI tools
Phase 6: Hardening           ──→  Benchmarks, fuzzing, production readiness
```

---

## Phase 1: Foundation (Read-Only Access)

**Goal:** Read an existing LMDB database file and traverse its B+ tree.

### 1.1 On-Disk Types (`lmdb-core/src/types.rs`)

- [ ] Define `PageHeader`, `NodeHeader`, `MetaPage`, `DbRecord` as `#[repr(C)]` structs
- [ ] Define `PageFlags`, `NodeFlags` as bitflags
- [ ] Implement `bytemuck` or manual byte-level access for zero-copy parsing
- [ ] Unit tests: round-trip serialization of all on-disk types

### 1.2 Page Abstraction (`lmdb-core/src/page.rs`)

- [ ] `Page` struct wrapping `&[u8]` with accessor methods
- [ ] `num_keys()`, `free_space()`, `is_branch()`, `is_leaf()`, etc.
- [ ] `node(idx)` → `Node` with key/data access
- [ ] `leaf2_key(idx, key_size)` for DUPFIXED pages
- [ ] Overflow page support (`overflow_pages()` accessor)
- [ ] Unit tests: parse sample pages from a real LMDB file

### 1.3 Memory Map (`lmdb-core/src/mmap.rs`)

- [ ] `MmapManager` using `memmap2::MmapRaw`
- [ ] `page(pgno) -> &[u8]` with bounds checking
- [ ] `meta_page(idx) -> MetaPage` to read meta pages
- [ ] Pick latest valid meta page
- [ ] Unit tests: open a real LMDB data file, read meta pages

### 1.4 Key Comparison (`lmdb-core/src/cmp.rs`)

- [ ] `Comparator` enum: Lexicographic, ReverseLexicographic, Integer, Custom
- [ ] `compare(&self, a: &[u8], b: &[u8]) -> Ordering`
- [ ] Unit tests: verify against C LMDB comparison results

### 1.5 ID List (`lmdb-core/src/idl.rs`)

- [ ] `IdList` sorted descending with binary search
- [ ] `append()`, `sort()`, `merge()`, `search()`
- [ ] `Id2List` sorted ascending (for dirty list)
- [ ] Unit tests: sort, merge, search edge cases

### 1.6 Read-Only Cursor (`lmdb-core/src/cursor.rs`)

- [ ] `CursorInner` with page stack and index stack
- [ ] `node_search()` — binary search within a page
- [ ] `page_search()` — traverse from root to leaf
- [ ] `cursor_first()`, `cursor_last()` — position at extremes
- [ ] `cursor_next()`, `cursor_prev()` — sequential traversal
- [ ] `cursor_set()`, `cursor_set_range()` — positioned lookup
- [ ] `cursor_sibling()` — move to adjacent pages
- [ ] Integration test: iterate entire database, verify key ordering

### 1.7 Read-Only Transaction (`lmdb-core/src/txn.rs`)

- [ ] `RoTransactionInner` with meta snapshot and reader slot
- [ ] `get(dbi, key) -> Result<&[u8]>`
- [ ] Named database resolution (lookup in MAIN_DBI)
- [ ] Integration test: read a database created by C LMDB

### 1.8 Public Read API (`lmdb/src/`)

- [ ] `Environment::builder() -> EnvironmentBuilder`
- [ ] `Environment::open(path) -> Result<Environment>`
- [ ] `Environment::begin_ro_txn() -> Result<RoTransaction>`
- [ ] `RoTransaction::get()`, `open_ro_cursor()`
- [ ] `RoCursor` with `Cursor` trait (get, iter, iter_from)
- [ ] `Error` enum with all error variants
- [ ] Integration test: full read workflow

**Milestone:** Can open a C LMDB database and read all keys.

---

## Phase 2: Write Path

**Goal:** Create a new database and perform single-key insertions.

### 2.1 Environment Creation (`lmdb-core/src/env.rs`)

- [ ] Create data file and lock file
- [ ] Initialize dual meta pages
- [ ] `mmap` with configurable size
- [ ] File locking (advisory locks on lock file)
- [ ] Pre-allocated write transaction (`txn0`)

### 2.2 Writer Lock (`lmdb-core/src/lock.rs`)

- [ ] Reader table in shared memory (mmap'd lock file)
- [ ] `ReaderSlot` with `AtomicU64` for txnid
- [ ] Reader slot allocation (acquire/release)
- [ ] Writer mutex (single-process: `std::sync::Mutex`)
- [ ] `find_oldest_reader()` — scan reader slots

### 2.3 Page Allocation (`lmdb-core/src/alloc.rs`)

- [ ] `page_malloc()` — allocate page buffer (reuse free list)
- [ ] `page_alloc()` — full allocation with free page search
- [ ] Loose page reuse
- [ ] File extension (`next_pgno` increment)
- [ ] Dirty list management (`DirtyList` insert/search)

### 2.4 Copy-on-Write (`lmdb-core/src/cow.rs`)

- [ ] `page_touch()` — COW a page before modification
- [ ] `page_copy()` — efficient page content copy
- [ ] Cursor fixup after COW (update all cursors on same dbi)

### 2.5 Node Operations (`lmdb-core/src/node.rs`)

- [ ] `node_add()` — insert a node at a given index
- [ ] `node_del()` — remove a node
- [ ] `node_shrink()` — reclaim unused space in a node
- [ ] Overflow page allocation for large values

### 2.6 Write Transaction (`lmdb-core/src/txn.rs`)

- [ ] `RwTransactionInner` with dirty list, free list
- [ ] `put(dbi, key, data, flags) -> Result<()>`
- [ ] `del(dbi, key, data) -> Result<()>`
- [ ] `_cursor_put()` — cursor-based insertion (simple case: page has room)

### 2.7 Commit — Simple Path

- [ ] `page_flush()` — write dirty pages via `pwritev()`
- [ ] `write_meta()` — atomic meta page update
- [ ] `fdatasync()` / `F_FULLFSYNC` (macOS)
- [ ] Free list save (simple: just write `mt_free_pgs` to FREE_DBI)

### 2.8 Public Write API

- [ ] `Environment::begin_rw_txn()`
- [ ] `RwTransaction::put()`, `del()`, `commit()`, `abort()`
- [ ] `RwCursor` with put/del operations
- [ ] Integration test: create database, insert keys, read back

**Milestone:** Can create a new database, write key-value pairs, and read them back.

---

## Phase 3: B+ Tree Operations

**Goal:** Handle page splits, merges, and all insertion/deletion edge cases.

### 3.1 Page Split (`lmdb-core/src/btree.rs`)

- [ ] `page_split()` — split a full page
- [ ] Split point calculation (50% fill, size-based refinement)
- [ ] Root split handling (new root creation)
- [ ] Recursive parent split
- [ ] LEAF2 page split (compact format)
- [ ] `MDB_APPEND` optimization (asymmetric split)
- [ ] Cursor fixup after split

### 3.2 Page Merge and Rebalance

- [ ] `rebalance()` — after deletion, check fill threshold
- [ ] Root collapse (branch root with single child)
- [ ] `page_merge()` — merge sibling pages
- [ ] `node_move()` — borrow a node from sibling
- [ ] `update_key()` — update separator key in parent after merge

### 3.3 Free Page Management (Full)

- [ ] `freelist_save()` — the full iterative algorithm
- [ ] Handle feedback loop (freeDB modifications during save)
- [ ] Multi-record splitting for large free lists
- [ ] `me_pghead` management (reclaimed page pool)
- [ ] Contiguous page range search for overflow allocations

### 3.4 Page Spilling

- [ ] `page_spill()` — write dirty pages to disk when dirty_room is low
- [ ] `page_unspill()` — bring spilled pages back
- [ ] `pages_xkeep()` — protect cursor pages from spilling
- [ ] Spill list management (shifted page numbers)

### 3.5 Stress Tests

- [ ] Insert 1M random keys, verify all readable
- [ ] Delete 50% of keys, verify remainder
- [ ] Mixed insert/delete patterns
- [ ] Large value (overflow page) insertion/deletion
- [ ] Sequential bulk load with MDB_APPEND

**Milestone:** All single-database CRUD operations work correctly under stress.

---

## Phase 4: Full Transaction System

**Goal:** Complete MVCC with concurrent readers, nested transactions.

### 4.1 Concurrent Readers

- [ ] Multiple read transactions from different threads
- [ ] Reader slot allocation thread safety
- [ ] `RoTransaction::reset()` / `InactiveTransaction::renew()`
- [ ] Stale reader detection (`check_readers()`)
- [ ] `NO_TLS` mode (multiple read txns per thread)

### 4.2 Nested Transactions

- [ ] `begin_nested_txn()` on `RwTransaction`
- [ ] Cursor shadowing (backup parent cursors)
- [ ] Nested commit: dirty list merge into parent
- [ ] Nested abort: discard changes, restore parent state
- [ ] Spill list reconciliation on nested commit
- [ ] `MDB_pgstate` save/restore for nested txns

### 4.3 Named Databases (Sub-databases)

- [ ] `open_db(name, flags)` — open or create named DB
- [ ] Named DB storage in MAIN_DBI (F_SUBDATA nodes)
- [ ] DBI handle management (sequence numbers for staleness)
- [ ] `db_stat()` for individual databases
- [ ] `drop()` — empty or delete a named database
- [ ] Default comparison function selection based on flags

### 4.4 MVCC Correctness Tests

- [ ] Writer + concurrent readers see consistent snapshots
- [ ] Long-lived reader prevents page reclamation (verified)
- [ ] Nested txn rollback restores exact parent state
- [ ] Named DB creation visible only after commit
- [ ] `MDB_RESERVE` zero-copy write

**Milestone:** Full transactional database with MVCC concurrency.

---

## Phase 5: Advanced Features

**Goal:** DUPSORT, compaction, and command-line tools.

### 5.1 DUPSORT

- [ ] Sub-page creation (inline duplicate storage)
- [ ] Sub-page → sub-database promotion
- [ ] `XCursor` for navigating within duplicate sets
- [ ] `FIRST_DUP`, `LAST_DUP`, `NEXT_DUP`, `PREV_DUP` cursor ops
- [ ] `GET_BOTH`, `GET_BOTH_RANGE` cursor ops
- [ ] `cursor_count()` — count duplicates
- [ ] `MDB_NODUPDATA` flag support
- [ ] DUPSORT deletion (single dup, all dups)

### 5.2 DUPFIXED

- [ ] LEAF2 page format (compact fixed-size keys)
- [ ] `GET_MULTIPLE`, `NEXT_MULTIPLE`, `PREV_MULTIPLE` cursor ops
- [ ] `MDB_MULTIPLE` write flag (batch duplicate insertion)
- [ ] LEAF2 page split/merge

### 5.3 Integer Keys

- [ ] `MDB_INTEGERKEY` comparison function
- [ ] `MDB_INTEGERDUP` comparison function
- [ ] Alignment handling for integer keys on branch vs leaf pages

### 5.4 Environment Copy

- [ ] `env.copy(path, compact: false)` — plain copy
- [ ] `env.copy(path, compact: true)` — compacting copy with page renumbering
- [ ] Double-buffered writer thread for compacting copy

### 5.5 CLI Tools (`lmdb-cli`)

- [ ] `lmdb-stat` — display database statistics
- [ ] `lmdb-dump` — dump database contents
- [ ] `lmdb-load` — load data from dump
- [ ] `lmdb-copy` — backup/compact database
- [ ] `lmdb-drop` — drop a named database

### 5.6 Environment Info and Stats

- [ ] `env.stat()` → `Stat`
- [ ] `env.info()` → `EnvInfo`
- [ ] `env.sync(force)`
- [ ] `env.set_flags(flags, on)`

**Milestone:** Feature-complete LMDB reimplementation.

---

## Phase 6: Hardening

**Goal:** Production readiness through testing, benchmarking, and documentation.

### 6.1 Benchmarking

- [ ] Set up `criterion` benchmarks
- [ ] Random read latency (vs C LMDB, redb)
- [ ] Sequential scan throughput
- [ ] Random write throughput
- [ ] Bulk load throughput (MDB_APPEND)
- [ ] Mixed read/write workload
- [ ] DUPSORT operations
- [ ] Profile with `cargo flamegraph` / `samply`
- [ ] Optimize hot paths based on profiling

### 6.2 Property-Based Testing

- [ ] B+ tree invariants (sorted keys, correct depth, valid pointers)
- [ ] Page fill invariants (lower <= upper, valid free space)
- [ ] Free list consistency (no double-free, no lost pages)
- [ ] MVCC invariant (readers see consistent snapshots)
- [ ] `proptest` for random insert/delete sequences
- [ ] Comparison testing: same operations against C LMDB, verify identical results

### 6.3 Crash Safety Testing

- [ ] Simulate crash at every point during commit
- [ ] Verify recovery always produces a valid database
- [ ] Test with `failpoints` crate for controlled I/O failure injection
- [ ] `fsync` failure simulation

### 6.4 Fuzzing

- [ ] `cargo fuzz` targets for:
  - Key/value insertion patterns
  - Cursor operation sequences
  - Database open with various flags
  - Concurrent reader/writer patterns

### 6.5 Documentation

- [ ] Rustdoc for all public types and methods
- [ ] `# Examples` in doc comments for key APIs
- [ ] `# Errors`, `# Panics`, `# Safety` sections
- [ ] Architecture guide in `docs/`
- [ ] Migration guide from C LMDB / heed

### 6.6 CI/CD

- [ ] GitHub Actions: build, test, clippy, fmt, audit, deny
- [ ] Miri runs for unsafe code validation
- [ ] Cross-platform testing (Linux, macOS)
- [ ] Benchmark regression tracking

**Milestone:** Production-ready v1.0 release.

---

## Dependency Matrix

| Crate | Purpose | Version |
|-------|---------|---------|
| `memmap2` | Memory-mapped file I/O | latest |
| `thiserror` | Error type derivation | latest |
| `bitflags` | Flag type definitions | latest |
| `tracing` | Structured logging | latest |
| `libc` | POSIX system calls (fdatasync, pwritev) | latest |

### Dev Dependencies

| Crate | Purpose |
|-------|---------|
| `criterion` | Benchmarking |
| `proptest` | Property-based testing |
| `rstest` | Parameterized tests |
| `tempfile` | Temporary directories for tests |

---

## Risk Assessment

### High Risk: Freelist Save Algorithm

The freelist save (`mdb_freelist_save`) is the single most complex part of LMDB. It has a feedback loop where writing to the freeDB can allocate/free pages, requiring iterative convergence. Incorrect implementation leads to lost pages or database corruption.

**Mitigation:** Implement a page audit function that verifies every page is either in use or in the free list. Run after every commit in test mode.

### Medium Risk: Page Split Correctness

Page splits involve complex pointer manipulation, especially for root splits and recursive parent splits. Off-by-one errors can corrupt the tree.

**Mitigation:** Property-based testing of tree invariants after every split. Comparison testing against C LMDB.

### Medium Risk: mmap Safety

Memory-mapped I/O requires careful lifetime management. Data references must not outlive their transaction.

**Mitigation:** Rust's lifetime system handles this naturally. The `&'txn [u8]` return type from `get()` ties data lifetimes to the transaction. All unsafe mmap access is confined to the `MmapManager` module.

### Low Risk: Concurrency Bugs

LMDB's concurrency model is simple (single writer, lock-free readers). The main risk is in reader slot management.

**Mitigation:** Use `AtomicU64` for reader slot txnid. Test with thread sanitizer.
