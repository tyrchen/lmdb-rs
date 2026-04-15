# lmdb-rs: Pure Rust LMDB Reimplementation — PRD

## 1. Vision

A pure-Rust, zero-copy, memory-mapped key-value store that faithfully reimplements LMDB's architecture and algorithms, achieving equivalent performance while providing Rust-native memory safety, type safety, and ergonomics.

## 2. Goals

### 2.1 Performance Parity

- Read performance within 5% of C LMDB (zero-copy reads via mmap)
- Write performance within 10% of C LMDB (COW B+ tree with vectored I/O)
- Same O(1) read transaction startup (lock-free reader table)
- Same O(log N) key lookup (B+ tree binary search)

### 2.2 Full Feature Set

- Single-writer, multi-reader MVCC concurrency
- Copy-on-Write B+ tree with dual meta page atomicity
- Named sub-databases (multiple B+ trees in one environment)
- DUPSORT with inline sub-pages and sub-database promotion
- DUPFIXED with compact LEAF2 page format
- Nested (child) write transactions
- Custom key/value comparison functions
- MDB_APPEND for bulk loading optimization
- MDB_RESERVE for zero-copy writes
- MDB_MULTIPLE for batch duplicate insertion
- Environment copy and compacting copy
- Stale reader detection and cleanup

### 2.3 Safety

- No `unsafe` in public API surface (all unsafe contained in internal modules)
- Lifetime-bound data references tied to transactions
- Type-safe database handles with compile-time key/value type checking
- No use-after-free, no data races by construction
- Proper error handling via Result types (no panics in library code)

### 2.4 Rust Ecosystem Integration

- `Send + Sync` environment (shareable across threads)
- `Send` transactions (usable from any single thread)
- serde integration for typed key/value serialization (optional feature)
- `tracing` integration for structured diagnostics
- Standard Rust error types via `thiserror`
- Iterator-based cursor API alongside imperative cursor operations

## 3. Non-Goals

### 3.1 Explicit Non-Goals

- **Wire-level compatibility with C LMDB data files.** We design our own on-disk format optimized for Rust's type system. A migration tool can be provided separately.
- **Windows support in v1.** Focus on Linux and macOS first. Windows can be added later.
- **`MDB_FIXEDMAP` support.** Fixed address mapping is inherently unsafe and not useful in modern environments.
- **`MDB_VL32` (32-bit huge pages).** Target 64-bit platforms only.
- **Write-ahead log (WAL).** Maintain LMDB's WAL-free design — COW B+ tree is the mechanism.
- **Multiple concurrent writers.** Maintain LMDB's single-writer design for simplicity and correctness.
- **Network/distributed operation.** Local filesystem only, same as LMDB.

### 3.2 Deferred to Later Versions

- WRITEMAP mode (v2)
- Cross-process sharing (v2 — start with single-process multi-thread)
- Online backup while writing (v2)
- Custom page sizes (v2 — start with OS page size)

## 4. API Surface

### 4.1 Environment

```rust
pub struct Environment { /* ... */ }

impl Environment {
    pub fn builder() -> EnvironmentBuilder;
    pub fn open_db(&self, name: Option<&str>) -> Result<Database>;
    pub fn begin_ro_txn(&self) -> Result<RoTransaction<'_>>;
    pub fn begin_rw_txn(&self) -> Result<RwTransaction<'_>>;
    pub fn sync(&self, force: bool) -> Result<()>;
    pub fn stat(&self) -> Result<Stat>;
    pub fn info(&self) -> Result<EnvInfo>;
    pub fn copy<P: AsRef<Path>>(&self, path: P, compact: bool) -> Result<()>;
    pub fn max_key_size(&self) -> usize;
    pub fn check_readers(&self) -> Result<u32>;
}
```

### 4.2 Transactions

```rust
pub struct RoTransaction<'env> { /* ... */ }
pub struct RwTransaction<'env> { /* ... */ }

impl<'env> RoTransaction<'env> {
    pub fn get<'txn>(&'txn self, db: Database, key: &[u8]) -> Result<&'txn [u8]>;
    pub fn open_ro_cursor(&self, db: Database) -> Result<RoCursor<'_, 'env>>;
    pub fn reset(self) -> InactiveTransaction<'env>;
    pub fn db_stat(&self, db: Database) -> Result<Stat>;
}

impl<'env> RwTransaction<'env> {
    pub fn get<'txn>(&'txn self, db: Database, key: &[u8]) -> Result<&'txn [u8]>;
    pub fn put(&mut self, db: Database, key: &[u8], data: &[u8], flags: WriteFlags) -> Result<()>;
    pub fn reserve(&mut self, db: Database, key: &[u8], len: usize, flags: WriteFlags) -> Result<&mut [u8]>;
    pub fn del(&mut self, db: Database, key: &[u8], data: Option<&[u8]>) -> Result<()>;
    pub fn open_rw_cursor(&mut self, db: Database) -> Result<RwCursor<'_, 'env>>;
    pub fn begin_nested_txn(&mut self) -> Result<RwTransaction<'env>>;
    pub fn open_db(&self, name: Option<&str>) -> Result<Database>;
    pub fn commit(self) -> Result<()>;
    pub fn abort(self);
    pub fn db_stat(&self, db: Database) -> Result<Stat>;
}

impl<'env> InactiveTransaction<'env> {
    pub fn renew(self) -> Result<RoTransaction<'env>>;
}
```

### 4.3 Cursors

```rust
pub struct RoCursor<'txn, 'env> { /* ... */ }
pub struct RwCursor<'txn, 'env> { /* ... */ }

// Both cursors implement:
pub trait Cursor<'txn> {
    fn get(&self, key: Option<&[u8]>, data: Option<&[u8]>, op: CursorOp) -> Result<(Option<&'txn [u8]>, &'txn [u8])>;
    fn iter(&mut self) -> CursorIter<'txn, '_>;
    fn iter_from(&mut self, key: &[u8]) -> CursorIter<'txn, '_>;
    fn iter_dup(&mut self) -> CursorDupIter<'txn, '_>;
    fn iter_dup_of(&mut self, key: &[u8]) -> CursorDupIter<'txn, '_>;
}

impl<'txn, 'env> RwCursor<'txn, 'env> {
    fn put(&mut self, key: &[u8], data: &[u8], flags: WriteFlags) -> Result<()>;
    fn del(&mut self, flags: WriteFlags) -> Result<()>;
}
```

### 4.4 Configuration

```rust
pub struct EnvironmentBuilder {
    map_size: usize,        // default: 1GB
    max_readers: u32,       // default: 126
    max_dbs: u32,           // default: 0 (unnamed DB only)
    flags: EnvFlags,
}

bitflags! {
    pub struct EnvFlags: u32 {
        const NO_SUB_DIR    = 0x4000;
        const NO_SYNC       = 0x10000;
        const READ_ONLY     = 0x20000;
        const NO_META_SYNC  = 0x40000;
        const NO_TLS        = 0x200000;
        const NO_LOCK       = 0x400000;
        const NO_READAHEAD  = 0x800000;
        const NO_MEM_INIT   = 0x1000000;
    }
}

bitflags! {
    pub struct DatabaseFlags: u32 {
        const REVERSE_KEY  = 0x02;
        const DUP_SORT     = 0x04;
        const INTEGER_KEY  = 0x08;
        const DUP_FIXED    = 0x10;
        const INTEGER_DUP  = 0x20;
        const REVERSE_DUP  = 0x40;
    }
}

bitflags! {
    pub struct WriteFlags: u32 {
        const NO_OVERWRITE = 0x10;
        const NO_DUP_DATA  = 0x20;
        const CURRENT      = 0x40;
        const APPEND       = 0x20000;
        const APPEND_DUP   = 0x40000;
    }
}
```

### 4.5 Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("key/data pair already exists")]
    KeyExist,
    #[error("no matching key/data pair found")]
    NotFound,
    #[error("requested page not found (corrupted)")]
    PageNotFound,
    #[error("located page was wrong type (corrupted)")]
    Corrupted,
    #[error("environment had fatal error")]
    Panic,
    #[error("database version mismatch")]
    VersionMismatch,
    #[error("file is not a valid database")]
    Invalid,
    #[error("environment mapsize limit reached")]
    MapFull,
    #[error("max databases limit reached")]
    DbsFull,
    #[error("max readers limit reached")]
    ReadersFull,
    #[error("transaction has too many dirty pages")]
    TxnFull,
    #[error("cursor stack limit reached (internal)")]
    CursorFull,
    #[error("page has no more space (internal)")]
    PageFull,
    #[error("database contents grew beyond mapsize")]
    MapResized,
    #[error("incompatible operation or database flags changed")]
    Incompatible,
    #[error("invalid reuse of reader lock table slot")]
    BadReaderSlot,
    #[error("transaction must abort, has child, or is invalid")]
    BadTxn,
    #[error("unsupported key/data size")]
    BadValSize,
    #[error("database handle was closed/changed unexpectedly")]
    BadDbi,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```

## 5. Crate Structure

```
lmdb-rs/
├── crates/
│   ├── lmdb-core/          # Core engine (no public API)
│   │   ├── page.rs         # Page types, layout, node operations
│   │   ├── btree.rs        # B+ tree search, split, merge, rebalance
│   │   ├── cursor.rs       # Cursor traversal and mutation
│   │   ├── txn.rs          # Transaction lifecycle, COW, MVCC
│   │   ├── env.rs          # Environment, mmap, file management
│   │   ├── freelist.rs     # Free page tracking and reclamation
│   │   ├── idl.rs          # ID list data structure (sorted arrays)
│   │   ├── lock.rs         # Reader table, writer mutex, stale detection
│   │   ├── meta.rs         # Meta page read/write, dual-meta ping-pong
│   │   ├── io.rs           # File I/O, sync, vectored writes
│   │   └── cmp.rs          # Key comparison functions
│   ├── lmdb/               # Public API crate
│   │   ├── env.rs          # Environment, EnvironmentBuilder
│   │   ├── txn.rs          # RoTransaction, RwTransaction
│   │   ├── cursor.rs       # RoCursor, RwCursor, iterators
│   │   ├── db.rs           # Database handle
│   │   ├── error.rs        # Error types
│   │   ├── flags.rs        # Flag types (bitflags)
│   │   └── lib.rs          # Re-exports
│   └── lmdb-cli/           # Command-line tools
│       ├── stat.rs          # mdb_stat equivalent
│       ├── dump.rs          # mdb_dump equivalent
│       ├── load.rs          # mdb_load equivalent
│       ├── copy.rs          # mdb_copy equivalent
│       └── drop.rs          # mdb_drop equivalent
├── specs/
├── docs/
└── tests/
    ├── integration/         # Integration tests
    └── compat/              # Compatibility tests (vs C LMDB)
```

## 6. Quality Requirements

### 6.1 Correctness

- All LMDB test scenarios must pass
- Crash recovery must be tested with simulated failures
- Concurrent reader/writer tests with thread sanitizer
- Property-based testing with proptest for B+ tree invariants
- Fuzzing of key/value insertion patterns

### 6.2 Performance Benchmarks

Using `criterion`:
- Single-key random read latency
- Sequential scan throughput
- Random write throughput
- Bulk load (MDB_APPEND) throughput
- Mixed read/write under concurrent load
- Large value (overflow page) read/write
- DUPSORT operations
- Comparison against C LMDB and redb

### 6.3 Safety

- `cargo clippy -- -D warnings -W clippy::pedantic`
- `cargo audit` clean
- `cargo deny` license check
- Miri for undefined behavior detection
- No unsafe in public API modules
