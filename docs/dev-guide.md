# Developer Guide

This guide covers the internals of lmdb-rs for contributors and anyone interested in understanding how a B+ tree key-value store works.

## Building and Testing

```bash
# Build
cargo build

# Run tests (uses nextest)
cargo nextest run --all-features

# Format (requires nightly for some options in rustfmt.toml)
cargo +nightly fmt

# Lint
cargo clippy -- -D warnings

# Strict lint
cargo clippy -- -D warnings -W clippy::pedantic
```

Or use the Makefile:

```bash
make build
make test
```

## On-Disk Format

The database is a single file plus an optional lock file. The file is memory-mapped for zero-copy reads.

### File Layout

```
┌─────────────────────────────────────────┐
│  Page 0: Meta Page A (P_META)           │  Dual meta pages for atomic commits
│  Page 1: Meta Page B (P_META)           │  (ping-pong between A and B)
├─────────────────────────────────────────┤
│  Pages 2..N: Data Pages                 │  B+ tree nodes
│  - FREE_DBI (dbi=0): free page B+ tree │
│  - MAIN_DBI (dbi=1): main database     │
│  - Named sub-databases                  │
│  - Overflow pages for large values      │
└─────────────────────────────────────────┘
```

### Page Header (16 bytes)

```
Offset  Size  Field
0       8     pgno   -- page number
8       2     pad    -- key size for LEAF2 pages
10      2     flags  -- PageFlags (BRANCH, LEAF, OVERFLOW, META, etc.)
12      2     lower  -- end of pointer array (grows down)
14      2     upper  -- start of node data (grows up)
```

For overflow pages, bytes 12-15 are reinterpreted as `overflow_pages: u32`.

### Node Header (8 bytes)

```
Offset  Size  Field
0       2     lo     -- low 16 bits of data size (leaf) or child pgno (branch)
2       2     hi     -- high 16 bits
4       2     flags  -- F_BIGDATA | F_SUBDATA | F_DUPDATA (leaf only)
6       2     ksize  -- key size in bytes
```

Followed by key bytes, then data bytes (leaf) or nothing (branch, pgno encoded in header).

## Core Modules

### `env.rs` -- Environment

The entry point. `Environment` manages:

- **Memory-mapped data file** via `memmap2::MmapRaw`
- **Per-database comparison functions** (key and value comparators)
- **Writer mutex** for single-writer semantics
- **Reader table** for tracking active read transactions
- **Configuration** (map size, max readers, max databases, flags)

Key types: `Environment`, `EnvironmentBuilder`, `EnvironmentInner`.

### `write.rs` -- Write Transactions

`RwTransaction` owns:

- **Dirty pages** (`DirtyPages`) -- modified pages stored in a sorted Vec of `(pgno, PageBuf)`
- **Free page lists** -- pages freed in this transaction (`free_pgs`), loose pages (freed + dirtied in same txn), reclaimed pages from FREE_DBI
- **Database metadata** (`dbs: Vec<DbStat>`) -- copies of each database's root, depth, page counts
- **Savepoints** for nested transactions

The write path:

1. `page_touch(pgno)` -- COW: free old page, allocate new, copy from mmap
2. `page_alloc()` -- try loose pages, then reclaim from freelist, then extend file
3. Mutations via `btree::cursor_put()` / `btree::cursor_del()`
4. `commit()`:
   - Save freed pages to FREE_DBI
   - Flush dirty named DB records to MAIN_DBI
   - `pwrite` / `pwritev` dirty pages to disk
   - `fdatasync` (or `F_FULLFSYNC` on macOS)
   - Write meta page (the atomic commit point)

### `btree.rs` -- B+ Tree Operations

Core algorithms:

- **`cursor_put`** -- insert or update a key-value pair
- **`cursor_del`** -- delete a key-value pair
- **`walk_and_touch`** -- traverse from root to leaf, COW-ing each page
- **`split_and_insert`** -- split a full leaf page with size-aware split point
- **`split_branch`** -- split a full branch page
- **`insert_separator`** -- propagate split separator up the tree
- **`rebalance`** -- merge underfull pages after deletion

#### Size-Aware Page Splitting

The split algorithm mirrors C LMDB's approach. Instead of always splitting at the midpoint by count (`total / 2`), it checks if size-aware splitting is needed:

```
need_size_split = nkeys < keythresh || nsize > pmax/16 || newindx >= nkeys
```

When needed, it accumulates actual node sizes from one direction until the page capacity is exceeded, ensuring both halves fit within a single page. This is critical for correctness on smaller page sizes (4 KB on Linux) where variable-size nodes can cause one half to overflow.

The helper functions `find_leaf_split_point` and `find_branch_split_point` implement this logic. They compute `leaf_entry_size` / `branch_entry_size` for each entry including the pointer slot and even-alignment padding.

### `cursor.rs` -- Cursor

The `Cursor` struct maintains a stack of `(page_ptr, key_index)` pairs representing the path from root to the current position. Operations:

- `first` / `last` -- seek to extremes
- `next` / `prev` -- advance/retreat within and across pages
- `set` / `set_range` -- binary search for a key
- `page_search` -- walk the tree to find a key

### `txn.rs` -- Read Transactions and Cursors

`RoTransaction` provides read-only access with zero-copy data references tied to the transaction lifetime. `RoCursor` wraps `Cursor` with DUPSORT-aware iteration and `CursorIter` provides idiomatic Rust iteration.

### `page.rs` -- Page Parsing

Zero-copy page parsing via `Page<'a>` which wraps a `&[u8]` slice. Provides accessors for page headers, node pointers, and `Node<'a>` views. The `even()` function rounds up to even alignment matching LMDB's `EVEN` macro.

### `node.rs` -- Node Operations

Low-level page mutation:

- `node_add` -- insert a node at a position, shifting pointers and updating lower/upper
- `node_add_bigdata` -- insert a BIGDATA node (overflow page reference)
- `node_del` -- remove a node with page compaction
- `init_page` -- initialize a page header (does NOT zero the data area)

### `types.rs` -- On-Disk Types

`#[repr(C)]` structs for the on-disk format: `Meta`, `DbStat`, `PageFlags`, `NodeFlags`, `EnvFlags`, `DatabaseFlags`, `WriteFlags`, `CursorOp`. All use little-endian byte order.

### `idl.rs` -- ID Lists

Sorted arrays of page numbers used for free page tracking. Operations: insert, merge, search, pop.

### `cmp.rs` -- Comparison Functions

- `default_cmp` -- lexicographic (memcmp)
- `reverse_cmp` -- reverse lexicographic
- `int_cmp` -- native-endian integer comparison (4 or 8 bytes)
- Custom functions via `Arc<dyn Fn(&[u8], &[u8]) -> Ordering>`

## Transaction Lifecycle

### Read Transaction

```
begin_ro_txn()
  ├── Acquire reader slot (atomic CAS on reader table)
  ├── Read latest committed meta page
  └── Snapshot database roots
      └── All reads go through mmap (zero-copy)
          └── Drop: release reader slot
```

### Write Transaction

```
begin_rw_txn()
  ├── Lock writer mutex
  ├── Read latest committed meta page
  └── Initialize dirty page list
      ├── put()/del() → walk_and_touch → page_touch (COW) → node mutations
      ├── Page full? → split_and_insert → insert_separator (may cascade)
      └── commit()
          ├── save_freelist() → write freed pages to FREE_DBI
          ├── flush_named_dbs() → update sub-DB records in MAIN_DBI
          ├── flush_dirty_pages() → pwrite/pwritev to data file
          ├── sync_data() → fdatasync / F_FULLFSYNC
          ├── write_meta() → atomic commit point
          └── Release writer mutex
```

## Free Page Management

LMDB uses an MVCC protocol for page reclamation:

1. When a write transaction COWs a page, the old page number is added to `free_pgs`.
2. At commit, `free_pgs` are serialized and written to FREE_DBI keyed by the transaction ID.
3. Future write transactions load FREE_DBI records, but only reclaim pages from transactions older than the oldest active reader.
4. Reclaimed pages are reused by `page_alloc()` before extending the file.

This ensures readers always see a consistent snapshot without locks.

## DUPSORT

DUPSORT databases allow multiple sorted values per key. Three storage strategies:

1. **Single value** -- stored inline as a normal node
2. **Inline sub-page** -- small number of dups stored as a sub-page within the leaf node
3. **Promoted sub-database** -- when the sub-page exceeds available space, promoted to a full B+ tree with its own root page

DUPFIXED is an optimization for fixed-size values that uses compact LEAF2 pages (no node headers, just packed keys).

## Overflow Pages

Values larger than `node_max` bytes are stored on overflow pages:

- The leaf node stores an 8-byte overflow page number with the `F_BIGDATA` flag
- The actual data is stored on contiguous overflow pages starting at that page number
- The node header's lo/hi fields store the actual data size (not 8)
- Overflow pages are freed when the key is deleted or the value is overwritten

## Nested Transactions

Implemented via savepoints on the write transaction:

- `begin_nested_txn()` -- snapshot current state (dbs, dirty pages, free lists)
- `commit_nested_txn()` -- discard the savepoint, keeping changes
- `abort_nested_txn()` -- restore from savepoint, discarding changes

## Testing Strategy

- **Unit tests** in each module (`#[cfg(test)] mod tests`)
- **Integration tests** in `crates/core/tests/integration.rs` -- end-to-end CRUD, stress tests, concurrent access
- **Real-world tests** in `crates/core/tests/realworld.rs` -- realistic usage patterns: caching, batch imports, named databases, nested transactions, MVCC isolation, persistence across reopens

Tests must pass on both 4 KB (Linux) and 16 KB (macOS) page sizes.

## Key Invariants

1. **Meta page ping-pong**: only one meta page is "active" at any time; the other is the previous commit
2. **COW guarantee**: every page on the write path is touched (copied) before mutation
3. **Reader isolation**: readers see the snapshot at their start time, unaffected by concurrent writes
4. **Single writer**: only one write transaction can be active at a time (enforced by mutex)
5. **Split correctness**: after a page split, both halves must fit within a single page (size-aware splitting)
6. **Even alignment**: all node sizes are rounded up to even boundaries (`even()` function)

## Reference

- [C LMDB source](vendors/lmdb/libraries/liblmdb/mdb.c) -- the authoritative reference for algorithms
- [LMDB Architecture Deep Dive](research/lmdb-architecture.md) -- comprehensive analysis of C LMDB internals
- [Rust LMDB Ecosystem](research/rust-ecosystem.md) -- survey of existing Rust bindings and alternatives
- [Design Specifications](../specs/) -- PRD, technical design, and implementation plan
