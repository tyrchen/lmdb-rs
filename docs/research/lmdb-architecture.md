# LMDB Architecture Deep Dive

## Overview

LMDB (Lightning Memory-Mapped Database) is a B+ tree-based, memory-mapped, transactional key-value store created by Howard Chu at Symas Corp. It was derived from `btree.c` by Martin Hedenfalk and is used extensively in OpenLDAP, Monero, Caffe, and many other systems.

**Key design principles:**
- Single file + lockfile, no WAL, no transaction log
- Memory-mapped I/O for zero-copy reads
- Copy-on-Write (COW) B+ tree for MVCC
- Single writer, multiple concurrent readers with no read locks
- Crash-safe without recovery procedures

**Source files:**
- `mdb.c` (~11,500 lines) — the entire database engine
- `midl.c` / `midl.h` (~420 + ~200 lines) — ID list management for free page tracking
- `lmdb.h` (~1,650 lines) — public API header

---

## 1. File Layout and Memory Mapping

### Database File Structure

```
┌─────────────────────────────────────────────────────────┐
│  Page 0: Meta Page 0 (P_META)                           │
├─────────────────────────────────────────────────────────┤
│  Page 1: Meta Page 1 (P_META)                           │
├─────────────────────────────────────────────────────────┤
│  Page 2..N: Data Pages (branch, leaf, overflow)         │
│  - B+ tree pages for FREE_DBI (free page list)          │
│  - B+ tree pages for MAIN_DBI (main database)           │
│  - B+ tree pages for named sub-databases                │
├─────────────────────────────────────────────────────────┤
│  (grows as needed up to mapsize)                        │
└─────────────────────────────────────────────────────────┘
```

The entire database file is memory-mapped via `mmap()`. The mapsize is set by `mdb_env_set_mapsize()` and defines the maximum database size. The actual file grows lazily as pages are allocated.

### Meta Pages (Dual-Meta Ping-Pong)

Two meta pages exist at pages 0 and 1. Transaction N writes to meta page `N % 2`. This provides atomic commit without a WAL:

```c
typedef struct MDB_meta {
    uint32_t    mm_magic;       // MDB_MAGIC (0xBEEFC0DE)
    uint32_t    mm_version;     // MDB_DATA_VERSION (1)
    void       *mm_address;     // fixed mapping address (0 if not fixed)
    mdb_size_t  mm_mapsize;     // size of mmap region
    MDB_db      mm_dbs[2];      // FREE_DBI and MAIN_DBI root info
    pgno_t      mm_last_pg;     // last used page in datafile
    txnid_t     mm_txnid;       // txnid that committed this page
} MDB_meta;
// mm_psize is aliased to mm_dbs[FREE_DBI].md_pad
// mm_flags is aliased to mm_dbs[FREE_DBI].md_flags
```

**Commit atomicity:** The meta page write is the commit point. Only `mm_mapsize` through `mm_txnid` are written (not the full page), and this write is atomic on most filesystems for aligned writes < sector size. On failure, the previous meta page remains valid.

### Lock File Structure

A separate file (`data.mdb-lock` or `lock.mdb`) is memory-mapped for inter-process coordination:

```c
typedef struct MDB_txninfo {
    // Header section (cache-line aligned):
    uint32_t        mtb_magic;        // MDB_MAGIC
    uint32_t        mtb_format;       // lock format version
    volatile txnid_t mtb_txnid;       // last committed txnid
    volatile uint    mtb_numreaders;   // number of reader slots used
    mdb_mutex_t      mtb_rmutex;       // reader table mutex
    mdb_mutex_t      mtb_wmutex;       // writer mutex (separate cache line)
    MDB_reader       mti_readers[1];   // reader slots (flexible array)
} MDB_txninfo;
```

Each reader slot is cache-line aligned (64 bytes) to prevent false sharing:

```c
typedef struct MDB_reader {
    volatile txnid_t  mr_txnid;   // reader's snapshot txnid, or (txnid_t)-1
    volatile pid_t    mr_pid;     // owning process ID
    volatile pthread_t mr_tid;    // owning thread ID
} MDB_reader; // padded to CACHELINE (64) bytes
```

---

## 2. Page Types and Layout

### Page Header

Every page starts with a common header (mdb.c:1003):

```c
typedef struct MDB_page {
    union {
        pgno_t       p_pgno;    // page number
        MDB_page    *p_next;    // for in-memory freed page list
    } mp_p;
    uint16_t  mp_pad;           // key size for LEAF2 pages
    uint16_t  mp_flags;         // page type flags
    union {
        struct {
            indx_t  pb_lower;   // lower bound of free space
            indx_t  pb_upper;   // upper bound of free space
        } pb;
        uint32_t  pb_pages;     // number of overflow pages
    } mp_pb;
    indx_t    mp_ptrs[0];       // dynamic array of offsets
} MDB_page;
```

**Header size:** `PAGEHDRSZ` = `offsetof(MDB_page, mp_ptrs)` = 16 bytes (on 64-bit)

### Page Flags

| Flag       | Value    | Description                                       |
|------------|----------|---------------------------------------------------|
| `P_BRANCH` | `0x01`   | Internal B+ tree node                             |
| `P_LEAF`   | `0x02`   | Leaf B+ tree node with key-value pairs            |
| `P_OVERFLOW`| `0x04`  | Overflow page for large values                    |
| `P_META`   | `0x08`   | Meta page                                         |
| `P_DIRTY`  | `0x10`   | Page has been modified (COW copy)                 |
| `P_LEAF2`  | `0x20`   | Fixed-size leaf for DUPFIXED                      |
| `P_SUBP`   | `0x40`   | Sub-page (inline duplicate data)                  |
| `P_LOOSE`  | `0x4000` | Dirtied then freed, can be reused in same txn     |
| `P_KEEP`   | `0x8000` | Don't spill this page to disk                     |

### Branch Page Layout

```
┌──────────────────────────────────────────────┐
│ Page Header (16 bytes)                       │
├──────────────────────────────────────────────┤
│ mp_ptrs[0] ─┐                                │
│ mp_ptrs[1]  │  Sorted offset array           │
│ ...         │  (grows downward)              │
│ mp_ptrs[n] ─┘                                │
├── mp_lower ─────────────────────────────────┤
│                                              │
│         (free space)                         │
│                                              │
├── mp_upper ─────────────────────────────────┤
│ Node[n]: [mn_lo|mn_hi|mn_flags|mn_ksize|key]│
│ ...                                          │
│ Node[1]: [mn_lo|mn_hi|mn_flags|mn_ksize|key]│
│ Node[0]: [mn_lo|mn_hi|  (implicit key)  ]   │
└──────────────────────────────────────────────┘
```

Branch nodes store a child page number in `mn_lo|mn_hi|mn_flags` (48 bits) and a separator key. The first branch node has an implicit (empty) key — it's the leftmost child pointer.

### Leaf Page Layout

Same layout as branch pages, but nodes contain key + data:

```c
typedef struct MDB_node {
    unsigned short mn_lo, mn_hi;   // data size (leaf) or child pgno (branch)
    unsigned short mn_flags;        // node flags
    unsigned short mn_ksize;        // key size
    char           mn_data[1];      // key bytes, then data bytes
} MDB_node;
```

**Node flags:**
| Flag        | Value  | Description                                    |
|-------------|--------|------------------------------------------------|
| `F_BIGDATA` | `0x01` | Data is on overflow page(s)                    |
| `F_SUBDATA` | `0x02` | Data is a sub-database (MDB_db struct)         |
| `F_DUPDATA` | `0x04` | Node has duplicate data (sub-page or sub-DB)   |

### Overflow Pages

When a value exceeds `(pagesize - PAGEHDRSZ) / MDB_MINKEYS`, it is stored in overflow pages. The leaf node's data contains just the page number (F_BIGDATA flag). The overflow pages are contiguous; only the first has a real header with `mp_pages` indicating the count.

```
Number of overflow pages = (PAGEHDRSZ - 1 + data_size) / pagesize + 1
```

### LEAF2 Pages (DUPFIXED)

For `MDB_DUPFIXED` databases, duplicate values are stored in compact LEAF2 pages with no node headers — just contiguous fixed-size keys packed after the page header. The key size is stored in `mp_pad`.

### Sub-pages (Inline Duplicates)

When a key has few small duplicates, they are stored inline as a sub-page within the leaf node's data area (F_DUPDATA without F_SUBDATA). The sub-page has its own mini page header and nodes. When duplicates grow too large, they're promoted to a sub-database (F_DUPDATA + F_SUBDATA).

---

## 3. B+ Tree Structure

### Tree Organization

LMDB uses a B+ tree where:
- **Branch pages** contain separator keys and child page numbers
- **Leaf pages** contain actual key-value pairs
- Keys are sorted in lexicographic order by default
- The tree grows upward: when the root splits, a new root is created

### Two Built-in Databases

Every environment has two mandatory databases:
1. **FREE_DBI (0)** — tracks free (reusable) pages, keyed by txnid
2. **MAIN_DBI (1)** — the main database; named sub-databases are stored here as key-value pairs where the value is an `MDB_db` struct

### MDB_db Structure (Per-Database Metadata)

```c
typedef struct MDB_db {
    uint32_t   md_pad;            // also ksize for LEAF2 pages
    uint16_t   md_flags;          // database flags
    uint16_t   md_depth;          // B+ tree depth
    pgno_t     md_branch_pages;   // count of branch pages
    pgno_t     md_leaf_pages;     // count of leaf pages
    pgno_t     md_overflow_pages; // count of overflow pages
    mdb_size_t md_entries;        // total number of entries
    pgno_t     md_root;           // root page number (P_INVALID if empty)
} MDB_db;
```

### Key Comparison

Default comparison is `memcmp`-based lexicographic ordering. Special comparison functions:
- `mdb_cmp_memn` — standard memcmp with length comparison
- `mdb_cmp_memnr` — reverse memcmp (for MDB_REVERSEKEY)
- `mdb_cmp_int` — native integer comparison (for MDB_INTEGERKEY)
- `mdb_cmp_long` — 64-bit integer comparison
- Custom comparators can be set via `mdb_set_compare()`

### Node Search (Binary Search)

`mdb_node_search()` (mdb.c:6046) performs binary search within a page:
1. Gets the number of keys on the page
2. For branch pages, adjusts to skip the first (implicit) key
3. Binary search comparing the target key against node keys
4. Returns the node and sets `exactp` to indicate exact match

### Page Search (Tree Traversal)

`mdb_page_search()` (mdb.c:6160) traverses from root to leaf:
1. Gets the root page from `MDB_db.md_root`
2. If `MDB_PS_MODIFY`, calls `mdb_page_touch()` to COW the page
3. Calls `mdb_page_search_root()` which:
   - At each branch level, uses `mdb_node_search()` to find the child
   - Pushes the page onto the cursor stack
   - Descends to the child page
   - Continues until a leaf page is reached

---

## 4. Cursor System

### Cursor Structure

```c
struct MDB_cursor {
    MDB_cursor    *mc_next;        // linked list of cursors on this DB/txn
    MDB_cursor    *mc_backup;      // backup for shadow cursors
    MDB_xcursor   *mc_xcursor;     // sub-cursor for DUPSORT databases
    MDB_txn       *mc_txn;         // owning transaction
    MDB_dbi        mc_dbi;         // database handle index
    MDB_db        *mc_db;          // database metadata record
    MDB_dbx       *mc_dbx;         // auxiliary DB info (comparators, name)
    unsigned char *mc_dbflag;      // per-DB flags for this txn
    unsigned short mc_snum;        // number of pages in stack
    unsigned short mc_top;         // index of top page (mc_snum - 1)
    unsigned int   mc_flags;       // cursor state flags
    MDB_page      *mc_pg[CURSOR_STACK];  // page stack (max depth 32)
    indx_t         mc_ki[CURSOR_STACK];  // key index at each level
};
```

The cursor maintains a **path stack** from root to current leaf position. `CURSOR_STACK` = 32, supporting trees up to depth 32 (2^32 nodes minimum at 2 keys/node).

### Cursor Operations

- `MDB_FIRST` / `MDB_LAST` — position at first/last entry
- `MDB_NEXT` / `MDB_PREV` — sequential traversal
- `MDB_SET` / `MDB_SET_KEY` / `MDB_SET_RANGE` — positioned lookup
- `MDB_GET_CURRENT` — return current position
- `MDB_GET_BOTH` / `MDB_GET_BOTH_RANGE` — match both key and data

### XCursor (Duplicate Data Cursor)

For `MDB_DUPSORT` databases, each cursor has an embedded `MDB_xcursor`:

```c
typedef struct MDB_xcursor {
    MDB_cursor  mx_cursor;   // sub-cursor for traversing duplicates
    MDB_db      mx_db;       // database record for the dup sub-tree
    MDB_dbx     mx_dbx;      // auxiliary record (comparator)
    unsigned char mx_dbflag;
} MDB_xcursor;
```

---

## 5. Transaction System (MVCC)

### Transaction Structure

```c
struct MDB_txn {
    MDB_txn     *mt_parent;      // parent of nested txn
    MDB_txn     *mt_child;       // nested txn under this one
    pgno_t       mt_next_pgno;   // next unallocated page
    txnid_t      mt_txnid;       // transaction ID (incrementing from 1)
    MDB_env     *mt_env;
    MDB_IDL      mt_free_pgs;    // pages freed during this txn
    MDB_page    *mt_loose_pgs;   // loose pages (freed then reusable in-txn)
    int           mt_loose_count;
    MDB_IDL      mt_spill_pgs;   // pages spilled to disk (sorted, shifted << 1)
    union {
        MDB_ID2L  dirty_list;    // write txn: sorted list of dirty pages
        MDB_reader *reader;      // read txn: reader table slot
    } mt_u;
    MDB_db       *mt_dbs;        // per-DB metadata array
    MDB_cursor  **mt_cursors;    // per-DB cursor linked lists (write only)
    unsigned char *mt_dbflags;   // per-DB flags (dirty, stale, etc.)
    unsigned int  mt_flags;      // transaction flags
    unsigned int  mt_dirty_room; // remaining capacity in dirty list
};
```

### Read Transactions

1. Acquire reader mutex briefly to find an empty reader slot
2. Record current `txnid` in the reader slot
3. Copy meta page's `MDB_db` records for FREE_DBI and MAIN_DBI
4. Release reader mutex — all subsequent reads are lock-free
5. Read directly from mmap'd pages — zero copy

Read transactions see a consistent snapshot. They never block writers and are never blocked.

### Write Transactions

1. Acquire writer mutex (only one writer at a time)
2. Read the latest meta page to get current state
3. Increment txnid
4. All modifications use Copy-on-Write:
   - `mdb_page_touch()` copies a page before modifying it
   - Original pages remain available for concurrent readers
5. Commit: flush dirty pages, write new meta page

### Copy-on-Write (COW) Mechanism

When a write transaction needs to modify a page:

1. **`mdb_page_touch()`** (mdb.c ~2800):
   - If page is already dirty in this txn, return it
   - If page was spilled, unspill it
   - Allocate a new page (from freelist or extend file)
   - Copy the page contents to the new page
   - Mark the new page as dirty
   - Update parent's pointer to the new page
   - Add the old page to the free list

2. The dirty page is tracked in `mt_u.dirty_list` (an ID2L: sorted array of pgno→page pointer pairs)

3. On commit, all dirty pages are written to their final positions in the file

### Nested Transactions

LMDB supports nested write transactions via `mdb_txn_begin()` with a parent txn:

- Child txn gets a copy of parent's DB state
- Child's dirty list is separate
- On child commit: merge dirty lists into parent
- On child abort: discard child's changes, restore parent state
- Uses `MDB_ntxn` which includes `MDB_pgstate` to save/restore freelist state

### Transaction Commit Flow

`_mdb_txn_commit()` (mdb.c:3975):

1. Commit any child transactions recursively
2. If nested txn: merge dirty list into parent and return
3. Close all cursors
4. If no changes, do empty commit
5. Update named DB root pointers in MAIN_DBI
6. **`mdb_freelist_save()`** — write freed page lists to FREE_DBI
7. **`mdb_page_flush()`** — write dirty pages to disk using `writev()`/`pwritev()`
8. `fsync()` the data file (unless MDB_NOSYNC)
9. **`mdb_env_write_meta()`** — write the new meta page (the commit point)
10. Release writer mutex

---

## 6. Free Page Management

### Overview

LMDB tracks free pages in the FREE_DBI database. Keys are transaction IDs; values are sorted ID lists of freed page numbers.

When pages are freed by a transaction, they cannot be reused until all readers that might reference them have finished. This is the MVCC reclamation protocol.

### Data Structure: ID Lists (midl.c)

The `MDB_IDL` is a sorted array of page numbers (IDs), stored in **descending order**:

```c
typedef MDB_ID *MDB_IDL;  // MDB_ID = size_t
// ids[0] = count
// ids[1] = highest (first) ID
// ids[n] = lowest (last) ID
// ids[-1] = allocated capacity
```

Key operations:
- **`mdb_midl_search()`** — binary search (descending order)
- **`mdb_midl_sort()`** — quicksort + insertion sort hybrid (descending)
- **`mdb_midl_xmerge()`** — merge two sorted IDLs in descending order
- **`mdb_midl_append()`** — unsorted append (sorted later)

The `MDB_ID2L` is a sorted array of (ID, pointer) pairs in **ascending order**, used for the dirty page list.

### Page Allocation (`mdb_page_alloc`, mdb.c:2501)

Priority order for finding pages:
1. **Loose pages** — pages freed then dirtied in same txn, reusable immediately
2. **Free list (me_pghead)** — reclaimed pages from old transactions
3. **FreeDB records** — read more records from FREE_DBI
4. **Extend the file** — allocate from `mt_next_pgno`

For multi-page allocations (overflow), it searches the free list for contiguous ranges.

### Freelist Save (`mdb_freelist_save`, mdb.c:3517)

The most complex part of LMDB. During commit:
1. Delete consumed freeDB records
2. Write the current txn's freed pages to freeDB with key = current txnid
3. Reserve space in freeDB for the reclaimed page head (`me_pghead`)
4. Split large page lists across multiple freeDB records to avoid overflow pages
5. Fill in the reserved records with the actual page numbers

This is iterative because writing to freeDB may itself allocate/free pages, creating a feedback loop.

### Finding Oldest Reader

`mdb_find_oldest()` scans all reader slots to find the minimum active txnid. Pages freed by transactions older than this can be reclaimed.

---

## 7. Page Spilling

When the dirty list grows too large (`MDB_TXN_FULL` threshold), pages are "spilled" to disk:

1. Identify dirty pages that aren't in active cursor paths
2. Write them to their on-disk positions
3. Mark them in `mt_spill_pgs` (page numbers shifted left by 1)
4. Remove from dirty list
5. If re-accessed, they're "unspilled" (brought back into dirty list)

---

## 8. Page Split Algorithm

`mdb_page_split()` (mdb.c:9747) is the most complex B+ tree operation:

1. **Create right sibling** page
2. **Root split handling**: if splitting the root, create a new root page and push down
3. **Find split point**:
   - Default: midpoint `(nkeys + 1) / 2`
   - For large keys or sequential inserts: calculate based on actual node sizes to ensure both pages have room
   - For `MDB_APPEND`: put everything on the new page (sequential optimization)
4. **For LEAF2**: direct memory copy of fixed-size keys
5. **For regular pages**:
   - Copy page to temp buffer with slot for new key
   - Move nodes from split point onward to right sibling
   - Insert new key at appropriate position
6. **Insert separator** key into parent branch page (may trigger recursive split)
7. **Update cursors** tracking affected pages

---

## 9. Locking and Concurrency

### Writer Mutex

A single process-shared mutex (`me_wmutex`) ensures only one writer at a time. On macOS, this uses SysV semaphores (since macOS doesn't support process-shared POSIX mutexes). On Linux, it uses robust POSIX mutexes.

### Reader Table

Readers don't acquire locks for data access. They:
1. Briefly lock `me_rmutex` to find/allocate a reader slot
2. Write their txnid to the slot (volatile write)
3. Release `me_rmutex`
4. All subsequent reads are completely lock-free

The writer scans reader slots (without locking) to determine the oldest active reader. This is safe because:
- Slots are cache-line aligned — no false sharing
- Only the owning thread writes to a slot
- The writer only needs an upper bound on the oldest reader

### Stale Reader Detection

Reader slots record PID and TID. `mdb_reader_check()` can detect stale readers (crashed processes) by checking if the PID is still alive via `kill(pid, 0)` or OpenProcess on Windows.

### Robust Mutexes

On platforms supporting robust mutexes, if a process dies while holding the writer mutex, the next process to lock it receives `EOWNERDEAD` and can recover by calling `pthread_mutex_consistent()`.

---

## 10. Sync and Durability

### Sync Modes

| Flag              | Behavior                                              |
|-------------------|-------------------------------------------------------|
| (default)         | Full sync: fdatasync data + sync meta page            |
| `MDB_NOSYNC`      | No sync at all — fastest, risk of data loss           |
| `MDB_NOMETASYNC`  | Sync data pages but not meta — good trade-off         |
| `MDB_MAPASYNC`    | Use MS_ASYNC for msync (with WRITEMAP)                |

### Meta Page Sync Strategy

When not using `MDB_WRITEMAP`:
- A separate file descriptor (`me_mfd`) is opened with `O_DSYNC` for meta page writes
- This ensures the meta page write is durable without a separate `fdatasync()` call
- On macOS, `F_FULLFSYNC` is used instead of `fdatasync()`

### WRITEMAP Mode

When `MDB_WRITEMAP` is set:
- Pages are written directly to the mmap (no separate malloc'd copies)
- The dirty flag is still tracked but pages are written in-place
- `msync()` is used for durability
- More efficient but less crash-safe on some systems

---

## 11. Environment Management

### Environment Creation and Opening

1. `mdb_env_create()` — allocate MDB_env, set defaults
2. Configure: `mdb_env_set_mapsize()`, `mdb_env_set_maxdbs()`, `mdb_env_set_maxreaders()`
3. `mdb_env_open()`:
   - Open/create data file and lock file
   - Set up shared locking
   - Read or initialize meta pages
   - `mmap()` the data file
   - `mmap()` the lock file (for reader table)
   - Pre-allocate write transaction structure (`me_txn0`)

### Named Databases (Sub-Databases)

Named databases are stored as entries in MAIN_DBI where:
- Key = database name (string)
- Value = `MDB_db` struct (root page, depth, counts, flags)

Each named DB gets a `dbi` handle (index into arrays). Maximum number set by `mdb_env_set_maxdbs()`.

### Environment Copy/Compaction

`mdb_env_copy()` creates a compacted copy:
1. Start a read transaction
2. Write meta pages to the copy
3. Walk the B+ tree and write pages sequentially
4. The copy has no free pages and is fully compacted

---

## 12. Performance-Critical Design Decisions

1. **Zero-copy reads**: Data is returned as pointers directly into the mmap. No deserialization, no buffer management.

2. **No WAL**: COW B+ tree means the old version is always intact. No log replay on crash recovery.

3. **Single writer simplicity**: No write-write conflicts, no lock contention on data pages, no deadlocks.

4. **Cache-line aligned readers**: Reader slots are 64-byte aligned to eliminate false sharing between threads.

5. **Page recycling**: Freed pages are tracked and reused, with contiguous page search for overflow allocations.

6. **Vectored I/O**: `writev()`/`pwritev()` batches dirty page writes (up to 64 at a time).

7. **Sequential insert optimization**: `MDB_APPEND` flag skips binary search and splits pages asymmetrically for bulk loading.

8. **Loose pages**: Pages freed and re-dirtied in the same transaction are reused immediately without touching the freelist DB.

9. **No malloc for reads**: Read transactions use the mmap directly. Write transactions reuse `me_txn0` and recycle page buffers via `me_dpages` linked list.
