# Remaining Features Design Specification

## Overview

This spec covers the 4 missing features identified in the code review, designed from deep analysis of the C LMDB source (`vendors/lmdb/libraries/liblmdb/mdb.c`).

## Phase A: Overflow Pages (Large Values)

**Problem:** Values larger than `(page_size - PAGE_HEADER_SIZE) / MIN_KEYS` (~2KB on 4096-byte pages) cannot fit in a leaf node inline. They must be stored on contiguous overflow pages.

### Design

**Write path (`cursor_put` / `cursor_put_with_flags` in btree.rs):**

1. Before `node_add`, check if `NODE_HEADER_SIZE + key.len() + data.len() > node_max`:
   - If yes, allocate overflow pages: `num_pages = (PAGE_HEADER_SIZE + data.len() - 1) / page_size + 1`
   - Allocate `num_pages` contiguous pages via `page_alloc_multi(txn, num_pages)`
   - Write data starting at `PAGE_HEADER_SIZE` of the first overflow page
   - Set the first page's header: `flags = P_OVERFLOW | P_DIRTY`, `overflow_pages = num_pages`
   - In the leaf node, store only a `u64` pgno (8 bytes) as the data, and set `F_BIGDATA` flag
   - Call `node_add` with data = pgno bytes (8 bytes) and `NodeFlags::BIGDATA`

2. `page_alloc_multi(txn, num)` — allocate `num` contiguous pages:
   - Simple: extend file by `num` pages (allocate from `next_pgno`)
   - Advanced (Phase C): search freelist for contiguous range

**Read path (already partially implemented):**
- `RoTransaction::get()` and `RwTransaction::get()` already check `node.is_bigdata()` and read from the overflow page. But they only read `page_size - PAGE_HEADER_SIZE` bytes. Fix: for multi-page overflow, read `data_size` bytes starting at `PAGE_HEADER_SIZE` of the first overflow page.

**Update path:**
- When overwriting a key with an existing overflow value:
  - If new value also needs overflow and `new_pages <= old_pages`: reuse the overflow pages
  - Otherwise: free old overflow pages, allocate new ones

**Delete path:**
- When deleting a node with `F_BIGDATA`: free the overflow pages by adding them to `txn.free_pgs`

### Implementation in node.rs

Add `node_add_bigdata()`:
```rust
pub fn node_add_bigdata(
    page: &mut [u8], page_size: usize, idx: usize,
    key: &[u8], overflow_pgno: u64, data_size: u32,
) -> Result<()>
```
- Stores `overflow_pgno` as 8-byte LE data, sets `F_BIGDATA` in node flags
- `lo/hi` encode `data_size` (the actual data size, not 8)

### Key constants
```rust
/// Maximum inline data size for a leaf node.
/// Values larger than this go to overflow pages.
fn node_max_data(page_size: usize) -> usize {
    (page_size - PAGE_HEADER_SIZE) / MIN_KEYS - NODE_HEADER_SIZE
}
```

---

## Phase B: Free Page Reuse

**Problem:** Currently, pages freed during a transaction are tracked in `txn.free_pgs` but never reused in subsequent transactions. The database file grows monotonically.

### Design (from C LMDB's `mdb_page_alloc` and `mdb_freelist_save`)

**Core concept:** The FREE_DBI (dbi=0) is a B+ tree keyed by `txnid` (u64), where each value is a sorted array of freed page numbers. When allocating pages, we first check if any freed pages from older transactions can be reused.

**Data structures:**

```rust
/// Accumulated pool of reclaimable pages from old transactions.
/// Sorted descending (largest pgno first).
pghead: Vec<u64>,

/// The txnid of the last FREE_DBI record we consumed.
pglast: u64,

/// Cached oldest active reader txnid.
pgoldest: u64,
```

**Page allocation (`page_alloc` in write.rs):**

```
1. Try loose pages (already freed in this txn)
2. Search pghead for reusable pages
3. If pghead is empty, read more records from FREE_DBI:
   a. Find the oldest active reader txnid (scan reader slots)
   b. Read FREE_DBI records with txnid < oldest_reader
   c. Merge their page lists into pghead
4. If still no pages, extend the file (next_pgno++)
```

**Freelist save (`freelist_save` during commit):**

This is the most complex part. During commit:

1. Write `txn.free_pgs` (pages freed by this txn) to FREE_DBI with key = `txn.txnid`
2. Delete consumed FREE_DBI records (those merged into `pghead`)
3. Reserve space in FREE_DBI for remaining `pghead` entries
4. Fill in the reserved records

The complexity: writing to FREE_DBI can itself allocate/free pages, requiring an iterative loop until stable.

**Simplified approach for our implementation:**

For Phase B, implement a simpler version:
1. On commit, write `txn.free_pgs` to FREE_DBI (key = txnid, value = sorted page list)
2. On page_alloc, read ALL FREE_DBI records with txnid < current_txnid into pghead
3. Pop pages from pghead

This avoids the iterative complexity of the full `freelist_save` but still recycles pages.

### Reader table (simplified)

For single-process use, maintain an in-memory vector of active reader txnids:
```rust
struct ReaderTable {
    slots: Vec<AtomicU64>,  // txnid or u64::MAX if free
}
```

`find_oldest_reader()` scans all slots and returns the minimum non-MAX txnid.

---

## Phase C: Nested Transactions

**Problem:** Users cannot create child transactions that can be independently committed or aborted within a parent write transaction.

### Design (from C LMDB's `mdb_txn_begin` with parent)

**Begin nested txn:**
1. Create a new `RwTransaction` with `parent = Some(parent_txn)`
2. Copy parent's `dbs`, `next_pgno`, `dirty_room`
3. Allocate fresh `dirty_list` and `free_pgs`
4. Save parent's freelist state (`pghead`, `pglast`)
5. Shadow parent's cursors: back up each cursor's state

**Commit nested txn (merge into parent):**
1. Append child's `free_pgs` to parent's
2. Merge child's dirty list into parent's:
   - Remove child's dirty pages from parent's spill list
   - Remove child's spilled pages from parent's dirty list
   - Merge remaining dirty lists (sorted merge)
3. Update parent's `next_pgno`, `dbs`
4. Merge cursor states back to parent

**Abort nested txn:**
1. Discard child's dirty list and free_pgs
2. Restore parent's freelist state
3. Restore parent's cursor states from backups

### Implementation approach

```rust
pub struct RwTransaction<'env> {
    // ... existing fields ...
    parent: Option<Box<ParentState>>,
}

struct ParentState {
    /// Saved parent dbs snapshot
    dbs: Vec<DbStat>,
    /// Saved parent free_pgs
    free_pgs: Vec<u64>,
    /// Saved parent next_pgno  
    next_pgno: u64,
    /// Saved parent dirty list (for abort)
    dirty_snapshot: DirtyPages,
}
```

For simplicity, nested txns share the parent's page resolution:
```rust
fn get_page(&self, pgno: u64) -> Result<*const u8> {
    // Check own dirty list
    if let Some(buf) = self.dirty.find(pgno) { return Ok(buf.as_ptr()); }
    // Check parent's dirty list (if nested)
    if let Some(ref parent) = self.parent_dirty {
        if let Some(buf) = parent.find(pgno) { return Ok(buf.as_ptr()); }
    }
    // Fall back to mmap
    self.env.get_page(pgno)
}
```

---

## Phase D: DUPSORT / DUPFIXED

**Problem:** LMDB supports databases where a single key can have multiple sorted values (duplicates). This is used extensively in OpenLDAP and other applications.

### Design (from C LMDB)

**Storage strategies:**

1. **Single value:** Normal leaf node with key + data. No F_DUPDATA flag.

2. **Inline sub-page (few small dups):** When a key gets its second value, the node's data becomes a mini page (P_SUBP) containing both values. The node has F_DUPDATA flag but NOT F_SUBDATA.

3. **Sub-database (many/large dups):** When the inline sub-page grows too large (exceeds `node_max`), it's promoted to a full B+ tree. The node's data becomes a `DbStat` struct (48 bytes). The node has both F_DUPDATA and F_SUBDATA flags.

**XCursor (sub-cursor):**

```rust
struct XCursor {
    cursor: Cursor,      // sub-cursor for navigating the dup tree
    db: DbStat,          // metadata for the dup tree
    // comparison function for dup values
}
```

Each cursor on a DUPSORT database has an optional `XCursor`. When positioned on a key with duplicates, the XCursor navigates within the dup values.

**DUPFIXED / LEAF2:**

For `MDB_DUPFIXED`, all dup values for a key have the same size. They're stored compactly in LEAF2 pages: no node headers, just contiguous fixed-size values after the page header. The value size is in `mp_pad`.

**Cursor operations:**

| Op | Behavior |
|---|---|
| `FIRST_DUP` | Position xcursor at first dup of current key |
| `LAST_DUP` | Position xcursor at last dup of current key |
| `NEXT_DUP` | Advance xcursor to next dup (same key) |
| `PREV_DUP` | Move xcursor to previous dup (same key) |
| `GET_BOTH` | Position at exact key + exact data match |
| `GET_BOTH_RANGE` | Position at key, then nearest data >= given |
| `GET_MULTIPLE` | Return a page of DUPFIXED values |
| `NEXT_MULTIPLE` | Next page of DUPFIXED values |

**Insert logic (simplified):**

```
cursor_put(key, data):
  search for key in main tree
  if key not found:
    insert normally (single value, no DUPDATA flag)
  else:
    leaf = current node
    if leaf has no DUPDATA:
      # First duplicate — convert single value to sub-page
      old_data = leaf.data
      create sub-page with [old_data, data] sorted
      replace leaf's data with sub-page, set F_DUPDATA
    elif leaf has SUBDATA:
      # Already a sub-DB — insert into it
      xcursor.put(data)
    else:
      # Has sub-page — insert into sub-page
      if sub-page has room:
        insert data into sub-page
      else:
        # Promote to sub-DB
        create new sub-DB B+ tree
        copy all sub-page entries to the new tree
        insert new data into the tree
        replace leaf's data with DbStat, set F_SUBDATA
```

**Delete logic:**

```
cursor_del(key, data):
  if NODUPDATA flag: delete entire key (all dups)
  else:
    delete one dup from sub-page or sub-DB
    if last dup deleted: delete the key entirely
    if sub-DB becomes small enough: demote to sub-page (LMDB doesn't do this)
```

### Implementation plan

1. Add `xcursor: Option<Box<XCursor>>` to `Cursor`
2. Add `init_xcursor()` — called when cursor positions on a key with dups
3. Modify `cursor_put` to handle dup insertion
4. Add sub-page creation and promotion logic
5. Implement all DUPSORT cursor ops
6. For DUPFIXED: support LEAF2 page format in sub-pages

---

## Implementation Order

```
Phase A: Overflow pages        (prerequisite: none)
Phase B: Free page reuse       (prerequisite: none, but A helps)
Phase C: Nested transactions   (prerequisite: none)
Phase D: DUPSORT               (prerequisite: A for large dup values)
```

Phases A and B can be implemented in parallel. Phase D depends on A for large dup values stored as overflow.

---

## Verification

After each phase, run:
1. All existing 189 tests (regression)
2. Phase-specific tests:
   - A: Insert/read/update/delete values from 1 byte to 100KB
   - B: Insert 10K keys, delete 5K, insert 5K more — verify file doesn't grow unbounded
   - C: Nested txn commit/abort preserves correct state
   - D: DUPSORT with 1-100K duplicates per key, cursor iteration, DUPFIXED batch ops
