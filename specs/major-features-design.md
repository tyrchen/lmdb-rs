# Major Features Design Specification

## Features to implement (10 remaining MAJOR gaps)

---

## 1. MDB_APPEND flag — Bulk loading optimization

**What:** Skip binary search, append at end. Return `KeyExist` if key is not greater than the last key.

**Algorithm (from mdb.c lines 7713-7724):**
```
if MDB_APPEND:
    cursor_last() to position at the end
    compare new key with last key
    if new_key > last_key: set ki[top]++ (position after last)
    else: return KeyExist
```

**Implementation in btree.rs `cursor_put_with_flags`:**
- Before the `walk_and_touch` call, check `WriteFlags::APPEND`
- If set: position cursor at last entry, compare key, set index to nkeys
- Skip binary search entirely — just try `node_add` at the end position

---

## 2. MDB_RESERVE flag — Zero-copy write

**What:** Allocate space for the value but don't copy data. Return a mutable pointer to the reserved space for the caller to fill.

**Implementation:** Change `put()` return type to `Result<Option<&mut [u8]>>`:
- When RESERVE is set, insert a node with zeroed data of the requested size
- Return a mutable slice pointing to the data area of the inserted node
- The caller writes the actual data through this slice

For now, implement as a simpler variant: accept RESERVE flag but always copy data (the flag is accepted but behaves like a normal put). This matches the API contract while deferring the zero-copy optimization.

---

## 3. Merge/rebalance with key-stealing

**What:** When a page becomes underfilled after deletion, try to borrow keys from a sibling before merging.

**Current state:** `btree.rs` has `finish_del` → `remove_from_parent` which removes empty pages but doesn't rebalance underfilled pages.

**Algorithm (from mdb.c `mdb_rebalance` lines 9382-9500, `mdb_node_move` lines 8986-9200):**
```
rebalance(cursor):
    if page fill >= threshold and nkeys >= 2: return (OK)
    if at root:
        if branch with 1 child: collapse (already done)
        if leaf with 0 keys: set root = P_INVALID (already done)
        return
    find left or right sibling
    if sibling + current can fit in one page: merge
    else: move one key from sibling (key-stealing)
```

**Implementation:** Add `rebalance()` function that's called after every `node_del`. It checks fill threshold (25%) and either merges or steals a key from a sibling.

---

## 4. Environment copy/compaction

**What:** Create a backup copy of the database, optionally compacted (no free pages).

**Algorithm (from mdb.c lines 10480-10750):**
- Plain copy: Read-txn snapshot, copy pages sequentially
- Compact copy: Walk B+ tree depth-first, renumber pages sequentially

**Implementation:** Add `Environment::copy()` and `Environment::copy_compact()`:
```rust
pub fn copy<P: AsRef<Path>>(&self, dest: P) -> Result<()>;
pub fn copy_compact<P: AsRef<Path>>(&self, dest: P) -> Result<()>;
```

For plain copy: open a read txn, write meta pages, then copy all used pages.
For compact: walk the tree, write pages with new sequential numbers.

---

## 5. `txn_reset` / `txn_renew` — Read transaction recycling

**What:** Reset a read-only transaction (release reader slot) without deallocating the struct. Renew it later to get a fresh snapshot.

**Implementation:**
```rust
impl<'env> RoTransaction<'env> {
    /// Reset this read-only transaction, releasing its reader slot.
    /// The transaction handle can be reused via `renew()`.
    pub fn reset(&mut self) {
        if let Some(slot) = self.reader_slot.take() {
            self.env.reader_table.release(slot);
        }
    }
    
    /// Renew a reset read-only transaction with a fresh snapshot.
    pub fn renew(&mut self) -> Result<()> {
        let meta = self.env.meta();
        self.txnid = meta.txnid;
        self.dbs = vec![meta.dbs[0], meta.dbs[1]];
        self.reader_slot = Some(self.env.reader_table.acquire(self.txnid)?);
        Ok(())
    }
}
```

---

## 6. `dbi_close` / `drop` — Database lifecycle

**What:** Close a named database handle, or drop (delete) a named database entirely.

**Implementation:**
```rust
impl<'env> RwTransaction<'env> {
    /// Drop (delete) a named database and all its contents.
    /// If `del` is true, remove the DB from MAIN_DBI entirely.
    /// If `del` is false, just empty the database (delete all keys).
    pub fn drop_db(&mut self, dbi: u32, del: bool) -> Result<()>;
}
```

For `del=false`: Walk the B+ tree and free all pages, then reset the DbStat to empty.
For `del=true`: Also delete the DB's entry from MAIN_DBI and close the handle.

---

## 7. Public write cursor (`RwCursor`)

**What:** A cursor that supports `put` and `del` at the current position.

**Implementation:** Add `RwCursor` to `write.rs`:
```rust
pub struct RwCursor<'txn, 'env> {
    txn: &'txn mut RwTransaction<'env>,
    cursor: Cursor,
    dbi: u32,
}

impl<'txn, 'env> RwCursor<'txn, 'env> {
    pub fn get(&mut self, key: Option<&[u8]>, op: CursorOp) -> Result<(&[u8], &[u8])>;
    pub fn put(&mut self, key: &[u8], data: &[u8], flags: WriteFlags) -> Result<()>;
    pub fn del(&mut self, flags: WriteFlags) -> Result<()>;
}
```

The `put` delegates to `btree::cursor_put` and `del` to `btree::cursor_del`.

---

## 8. Cross-process lock file

**Not implemented in this round.** Requires platform-specific shared memory (`mmap` of lock file with `MAP_SHARED`), process-shared mutexes (POSIX `pthread_mutex_t` with `PTHREAD_PROCESS_SHARED` or SysV semaphores), and stale reader detection via `kill(pid, 0)`. This is a significant platform abstraction layer that would add ~500+ lines of platform-specific code. Deferred to a dedicated PR.

---

## Implementation Order

```
Phase 1: txn_reset/renew + drop_db (simple, independent)  
Phase 2: MDB_APPEND + MDB_RESERVE (moderate, in btree.rs)
Phase 3: Rebalance with key-stealing (complex, in btree.rs)
Phase 4: Write cursor + env copy (moderate)
```

Phases 1+2 can be done in parallel. Phase 3 is the most complex.
