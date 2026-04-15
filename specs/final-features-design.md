# Final Features Design: Lock File, WRITEMAP, RESERVE, Compacting Copy

## 1. Cross-Process Lock File

### Design
On macOS (our primary target), process-shared POSIX mutexes are not supported. LMDB uses SysV semaphores on macOS/BSD. For our implementation, we'll use file-based locking via `flock(2)` for the writer mutex and the existing `ReaderTable` for reader tracking.

### Lock File Format
```
Path: {db_dir}/lock.mdb (or {data_file}-lock for NO_SUB_DIR)
Content: Just a regular file used for flock() advisory locking
```

### Writer Lock
- Use `flock(LOCK_EX | LOCK_NB)` on the lock file for exclusive writer access
- Hold the lock for the entire write transaction duration
- On drop, `flock(LOCK_UN)` releases it

### Reader Tracking
Our existing in-process `ReaderTable` with `AtomicU64` handles single-process readers correctly. For cross-process, we would need shared memory (mmap'd lock file). For now, we create the lock file and use it for writer serialization, while reader tracking remains in-process.

### Implementation
```rust
// In env.rs: open/create lock file alongside data file
// In write.rs: flock() the lock file instead of in-process mutex
```

## 2. WRITEMAP Mode

### Design
In WRITEMAP mode, the mmap is created with `PROT_READ | PROT_WRITE`. Pages are modified directly in the mmap instead of malloc'd dirty buffers. This eliminates the dirty page copy on `page_touch` and the `pwrite` flush on commit.

### Key Differences from Normal Mode
| Operation | Normal | WRITEMAP |
|-----------|--------|----------|
| page_touch | Copy page to new PageBuf | Just set DIRTY flag on mmap page |
| page_alloc | Return new PageBuf | Return pointer into mmap |
| flush_dirty | pwrite each dirty page | Just clear DIRTY flags |
| write_meta | pwrite meta page | Modify mmap directly + msync |

### Implementation
Add a `writemap: bool` flag to RwTransaction. When true:
- `page_touch` modifies pages in-place via mmap (using `MmapRaw::as_mut_ptr()`)
- `page_alloc` returns pointers into the mmap region
- `commit` calls `msync` instead of `pwrite` + `fdatasync`

For safety, we need to be careful about aliasing: only the writer can have mutable access to the mmap, and readers only see committed state.

## 3. MDB_RESERVE Zero-Copy

### Design
MDB_RESERVE allocates space for a value but returns a mutable pointer to the caller to fill in, avoiding a memcpy. The caller must write the data before the transaction commits.

### Implementation
Change `put()` to return `Result<Option<&mut [u8]>>`:
- When RESERVE is NOT set: normal put, return `Ok(None)`
- When RESERVE IS set:
  1. Insert a node with zeroed data of the requested size
  2. Find the inserted node in the dirty page
  3. Return a mutable slice pointing to the data area

For dirty pages (normal mode), the data lives in the `PageBuf`'s Vec, so we can return a mutable slice. The tricky part is lifetime management: the returned slice must not outlive the transaction.

Simpler approach: `reserve()` method that returns `&mut [u8]`:
```rust
pub fn reserve(&mut self, dbi: u32, key: &[u8], data_len: usize) -> Result<&mut [u8]> {
    // Insert a node with zeroed data of data_len
    // Return mutable slice to the data area in the dirty page
}
```

## 4. Compacting Copy

### Design
Walk the B+ tree depth-first, write pages with new sequential numbers. This eliminates free pages and defragments the database.

### Algorithm
```
1. Create output file, write initial meta pages
2. Set next_pgno = NUM_METAS (2)
3. Walk MAIN_DBI tree depth-first:
   a. For each leaf page:
      - Assign new pgno = next_pgno++
      - For F_BIGDATA nodes: copy overflow pages with new pgnos
      - For F_SUBDATA nodes: recursively walk sub-DB tree
      - Write page to output with new pgno
   b. For each branch page:
      - After all children are processed, update child pointers to new pgnos
      - Assign new pgno = next_pgno++
      - Write page to output
4. Write final meta page with new root pgno and last_pgno
```

### Implementation
```rust
pub fn copy_compact<P: AsRef<Path>>(&self, dest: P) -> Result<()> {
    let txn = self.begin_ro_txn()?;
    let mut writer = CompactWriter::new(dest, self.inner.page_size)?;
    // Walk tree and write pages with renumbered pgnos
    writer.walk_db(&txn, MAIN_DBI)?;
    writer.finalize(&txn)?;
    Ok(())
}
```
