# lmdb-rs: Technical Design Specification

## 1. On-Disk Format

### 1.1 File Layout

```
┌─────────────────────────────────────────┐
│  Page 0: Meta Page A (P_META)           │  ← Meta ping-pong
│  Page 1: Meta Page B (P_META)           │
├─────────────────────────────────────────┤
│  Pages 2..N: Data Pages                 │  ← B+ tree nodes
│  - FREE_DBI (dbi=0): free page B+ tree │
│  - MAIN_DBI (dbi=1): main database     │
│  - Named sub-databases                  │
│  - Overflow pages for large values      │
└─────────────────────────────────────────┘
```

### 1.2 Page Header (16 bytes on 64-bit)

```rust
#[repr(C)]
struct PageHeader {
    pgno: u64,          // page number (or next pointer for in-memory free list)
    pad: u16,           // key size for LEAF2 pages
    flags: PageFlags,   // u16
    lower: u16,         // end of pointer array (offset from PAGEBASE)
    upper: u16,         // start of node data (offset from PAGEBASE)
}
// For overflow pages, lower+upper is reinterpreted as `overflow_pages: u32`
```

### 1.3 Page Types

```rust
bitflags! {
    struct PageFlags: u16 {
        const BRANCH   = 0x01;   // internal B+ tree node
        const LEAF     = 0x02;   // leaf B+ tree node
        const OVERFLOW = 0x04;   // overflow page for large values
        const META     = 0x08;   // meta page
        const DIRTY    = 0x10;   // page modified (COW copy)
        const LEAF2    = 0x20;   // compact leaf for DUPFIXED
        const SUBPAGE  = 0x40;   // inline sub-page for dup data
        const LOOSE    = 0x4000; // freed in same txn, reusable
        const KEEP     = 0x8000; // don't spill during page spilling
    }
}
```

### 1.4 Node Format (8 bytes header)

```rust
#[repr(C)]
struct NodeHeader {
    // On little-endian: lo is first
    lo: u16,        // low 16 bits of data size (leaf) or child pgno (branch)
    hi: u16,        // high 16 bits
    flags: u16,     // F_BIGDATA | F_SUBDATA | F_DUPDATA (leaf only; pgno bits on branch)
    ksize: u16,     // key size in bytes
    // followed by: key bytes, then data bytes (or pgno for overflow)
}
```

Branch nodes: `child_pgno = lo | (hi << 16) | (flags << 32)` (48-bit page number)
Leaf nodes: `data_size = lo | (hi << 16)` (32-bit size)

### 1.5 Meta Page

```rust
#[repr(C)]
struct MetaPage {
    magic: u32,         // 0xBEEFC0DE
    version: u32,       // format version
    address: u64,       // fixed mapping address (0 = dynamic)
    map_size: u64,      // mmap region size
    dbs: [DbRecord; 2], // FREE_DBI and MAIN_DBI
    last_pgno: u64,     // last used page number
    txnid: u64,         // transaction ID that committed this page
}
```

### 1.6 Database Record (48 bytes)

```rust
#[repr(C)]
struct DbRecord {
    pad: u32,           // key size for LEAF2 pages; page size for FREE_DBI
    flags: u16,         // database flags
    depth: u16,         // B+ tree depth
    branch_pages: u64,  // count of branch pages
    leaf_pages: u64,    // count of leaf pages
    overflow_pages: u64,// count of overflow pages
    entries: u64,       // total entry count
    root: u64,          // root page number (u64::MAX if empty)
}
```

## 2. Memory Architecture

### 2.1 Memory-Mapped Data File

```rust
use memmap2::MmapRaw;

struct MmapManager {
    mmap: MmapRaw,         // the memory map
    map_size: usize,       // total map size
    page_size: usize,      // database page size
    max_pgno: u64,         // map_size / page_size
}

impl MmapManager {
    /// Get a page by number. Returns a raw pointer into the mmap.
    /// Safety: pgno must be < max_pgno, and the returned reference
    /// must not outlive the transaction that provides the snapshot.
    unsafe fn page(&self, pgno: u64) -> &[u8] {
        let offset = pgno as usize * self.page_size;
        &self.mmap[offset..offset + self.page_size]
    }
}
```

### 2.2 Dirty Page Management

For write transactions (non-WRITEMAP mode), modified pages are malloc'd copies:

```rust
/// Sorted map of page number → owned page buffer
struct DirtyList {
    entries: Vec<DirtyEntry>,  // sorted by pgno, ascending
    capacity: usize,           // MDB_IDL_UM_MAX = 131071
}

struct DirtyEntry {
    pgno: u64,
    page: Box<PageBuf>,  // owned copy of the page
}

impl DirtyList {
    fn search(&self, pgno: u64) -> Option<usize>;     // binary search
    fn insert(&mut self, pgno: u64, page: Box<PageBuf>) -> Result<()>;
    fn len(&self) -> usize;
}
```

### 2.3 Page Resolution

When a cursor needs a page, resolution follows this priority:

```
1. Check current txn's dirty_list (binary search)
2. Check parent txn's dirty_list (walk up chain)
3. Check spill list (was it written to disk?)
4. Fall through to mmap (the committed snapshot)
```

```rust
fn get_page(&self, txn: &Transaction, pgno: u64) -> Result<PageRef<'_>> {
    // Walk up the transaction chain
    let mut tx = Some(txn);
    while let Some(t) = tx {
        // Check spill list first (spilled pages should be read from disk)
        if let Some(spill) = &t.spill_pgs {
            if spill.contains(pgno) {
                // Page was spilled — read from mmap
                break;
            }
        }
        // Check dirty list
        if let Some(idx) = t.dirty_list.search(pgno) {
            return Ok(PageRef::Dirty(&t.dirty_list.entries[idx].page));
        }
        tx = t.parent.as_deref();
    }
    // Fall through to mmap
    Ok(PageRef::Mapped(unsafe { self.mmap.page(pgno) }))
}
```

## 3. B+ Tree Implementation

### 3.1 Page Abstraction

```rust
/// A reference to a page, either from the mmap or from a dirty buffer
enum PageRef<'a> {
    Mapped(&'a [u8]),
    Dirty(&'a PageBuf),
}

/// Operations on a page, regardless of source
struct Page<'a> {
    data: &'a [u8],
}

impl<'a> Page<'a> {
    fn header(&self) -> &PageHeader;
    fn flags(&self) -> PageFlags;
    fn num_keys(&self) -> usize;
    fn free_space(&self) -> usize;
    fn node(&self, idx: usize) -> Node<'a>;
    fn leaf2_key(&self, idx: usize, key_size: usize) -> &'a [u8];
}

/// A node within a page
struct Node<'a> {
    header: &'a NodeHeader,
    key: &'a [u8],
    data: NodeData<'a>,
}

enum NodeData<'a> {
    /// Inline data on leaf page
    Inline(&'a [u8]),
    /// Data on overflow page(s)
    Overflow { pgno: u64, size: u32 },
    /// Sub-database
    SubDb(DbRecord),
    /// Sub-page (inline duplicates)
    SubPage(&'a [u8]),
    /// Branch child pointer
    Branch { child_pgno: u64 },
}
```

### 3.2 Cursor

```rust
const CURSOR_STACK_DEPTH: usize = 32;

struct CursorInner {
    /// Page stack from root to current position
    pages: [Option<PageRef>; CURSOR_STACK_DEPTH],
    /// Key index at each level
    indices: [u16; CURSOR_STACK_DEPTH],
    /// Stack depth (number of pushed pages)
    depth: u16,
    /// Top of stack index (depth - 1)
    top: u16,
    /// Cursor state flags
    flags: CursorFlags,
    /// Database handle index
    dbi: u32,
    /// Sub-cursor for DUPSORT databases
    xcursor: Option<Box<XCursor>>,
}

struct XCursor {
    cursor: CursorInner,
    db: DbRecord,
    // comparison function for dup values
}

bitflags! {
    struct CursorFlags: u32 {
        const INITIALIZED = 0x01;
        const EOF         = 0x02;
        const SUB         = 0x04;  // this is a sub-cursor
        const DEL         = 0x08;  // last op was delete
    }
}
```

### 3.3 Binary Search within a Page

```rust
/// Binary search for a key in a page. Returns the index and whether exact match.
fn node_search(
    page: &Page,
    key: &[u8],
    cmp: &dyn Fn(&[u8], &[u8]) -> Ordering,
) -> (usize, bool) {
    let nkeys = page.num_keys();
    if nkeys == 0 {
        return (0, false);
    }

    // For branch pages, skip index 0 (implicit leftmost pointer)
    let low = if page.is_branch() { 1 } else { 0 };
    let mut lo = low;
    let mut hi = nkeys;
    let mut exact = false;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let node_key = page.node_key(mid);
        match cmp(key, node_key) {
            Ordering::Equal => {
                exact = true;
                lo = mid;
                break;
            }
            Ordering::Greater => lo = mid + 1,
            Ordering::Less => hi = mid,
        }
    }
    (lo, exact)
}
```

### 3.4 Page Split Algorithm

The page split is the most complex operation. Key design decisions:

```rust
fn page_split(
    cursor: &mut CursorInner,
    txn: &mut WriteTxn,
    new_key: &[u8],
    new_data: &[u8],
    new_pgno: u64,   // for branch nodes
    flags: NodeFlags,
) -> Result<()> {
    let page = cursor.current_page();
    let new_idx = cursor.current_index();
    let nkeys = page.num_keys();

    // 1. Allocate right sibling
    let right = txn.alloc_page(page.flags() & (PageFlags::BRANCH | PageFlags::LEAF))?;

    // 2. Handle root split
    if cursor.depth <= 1 {
        let new_root = txn.alloc_page(PageFlags::BRANCH)?;
        // Push existing pages down, make new_root the root
        cursor.push_root(new_root);
        cursor.db_mut().root = new_root.pgno();
        cursor.db_mut().depth += 1;
        // Add implicit leftmost pointer to old root
        new_root.add_branch_node(0, &[], page.pgno())?;
    }

    // 3. Find split point
    let split_idx = if flags.contains(NodeFlags::APPEND) {
        // Sequential optimization: put everything on right page
        new_idx
    } else {
        find_split_point(page, new_key, new_data, new_idx, txn.env().page_size())
    };

    // 4. Get separator key
    let sep_key = if split_idx == new_idx {
        new_key.to_vec()
    } else {
        page.node_key(split_idx).to_vec()
    };

    // 5. Insert separator into parent (may recursively split)
    insert_separator_in_parent(cursor, txn, &sep_key, right.pgno())?;

    // 6. Redistribute nodes between left and right pages
    redistribute_nodes(page, right, cursor, new_key, new_data, new_pgno, flags, split_idx)?;

    Ok(())
}

/// Find optimal split point based on actual node sizes
fn find_split_point(
    page: &Page,
    new_key: &[u8],
    new_data: &[u8],
    new_idx: usize,
    page_size: usize,
) -> usize {
    let nkeys = page.num_keys();
    let default_split = (nkeys + 1) / 2;
    let max_space = page_size - PAGE_HEADER_SIZE;

    // For small pages or large keys, refine the split point
    // by summing actual node sizes until we reach ~50% fill
    let new_size = node_size(new_key, new_data);
    let threshold = page_size >> 7; // ~1% of page size

    if nkeys >= threshold && new_size <= max_space / 16 && new_idx < nkeys {
        return default_split;
    }

    // Walk from the beginning, accumulating sizes
    let mut size = 0;
    let mut split = default_split;
    for i in 0..=nkeys {
        let s = if i == new_idx { new_size } else { page.node_total_size(i) };
        size += s;
        if size > max_space / 2 {
            split = i;
            break;
        }
    }
    split
}
```

### 3.5 Page Merge and Rebalance

After deletion, pages may become underfilled:

```rust
fn rebalance(cursor: &mut CursorInner, txn: &mut WriteTxn) -> Result<()> {
    let page = cursor.current_page();
    let nkeys = page.num_keys();

    // Branch pages need at least 2 keys; leaves check fill threshold
    let threshold = if page.is_branch() {
        2
    } else {
        (txn.env().page_size() * FILL_THRESHOLD_PERCENT) / 100
    };

    if page.used_space() >= threshold && nkeys >= 2 {
        return Ok(()); // page is fine
    }

    // Special case: root page
    if cursor.is_at_root() {
        return handle_root_rebalance(cursor, txn);
    }

    // Try to borrow from or merge with a sibling
    let (sibling, from_left) = find_sibling(cursor, txn)?;

    if can_merge(page, sibling) {
        merge_pages(cursor, txn, sibling, from_left)?;
    } else {
        move_node(cursor, txn, sibling, from_left)?;
    }

    Ok(())
}
```

## 4. Transaction System

### 4.1 Read Transaction

```rust
pub struct RoTransaction<'env> {
    env: &'env EnvironmentInner,
    txnid: u64,
    /// Snapshot of database metadata
    dbs: Vec<DbRecord>,
    /// Reader table slot
    reader_slot: ReaderSlotGuard,
    /// Database auxiliary info (comparators, etc.)
    dbxs: Vec<DbAux>,
}

/// RAII guard that releases the reader slot on drop
struct ReaderSlotGuard {
    slot: *mut ReaderSlot,
}

impl Drop for ReaderSlotGuard {
    fn drop(&mut self) {
        unsafe {
            // Mark slot as idle: write txnid = u64::MAX
            (*self.slot).txnid.store(u64::MAX, Ordering::Release);
        }
    }
}
```

### 4.2 Write Transaction

```rust
pub struct RwTransaction<'env> {
    env: &'env EnvironmentInner,
    txnid: u64,
    next_pgno: u64,
    /// Parent transaction (for nested txns)
    parent: Option<Box<RwTransaction<'env>>>,
    /// Pages freed during this transaction
    free_pgs: IdList,
    /// Loose pages: dirty pages freed in same txn, immediately reusable
    loose_pgs: Vec<u64>,
    /// Dirty page list (sorted by pgno)
    dirty_list: DirtyList,
    /// Pages spilled to disk
    spill_pgs: Option<IdList>,
    /// Remaining dirty list capacity
    dirty_room: usize,
    /// Per-database metadata snapshots
    dbs: Vec<DbRecord>,
    /// Per-database flags (dirty, stale, etc.)
    db_flags: Vec<DbStateFlags>,
    /// Active cursors per database (for fixup on page COW)
    cursors: Vec<Vec<*mut CursorInner>>,
    /// Transaction state flags
    flags: TxnFlags,
    /// Writer mutex guard (released on commit/abort)
    _writer_guard: MutexGuard<'env>,
}
```

### 4.3 Copy-on-Write

```rust
fn page_touch(cursor: &mut CursorInner, txn: &mut RwTransaction) -> Result<()> {
    let pgno = cursor.current_pgno();
    let page = cursor.current_page();

    // Already dirty in this txn?
    if page.is_dirty() {
        if let Some(_) = txn.dirty_list.search(pgno) {
            return Ok(());
        }
    }

    // Try unspill first
    if let Some(ref spill) = txn.spill_pgs {
        if unspill_page(txn, pgno)? {
            return Ok(());
        }
    }

    // Record old page as freed
    txn.free_pgs.append(pgno)?;

    // Allocate new page
    let new_page = txn.alloc_page(1)?;
    let new_pgno = new_page.pgno();

    // Copy page contents
    page_copy(&mut new_page, page, txn.env().page_size());
    new_page.set_dirty();

    // Update parent's pointer to new page
    if cursor.depth > 1 {
        let parent_page = cursor.page_at(cursor.top - 1);
        let parent_idx = cursor.index_at(cursor.top - 1);
        parent_page.set_branch_pgno(parent_idx, new_pgno);
    } else {
        cursor.db_mut().root = new_pgno;
    }

    // Fix up all cursors pointing to old page
    for c in txn.cursors[cursor.dbi as usize].iter() {
        unsafe {
            if (**c).page_at(cursor.top).pgno() == pgno {
                (**c).set_page_at(cursor.top, new_page.as_ref());
            }
        }
    }

    cursor.set_current_page(new_page);
    Ok(())
}
```

### 4.4 Transaction Commit Flow

```rust
impl<'env> RwTransaction<'env> {
    pub fn commit(mut self) -> Result<()> {
        // 1. Commit child transactions recursively
        // (handled by nested txn structure)

        // 2. If nested: merge into parent and return
        if let Some(parent) = self.parent.take() {
            return self.merge_into_parent(parent);
        }

        // 3. Close all cursors
        self.close_cursors();

        // 4. Skip if no changes
        if self.dirty_list.is_empty() && !self.flags.contains(TxnFlags::DIRTY) {
            return Ok(());
        }

        // 5. Update named DB roots in MAIN_DBI
        self.update_db_roots()?;

        // 6. Save freelist to FREE_DBI
        self.save_freelist()?;

        // 7. Flush dirty pages to disk
        self.flush_dirty_pages()?;

        // 8. Sync data file
        if !self.flags.contains(TxnFlags::NO_SYNC) {
            self.env.sync_data()?;
        }

        // 9. Write meta page (THE COMMIT POINT)
        self.write_meta()?;

        // 10. Release writer mutex (via Drop of _writer_guard)
        Ok(())
    }
}
```

## 5. Free Page Management

### 5.1 ID List (Sorted Descending)

```rust
/// A sorted list of page IDs, stored in descending order.
/// The first element is the count. Allocated capacity is stored
/// at a "hidden" position before the array start.
struct IdList {
    /// Sorted descending. ids[0] is unused (matches C layout convenience).
    ids: Vec<u64>,
}

impl IdList {
    fn new() -> Self;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;

    /// Binary search for an ID. Returns index if found.
    fn search(&self, id: u64) -> usize;

    /// Append without maintaining sort order (sort later)
    fn append(&mut self, id: u64) -> Result<()>;

    /// Append a range [id, id+1, ..., id+n-1]
    fn append_range(&mut self, id: u64, n: u32) -> Result<()>;

    /// Sort in descending order (quicksort + insertion sort hybrid)
    fn sort(&mut self);

    /// Merge another sorted IDL into this one (both must be sorted descending)
    fn merge(&mut self, other: &IdList);

    /// First (largest) ID
    fn first(&self) -> Option<u64>;

    /// Last (smallest) ID
    fn last(&self) -> Option<u64>;
}
```

### 5.2 Page Allocator

```rust
fn alloc_pages(
    cursor: &mut CursorInner,
    txn: &mut RwTransaction,
    num: usize,
) -> Result<AllocatedPage> {
    // Priority 1: Loose pages (single page only)
    if num == 1 {
        if let Some(pgno) = txn.loose_pgs.pop() {
            return Ok(AllocatedPage::from_loose(pgno));
        }
    }

    // Check dirty_room
    if txn.dirty_room == 0 {
        return Err(Error::TxnFull);
    }

    // Priority 2: Reclaimed free pages from me_pghead
    let env = txn.env();
    if let Some(pgno) = env.find_free_pages(num, txn)? {
        let page = txn.materialize_page(pgno, num)?;
        return Ok(page);
    }

    // Priority 3: Read more freeDB records
    let oldest = env.find_oldest_reader();
    env.load_free_records(txn, oldest)?;

    if let Some(pgno) = env.find_free_pages(num, txn)? {
        let page = txn.materialize_page(pgno, num)?;
        return Ok(page);
    }

    // Priority 4: Extend the file
    let pgno = txn.next_pgno;
    if pgno + num as u64 > env.max_pgno() {
        return Err(Error::MapFull);
    }
    txn.next_pgno += num as u64;
    let page = txn.materialize_page(pgno, num)?;
    Ok(page)
}
```

### 5.3 Freelist Save (The Complex Part)

```rust
fn save_freelist(txn: &mut RwTransaction) -> Result<()> {
    let env = txn.env();
    let max_free_per_page = env.max_free_per_page();

    // Open cursor on FREE_DBI
    let mut mc = txn.open_internal_cursor(FREE_DBI)?;

    // Phase 1: Move loose pages to free_pgs or pghead
    flush_loose_pages(txn, &mut mc)?;

    loop {
        // Phase 2: Delete consumed freeDB records
        delete_consumed_records(txn, &mut mc)?;

        // Phase 3: Write this txn's freed pages
        if !txn.free_pgs.is_empty() {
            let key = txn.txnid.to_ne_bytes();
            let data = txn.free_pgs.as_bytes();
            // Use MDB_RESERVE to avoid double copy
            let reserved = mc.put_reserve(&key, data.len(), WriteFlags::empty())?;
            txn.free_pgs.sort();
            reserved.copy_from_slice(&txn.free_pgs.as_bytes());
        }

        // Phase 4: Reserve space for reclaimed pages (pghead)
        let pghead_len = env.pghead_len();
        if total_reserved >= pghead_len {
            break; // enough space reserved
        }

        // Reserve records with keys [1..pglast]
        reserve_pghead_records(txn, &mut mc, max_free_per_page)?;
    }

    // Phase 5: Fill in reserved records with actual page numbers
    fill_reserved_records(txn, &mut mc)?;

    Ok(())
}
```

## 6. Concurrency Design

### 6.1 Reader Table (Shared Memory)

```rust
/// Cache-line aligned reader slot (64 bytes)
#[repr(C, align(64))]
struct ReaderSlot {
    txnid: AtomicU64,     // current snapshot txnid, or u64::MAX
    pid: AtomicU32,       // owning process ID (0 = free)
    tid: AtomicU64,       // owning thread ID
    _padding: [u8; 38],   // pad to 64 bytes
}

/// Lock file header
#[repr(C)]
struct LockFileHeader {
    magic: u32,
    format: u32,
    txnid: AtomicU64,        // last committed txnid
    num_readers: AtomicU32,  // high-water mark of reader slots used
    _pad1: [u8; CACHELINE - 24],
    // Writer mutex follows (platform-specific)
    // Then reader mutex
    // Then reader slots array
}
```

### 6.2 Writer Serialization

```rust
/// Single-process writer mutex using std::sync::Mutex
/// For cross-process: use file-based locking or shared memory mutex
struct WriterLock {
    mutex: Mutex<()>,
}

impl WriterLock {
    fn acquire(&self) -> MutexGuard<'_, ()> {
        self.mutex.lock().unwrap_or_else(|e| e.into_inner())
    }
}
```

### 6.3 Finding Oldest Reader

```rust
fn find_oldest_reader(&self) -> u64 {
    let num_readers = self.lock_header().num_readers.load(Ordering::Acquire);
    let mut oldest = u64::MAX;

    for i in 0..num_readers as usize {
        let slot = &self.reader_slots()[i];
        let pid = slot.pid.load(Ordering::Acquire);
        if pid == 0 { continue; } // free slot

        let txnid = slot.txnid.load(Ordering::Acquire);
        if txnid < oldest {
            oldest = txnid;
        }
    }

    oldest
}
```

### 6.4 Thread Safety Model

```rust
// Environment is shareable across threads
unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

// Read-only transactions can be sent between threads (with NO_TLS)
unsafe impl<'env> Send for RoTransaction<'env> {}
// But not shared — only one thread uses it at a time
// impl !Sync for RoTransaction<'env> {}

// Write transactions can be sent but not shared
unsafe impl<'env> Send for RwTransaction<'env> {}
// impl !Sync for RwTransaction<'env> {}

// Data references are tied to transaction lifetime
// &'txn [u8] from get() cannot outlive the transaction
```

## 7. Key Comparison Functions

```rust
/// Built-in comparison functions matching LMDB's behavior
pub enum Comparator {
    /// Default: lexicographic memcmp with length tiebreaker
    Lexicographic,
    /// Reverse byte order comparison
    ReverseLexicographic,
    /// Native integer comparison (4 or 8 bytes)
    Integer,
    /// Custom comparison function
    Custom(Arc<dyn Fn(&[u8], &[u8]) -> Ordering + Send + Sync>),
}

impl Comparator {
    fn compare(&self, a: &[u8], b: &[u8]) -> Ordering {
        match self {
            Self::Lexicographic => {
                let len = a.len().min(b.len());
                match a[..len].cmp(&b[..len]) {
                    Ordering::Equal => a.len().cmp(&b.len()),
                    other => other,
                }
            }
            Self::ReverseLexicographic => {
                // Compare from end to beginning
                let len = a.len().min(b.len());
                for i in (0..len).rev() {
                    match a[a.len() - 1 - i].cmp(&b[b.len() - 1 - i]) {
                        Ordering::Equal => continue,
                        other => return other,
                    }
                }
                a.len().cmp(&b.len())
            }
            Self::Integer => {
                // Compare as native unsigned integers
                debug_assert!(a.len() == b.len());
                match a.len() {
                    4 => {
                        let a = u32::from_ne_bytes(a.try_into().unwrap());
                        let b = u32::from_ne_bytes(b.try_into().unwrap());
                        a.cmp(&b)
                    }
                    8 => {
                        let a = u64::from_ne_bytes(a.try_into().unwrap());
                        let b = u64::from_ne_bytes(b.try_into().unwrap());
                        a.cmp(&b)
                    }
                    _ => unreachable!("integer key must be 4 or 8 bytes"),
                }
            }
            Self::Custom(f) => f(a, b),
        }
    }
}
```

## 8. I/O Layer

### 8.1 Vectored Page Flush

```rust
use std::io::IoSlice;

fn flush_dirty_pages(txn: &RwTransaction) -> Result<()> {
    let page_size = txn.env().page_size();
    let fd = txn.env().data_fd();

    let mut batch: Vec<IoSlice<'_>> = Vec::with_capacity(64);
    let mut batch_offset: u64 = 0;
    let mut batch_size: usize = 0;

    for entry in txn.dirty_list.iter() {
        if entry.page.flags().intersects(PageFlags::LOOSE | PageFlags::KEEP) {
            continue;
        }

        let offset = entry.pgno * page_size as u64;
        let size = if entry.page.is_overflow() {
            entry.page.overflow_pages() as usize * page_size
        } else {
            page_size
        };

        // If contiguous with previous, extend batch
        let contiguous = batch_offset + batch_size as u64 == offset;
        if !contiguous && !batch.is_empty() || batch.len() >= 64 {
            // Flush current batch
            pwritev(fd, &batch, batch_offset)?;
            batch.clear();
            batch_size = 0;
        }

        if batch.is_empty() {
            batch_offset = offset;
        }

        batch.push(IoSlice::new(entry.page.as_bytes()));
        batch_size += size;
        entry.page.clear_dirty();
    }

    // Flush remaining
    if !batch.is_empty() {
        pwritev(fd, &batch, batch_offset)?;
    }

    Ok(())
}
```

### 8.2 Meta Page Write (Atomic Commit Point)

```rust
fn write_meta(txn: &RwTransaction) -> Result<()> {
    let env = txn.env();
    let toggle = (txn.txnid & 1) as usize; // meta page 0 or 1

    let mut meta = MetaPage {
        magic: META_MAGIC,
        version: DATA_VERSION,
        address: 0,
        map_size: env.map_size() as u64,
        dbs: [txn.dbs[FREE_DBI], txn.dbs[MAIN_DBI]],
        last_pgno: txn.next_pgno - 1,
        txnid: txn.txnid,
    };

    // Write only the mutable portion (from map_size onward)
    let offset = (toggle * env.page_size()) as u64
        + PAGE_HEADER_SIZE as u64
        + offset_of!(MetaPage, map_size) as u64;

    let data = &meta.as_bytes()[offset_of!(MetaPage, map_size)..];

    // Write to the sync fd (O_DSYNC) for atomic durability
    pwrite(env.sync_fd(), data, offset)?;

    // Update shared state in lock file
    env.lock_header().txnid.store(txn.txnid, Ordering::Release);

    Ok(())
}
```

## 9. DUPSORT Implementation

### 9.1 Storage Strategies

Duplicate values for a single key can be stored in two ways:

1. **Sub-page (inline):** For small sets of small duplicates, the values are stored in a miniature page embedded in the leaf node's data area. The node has `F_DUPDATA` flag but NOT `F_SUBDATA`.

2. **Sub-database:** For larger sets, the duplicates are promoted to a full B+ tree. The leaf node's data contains a `DbRecord` struct. The node has both `F_DUPDATA` and `F_SUBDATA` flags.

### 9.2 Promotion Threshold

A sub-page is promoted to a sub-database when:
- The sub-page would exceed the node's available space
- Adding a new duplicate would overflow the current page

### 9.3 Sub-cursor Integration

```rust
fn init_xcursor(cursor: &mut CursorInner, node: &Node) {
    if let Some(ref mut xc) = cursor.xcursor {
        match node.data {
            NodeData::SubPage(subpage_data) => {
                // Point sub-cursor at the inline sub-page
                xc.cursor.pages[0] = Some(PageRef::SubPage(subpage_data));
                xc.cursor.depth = 1;
                xc.cursor.top = 0;
                xc.cursor.flags |= CursorFlags::INITIALIZED;
            }
            NodeData::SubDb(db_record) => {
                // Set up sub-cursor with its own B+ tree
                xc.db = db_record;
                xc.cursor.depth = 0;
                xc.cursor.flags &= !CursorFlags::INITIALIZED;
                // Will be initialized on first access via page_search
            }
            _ => {}
        }
    }
}
```

## 10. Compacting Copy

### 10.1 Architecture

Uses a producer-consumer pattern with double buffering:

```rust
fn compact_copy(env: &Environment, dest_fd: RawFd) -> Result<()> {
    let txn = env.begin_ro_txn()?;
    let (tx, rx) = std::sync::mpsc::sync_channel::<CopyBuffer>(2);

    // Writer thread: consumes buffers and writes to fd
    let writer = std::thread::spawn(move || -> Result<()> {
        for buf in rx {
            pwrite_all(dest_fd, &buf.data, buf.offset)?;
        }
        Ok(())
    });

    // Walk the B+ tree depth-first, renumbering pages sequentially
    let mut next_pgno = NUM_METAS as u64;
    let mut buffer = CopyBuffer::new(BUFFER_SIZE);

    // Write meta pages first
    write_initial_meta(&mut buffer, &txn)?;

    // Walk MAIN_DBI tree
    walk_tree_compact(&txn, MAIN_DBI, &mut next_pgno, &mut buffer, &tx)?;

    // Final meta with correct roots
    write_final_meta(&mut buffer, &txn, next_pgno - 1)?;

    drop(tx); // signal EOF
    writer.join().unwrap()?;

    Ok(())
}
```

## 11. Error Recovery

### 11.1 Crash Recovery

LMDB's design provides automatic crash recovery:

1. The data file always contains a valid previous state
2. Meta pages alternate: txn N writes meta page N%2
3. On startup, read both meta pages, pick the one with the higher valid txnid
4. All pages referenced by the chosen meta are guaranteed consistent (COW ensures this)
5. No WAL replay, no recovery log — just pick the right meta page

### 11.2 Fatal Error Handling

```rust
enum EnvHealth {
    Healthy,
    /// A previous meta write failed; environment is read-only until reopened
    FatalError,
}

// Check before every write operation
fn check_env_health(env: &EnvironmentInner) -> Result<()> {
    if env.health.load(Ordering::Acquire) == EnvHealth::FatalError as u8 {
        return Err(Error::Panic);
    }
    Ok(())
}
```
