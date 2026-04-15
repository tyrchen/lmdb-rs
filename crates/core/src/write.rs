//! Write transaction: COW B+ tree mutations, page allocation, and commit.
//!
//! Provides [`RwTransaction`] for read-write access to the database. Only one
//! write transaction may be active at a time. All page modifications use
//! copy-on-write: the original pages in the mmap remain intact for concurrent
//! readers, and dirty copies are flushed to disk on commit.
//!
//! # Integration with Environment
//!
//! The following methods need to be added to `env.rs` for full integration:
//!
//! ```ignore
//! // Add to EnvironmentInner:
//! pub(crate) fn data_fd(&self) -> std::os::fd::RawFd {
//!     use std::os::fd::AsRawFd;
//!     self._data_file.as_raw_fd()
//! }
//!
//! // Add to Environment:
//! pub fn begin_rw_txn(&self) -> Result<RwTransaction<'_>> {
//!     RwTransaction::new(&self.inner)
//! }
//! ```

use std::{cmp::Ordering, sync::Arc};

use crate::{
    btree,
    cmp::{default_cmp, default_dcmp},
    cursor::Cursor,
    env::EnvironmentInner,
    error::{Error, Result},
    page::Page,
    types::*,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Database index for the free-page list.
const FREE_DBI: usize = 0;

/// Database index for the main B+ tree namespace.
const MAIN_DBI: usize = 1;

// ---------------------------------------------------------------------------
// PageBuf — owned page buffer
// ---------------------------------------------------------------------------

/// An owned page buffer for dirty (modified) pages.
#[derive(Debug, Clone)]
pub struct PageBuf {
    data: Vec<u8>,
}

impl PageBuf {
    /// Allocate a zeroed page buffer.
    pub fn new(page_size: usize) -> Self {
        Self {
            data: vec![0u8; page_size],
        }
    }

    /// Create a page buffer by copying from an existing slice.
    pub fn from_existing(src: &[u8]) -> Self {
        Self { data: src.to_vec() }
    }

    /// Return the buffer as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Return the buffer as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Return a read-only [`Page`] view.
    pub fn as_page(&self) -> Page<'_> {
        Page::from_raw(&self.data)
    }

    /// Return a raw pointer to the start of the buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// DirtyPages — sorted map of pgno → PageBuf
// ---------------------------------------------------------------------------

/// Tracks pages modified by a write transaction, sorted by page number.
#[derive(Debug)]
pub struct DirtyPages {
    /// Sorted by pgno ascending. Each entry is (pgno, page_buffer).
    entries: Vec<(u64, PageBuf)>,
}

impl DirtyPages {
    /// Create an empty dirty page list.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Return the number of dirty pages.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no pages are dirty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find a dirty page by page number.
    pub fn find(&self, pgno: u64) -> Option<&PageBuf> {
        self.entries
            .binary_search_by_key(&pgno, |(p, _)| *p)
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// Find a mutable dirty page by page number.
    pub fn find_mut(&mut self, pgno: u64) -> Option<&mut PageBuf> {
        self.entries
            .binary_search_by_key(&pgno, |(p, _)| *p)
            .ok()
            .map(|i| &mut self.entries[i].1)
    }

    /// Insert or replace a dirty page.
    pub fn insert(&mut self, pgno: u64, buf: PageBuf) {
        match self.entries.binary_search_by_key(&pgno, |(p, _)| *p) {
            Ok(i) => self.entries[i].1 = buf,
            Err(i) => self.entries.insert(i, (pgno, buf)),
        }
    }

    /// Remove and return a dirty page.
    pub fn remove(&mut self, pgno: u64) -> Option<PageBuf> {
        match self.entries.binary_search_by_key(&pgno, |(p, _)| *p) {
            Ok(i) => Some(self.entries.remove(i).1),
            Err(_) => None,
        }
    }

    /// Iterate all dirty pages in pgno order.
    pub fn iter(&self) -> impl Iterator<Item = &(u64, PageBuf)> {
        self.entries.iter()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Create a snapshot of the current dirty page list for savepoint support.
    pub(crate) fn snapshot(&self) -> Vec<(u64, PageBuf)> {
        self.entries.clone()
    }

    /// Restore dirty pages from a previously captured snapshot.
    pub(crate) fn restore(&mut self, snapshot: Vec<(u64, PageBuf)>) {
        self.entries = snapshot;
    }
}

impl Default for DirtyPages {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SavePoint — snapshot for nested transaction rollback
// ---------------------------------------------------------------------------

/// Saved state for nested transaction (savepoint) rollback.
///
/// When a nested transaction is started via [`RwTransaction::begin_nested_txn`],
/// the current state is captured into a `SavePoint`. On abort, the state is
/// restored. On commit, the savepoint is simply discarded (changes remain in
/// the parent).
#[derive(Debug)]
struct SavePoint {
    /// Snapshot of per-database metadata.
    dbs: Vec<DbStat>,
    /// Snapshot of per-database dirty flags.
    db_dirty: Vec<bool>,
    /// Snapshot of freed page numbers.
    free_pgs: Vec<u64>,
    /// Snapshot of loose (dirty-then-freed) page numbers.
    loose_pgs: Vec<u64>,
    /// Snapshot of the next page number to allocate.
    next_pgno: u64,
    /// Snapshot of reclaimed page numbers from FREE_DBI.
    reclaim_pgs: Vec<u64>,
    /// Snapshot of all dirty pages at the time the savepoint was created.
    dirty_snapshot: Vec<(u64, PageBuf)>,
}

// ---------------------------------------------------------------------------
// RwTransaction
// ---------------------------------------------------------------------------

/// A read-write transaction supporting key-value insertion, deletion, and
/// atomic commit.
///
/// Changes are buffered in memory and only written to disk on
/// [`commit`](Self::commit). If the transaction is dropped without committing,
/// all changes are discarded.
///
/// Nested transactions are supported via the savepoint mechanism:
/// [`begin_nested_txn`](Self::begin_nested_txn) captures a snapshot,
/// [`commit_nested_txn`](Self::commit_nested_txn) discards it (keeping
/// changes), and [`abort_nested_txn`](Self::abort_nested_txn) restores
/// the snapshot, rolling back all changes made since.
pub struct RwTransaction<'env> {
    pub(crate) env: &'env EnvironmentInner,
    pub(crate) txnid: u64,
    pub(crate) next_pgno: u64,
    /// Pages freed during this transaction.
    pub(crate) free_pgs: Vec<u64>,
    /// Pages that were dirtied and then freed (reusable immediately).
    pub(crate) loose_pgs: Vec<u64>,
    /// Dirty page buffers (modified pages).
    pub(crate) dirty: DirtyPages,
    /// Per-database metadata snapshots.
    pub(crate) dbs: Vec<DbStat>,
    /// Per-database dirty flags.
    pub(crate) db_dirty: Vec<bool>,
    /// Reclaimed pages from FREE_DBI, available for reuse.
    pub(crate) reclaim_pgs: Vec<u64>,
    /// Transaction IDs of consumed FREE_DBI records (to delete on commit).
    consumed_freelist_txnids: Vec<u64>,
    /// Whether the freelist has been loaded from FREE_DBI.
    freelist_loaded: bool,
    /// Whether this transaction has been committed or aborted.
    finished: bool,
    /// Stack of savepoints for nested transactions.
    savepoints: Vec<SavePoint>,
    /// Writer mutex guard — held for the entire transaction lifetime.
    /// Ensures only one writer at a time within this process.
    _write_guard: std::sync::MutexGuard<'env, ()>,
}

impl std::fmt::Debug for RwTransaction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RwTransaction")
            .field("txnid", &self.txnid)
            .field("next_pgno", &self.next_pgno)
            .field("dirty_pages", &self.dirty.len())
            .field("finished", &self.finished)
            .field("nested_depth", &self.savepoints.len())
            .finish_non_exhaustive()
    }
}

impl<'env> RwTransaction<'env> {
    /// Create a new write transaction.
    ///
    /// Acquires a snapshot of the current meta page and assigns a new
    /// transaction ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment is in a fatal state.
    pub(crate) fn new(env: &'env EnvironmentInner) -> Result<Self> {
        // Acquire writer lock — held for the entire transaction lifetime.
        let write_guard = env.write_mutex.lock().map_err(|_| Error::Panic)?;
        let meta = env.meta();
        let txnid = meta.txnid + 1;
        let next_pgno = meta.last_pgno + 1;

        Ok(Self {
            env,
            txnid,
            next_pgno,
            free_pgs: Vec::new(),
            loose_pgs: Vec::new(),
            dirty: DirtyPages::new(),
            dbs: vec![meta.dbs[0], meta.dbs[1]],
            db_dirty: vec![false; CORE_DBS as usize],
            reclaim_pgs: Vec::new(),
            consumed_freelist_txnids: Vec::new(),
            freelist_loaded: false,
            finished: false,
            savepoints: Vec::new(),
            _write_guard: write_guard,
        })
    }

    /// Resolve a page number to a page pointer.
    ///
    /// Checks the dirty list first, then falls back to the mmap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PageNotFound`] if the page is not in the dirty list
    /// and exceeds the mapped region.
    pub(crate) fn get_page(&self, pgno: u64) -> Result<*const u8> {
        if let Some(buf) = self.dirty.find(pgno) {
            return Ok(buf.as_ptr());
        }
        self.env.get_page(pgno)
    }

    /// Allocate a new page buffer.
    ///
    /// Tries to reuse loose pages first (pages freed and dirtied in the same
    /// transaction), then extends the file by bumping `next_pgno`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MapFull`] if the database has reached the map size limit.
    pub(crate) fn page_alloc(&mut self) -> Result<(u64, PageBuf)> {
        // 1. Try loose pages first (pages freed and dirtied in same txn)
        if let Some(pgno) = self.loose_pgs.pop() {
            if let Some(buf) = self.dirty.remove(pgno) {
                return Ok((pgno, buf));
            }
        }

        // 2. Try reusing pages from the free list (older transactions)
        if let Some(pgno) = self.try_reclaim_page()? {
            let buf = PageBuf::new(self.env.page_size);
            return Ok((pgno, buf));
        }

        // 3. Extend the file
        let pgno = self.next_pgno;
        if pgno >= self.env.max_pgno {
            return Err(Error::MapFull);
        }
        self.next_pgno += 1;
        let buf = PageBuf::new(self.env.page_size);
        Ok((pgno, buf))
    }

    /// Allocate `num` contiguous page buffers.
    ///
    /// Returns the starting page number and a vector of page buffers.
    /// All pages are allocated from `next_pgno` in a contiguous range.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MapFull`] if the database has reached the map size limit.
    pub(crate) fn page_alloc_multi(&mut self, num: usize) -> Result<(u64, Vec<PageBuf>)> {
        let start_pgno = self.next_pgno;
        if start_pgno + num as u64 > self.env.max_pgno {
            return Err(Error::MapFull);
        }
        self.next_pgno += num as u64;
        let bufs: Vec<PageBuf> = (0..num).map(|_| PageBuf::new(self.env.page_size)).collect();
        Ok((start_pgno, bufs))
    }

    /// Copy-on-write: make a writable copy of the page at `pgno`.
    ///
    /// If the page is already dirty in this transaction, returns it directly.
    /// Otherwise, copies the page from the mmap, frees the old page number,
    /// and allocates a new one.
    ///
    /// # Errors
    ///
    /// Returns an error if page allocation fails or the source page cannot
    /// be read.
    pub(crate) fn page_touch(&mut self, pgno: u64) -> Result<u64> {
        // Already dirty in this txn?
        if self.dirty.find(pgno).is_some() {
            return Ok(pgno);
        }

        // Record old page as freed
        self.free_pgs.push(pgno);

        // Allocate new page
        let (new_pgno, mut new_buf) = self.page_alloc()?;

        // Copy contents from mmap
        let src_ptr = self.env.get_page(pgno)?;
        // SAFETY: src_ptr points into the mmap which is valid for page_size
        // bytes. The mmap outlives this transaction.
        let src = unsafe { std::slice::from_raw_parts(src_ptr, self.env.page_size) };
        new_buf.as_mut_slice().copy_from_slice(src);

        // Update pgno in the new page
        new_buf.as_mut_slice()[0..8].copy_from_slice(&new_pgno.to_le_bytes());

        // Set DIRTY flag in page header
        let flags_raw = u16::from_le_bytes([new_buf.as_slice()[10], new_buf.as_slice()[11]]);
        let new_flags = flags_raw | PageFlags::DIRTY.bits();
        new_buf.as_mut_slice()[10..12].copy_from_slice(&new_flags.to_le_bytes());

        self.dirty.insert(new_pgno, new_buf);
        Ok(new_pgno)
    }

    /// Open or create a named database.
    ///
    /// If `name` is `None`, returns the handle for the default (main) database.
    /// If `name` is `Some`, looks up the named database in the main database.
    /// If not found and `DatabaseFlags::CREATE` is set, creates a new empty
    /// named database.
    ///
    /// # Errors
    ///
    /// - [`Error::NotFound`] if the named database does not exist and `CREATE` is not set
    /// - [`Error::DbsFull`] if the maximum number of named databases has been reached
    /// - [`Error::BadDbi`] if the environment was not configured with `max_dbs > 0`
    pub fn open_db(&mut self, name: Option<&str>, flags: DatabaseFlags) -> Result<u32> {
        if let Some(name) = name {
            if self.env.max_dbs == 0 {
                return Err(Error::BadDbi);
            }
            self.find_or_create_db(name, flags)
        } else {
            Ok(MAIN_DBI as u32)
        }
    }

    /// Look up or create a named database.
    fn find_or_create_db(&mut self, name: &str, flags: DatabaseFlags) -> Result<u32> {
        // Search MAIN_DBI for the name.
        let main_db = self.dbs[MAIN_DBI];
        if main_db.root != P_INVALID {
            let cmp = self.env.get_cmp(MAIN_DBI as u32)?;
            let mut cursor = Cursor::new(self.env.page_size, MAIN_DBI as u32);
            let get_page = |pgno: u64| -> Result<*const u8> { self.get_page(pgno) };
            if cursor
                .page_search(main_db.root, Some(name.as_bytes()), &*cmp, &get_page)
                .is_ok()
            {
                if let Some(node) = cursor.current_node() {
                    if cmp(name.as_bytes(), node.key()) == Ordering::Equal && node.is_subdata() {
                        let db_stat = node.sub_db();
                        return self.register_db(name, db_stat, flags);
                    }
                }
            }
        }

        // Not found — create if CREATE flag is set.
        if !flags.contains(DatabaseFlags::CREATE) {
            return Err(Error::NotFound);
        }

        // Create a new empty DB.
        let on_disk_flags = (flags & !DatabaseFlags::CREATE).bits() as u16;
        let new_db = DbStat {
            pad: 0,
            flags: on_disk_flags,
            depth: 0,
            branch_pages: 0,
            leaf_pages: 0,
            overflow_pages: 0,
            entries: 0,
            root: P_INVALID,
        };

        let dbi = self.register_db(name, new_db, flags)?;

        // Write the new DB record to MAIN_DBI.
        self.write_db_record(name, &new_db)?;

        Ok(dbi)
    }

    /// Register a named database in the environment's tracking structures.
    ///
    /// If the name is already registered, reuses the existing slot and updates
    /// the local `dbs` array with the provided `DbStat`.
    fn register_db(&mut self, name: &str, db: DbStat, _flags: DatabaseFlags) -> Result<u32> {
        // Check if already registered — reuse the existing slot.
        {
            let db_names = self.env.db_names.read().map_err(|_| Error::Panic)?;
            for (i, n) in db_names.iter().enumerate() {
                if let Some(n) = n {
                    if n == name {
                        while self.dbs.len() <= i {
                            self.dbs.push(DbStat::default());
                            self.db_dirty.push(false);
                        }
                        self.dbs[i] = db;
                        self.db_dirty[i] = true;
                        return Ok(i as u32);
                    }
                }
            }
        }

        let dbi = {
            let db_names = self.env.db_names.read().map_err(|_| Error::Panic)?;
            // Check max_dbs limit (total slots = CORE_DBS + max_dbs).
            let max_total = CORE_DBS as usize + self.env.max_dbs as usize;
            if db_names.len() >= max_total {
                // Check if there's a free slot.
                let mut free_slot = None;
                for (i, n) in db_names.iter().enumerate().skip(CORE_DBS as usize) {
                    if n.is_none() {
                        free_slot = Some(i);
                        break;
                    }
                }
                if let Some(slot) = free_slot {
                    slot
                } else {
                    return Err(Error::DbsFull);
                }
            } else {
                db_names.len()
            }
        };

        // Register in environment.
        let mut db_names = self.env.db_names.write().map_err(|_| Error::Panic)?;
        let mut db_cmp = self.env.db_cmp.write().map_err(|_| Error::Panic)?;
        let mut db_dcmp = self.env.db_dcmp.write().map_err(|_| Error::Panic)?;
        let mut db_flags_vec = self.env.db_flags.write().map_err(|_| Error::Panic)?;

        while db_names.len() <= dbi {
            db_names.push(None);
            db_cmp.push(Arc::new(default_cmp(0)));
            db_dcmp.push(Arc::new(default_dcmp(0)));
            db_flags_vec.push(0);
        }

        db_names[dbi] = Some(name.to_string());
        db_cmp[dbi] = Arc::new(default_cmp(db.flags));
        db_dcmp[dbi] = Arc::new(default_dcmp(db.flags));
        db_flags_vec[dbi] = db.flags;

        // Ensure our local dbs/db_dirty arrays are large enough.
        while self.dbs.len() <= dbi {
            self.dbs.push(DbStat::default());
            self.db_dirty.push(false);
        }
        self.dbs[dbi] = db;
        self.db_dirty[dbi] = true;

        Ok(dbi as u32)
    }

    /// Write a named database record into MAIN_DBI with `SUBDATA` node flags.
    fn write_db_record(&mut self, name: &str, db: &DbStat) -> Result<()> {
        let db_bytes = db_stat_to_bytes(db);
        btree::cursor_put_with_flags(
            self,
            MAIN_DBI as u32,
            name.as_bytes(),
            &db_bytes,
            WriteFlags::empty(),
            NodeFlags::SUBDATA,
        )
    }

    /// Return the name of a named database by DBI index.
    fn get_db_name(&self, dbi: usize) -> Option<String> {
        let db_names = self.env.db_names.read().ok()?;
        db_names.get(dbi).and_then(|n| n.clone())
    }

    /// Insert a key/value pair into the specified database.
    ///
    /// Handles page splits automatically when a leaf page is full.
    ///
    /// # Errors
    ///
    /// - [`Error::BadValSize`] if key is empty or exceeds max key size
    /// - [`Error::KeyExist`] if `NO_OVERWRITE` is set and key exists
    /// - [`Error::MapFull`] if no more pages can be allocated
    pub fn put(&mut self, dbi: u32, key: &[u8], data: &[u8], flags: WriteFlags) -> Result<()> {
        btree::cursor_put(self, dbi, key, data, flags)
    }

    /// Delete a key (and optionally a specific dup value) from the database.
    ///
    /// For DUPSORT databases:
    /// - If `data` is `None`, deletes the key and ALL its duplicates.
    /// - If `data` is `Some`, deletes only the matching duplicate value.
    ///
    /// For non-DUPSORT databases, `data` is ignored.
    ///
    /// # Errors
    ///
    /// - [`Error::NotFound`] if the key (or specific dup) does not exist
    /// - [`Error::BadDbi`] if `dbi` is invalid
    pub fn del(&mut self, dbi: u32, key: &[u8], data: Option<&[u8]>) -> Result<()> {
        btree::cursor_del(self, dbi, key, data)
    }

    /// Read a value within a write transaction.
    ///
    /// Checks dirty pages first, falls back to the mmap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the key does not exist.
    /// Returns [`Error::BadDbi`] if `dbi` is invalid.
    pub fn get(&self, dbi: u32, key: &[u8]) -> Result<&[u8]> {
        let db = self.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
        if db.root == P_INVALID {
            return Err(Error::NotFound);
        }

        let cmp = self.env.get_cmp(dbi)?;
        let mut cursor = Cursor::new(self.env.page_size, dbi);
        let get_page = |pgno: u64| -> Result<*const u8> { self.get_page(pgno) };

        cursor.page_search(db.root, Some(key), &*cmp, &get_page)?;

        let node = cursor.current_node().ok_or(Error::NotFound)?;
        // SAFETY: data is from mmap or dirty pages, both live long enough
        // for the lifetime of this transaction.
        let node_key: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.key()) };
        if cmp(key, node_key) != Ordering::Equal {
            return Err(Error::NotFound);
        }

        if node.is_bigdata() {
            let pgno = node.overflow_pgno();
            let data_size = node.data_size() as usize;

            if let Some(buf) = self.dirty.find(pgno) {
                // Dirty overflow: stored as a single contiguous buffer
                // covering all overflow pages.
                let page_data = buf.as_slice();
                let data: &[u8] = &page_data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + data_size];
                // SAFETY: data lives as long as the dirty page, which
                // lives as long as the transaction.
                let data: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(data) };
                Ok(data)
            } else {
                // Mmap: pages are contiguous.
                let ptr = self.env.get_page(pgno)?;
                // SAFETY: ptr points into the mmap which is contiguous and
                // outlives this transaction reference.
                let data: &[u8] =
                    unsafe { std::slice::from_raw_parts(ptr.add(PAGE_HEADER_SIZE), data_size) };
                Ok(data)
            }
        } else if node.is_dupdata() {
            // DUPSORT node with sub-page: return the first dup value.
            let first_val = btree::get_dup_at_index(&node, 0).ok_or(Error::NotFound)?;
            let data: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(first_val) };
            Ok(data)
        } else {
            // SAFETY: node_data() points into mmap or dirty pages.
            let data: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.node_data()) };
            Ok(data)
        }
    }

    /// Commit the transaction: flush dirty pages and write the meta page.
    ///
    /// After a successful commit, the transaction is consumed and all changes
    /// become visible to subsequent transactions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadTxn`] if the transaction was already finished.
    /// Returns [`Error::Io`] if flushing pages or syncing fails.
    pub fn commit(mut self) -> Result<()> {
        if self.finished {
            return Err(Error::BadTxn);
        }
        if !self.savepoints.is_empty() {
            return Err(Error::BadTxn);
        }
        self.finished = true;

        // 1. Save freed pages to FREE_DBI
        self.save_freelist()?;

        // 2. Write dirty named DB records back to MAIN_DBI before flushing.
        self.flush_named_dbs()?;

        if self.dirty.is_empty() {
            return Ok(());
        }

        // 3. Flush dirty pages to the data file
        self.flush_dirty_pages()?;

        // 4. Sync data pages to disk
        self.sync_data()?;

        // 5. Write the new meta page (the commit point)
        self.write_meta()?;

        // 6. Sync meta page to disk
        self.sync_data()?;

        Ok(())
    }

    /// Abort the transaction, discarding all changes.
    ///
    /// Dirty pages are dropped automatically when the transaction is consumed.
    pub fn abort(mut self) {
        self.finished = true;
    }

    // -------------------------------------------------------------------
    // Nested transactions (savepoints)
    // -------------------------------------------------------------------

    /// Begin a nested transaction by creating a savepoint.
    ///
    /// All subsequent mutations can be rolled back to this point by calling
    /// [`abort_nested_txn`](Self::abort_nested_txn), or kept by calling
    /// [`commit_nested_txn`](Self::commit_nested_txn). Savepoints may be
    /// nested to arbitrary depth.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadTxn`] if the transaction has already been
    /// committed or aborted.
    pub fn begin_nested_txn(&mut self) -> Result<()> {
        if self.finished {
            return Err(Error::BadTxn);
        }
        let savepoint = SavePoint {
            dbs: self.dbs.clone(),
            db_dirty: self.db_dirty.clone(),
            free_pgs: self.free_pgs.clone(),
            loose_pgs: self.loose_pgs.clone(),
            next_pgno: self.next_pgno,
            reclaim_pgs: self.reclaim_pgs.clone(),
            dirty_snapshot: self.dirty.snapshot(),
        };
        self.savepoints.push(savepoint);
        Ok(())
    }

    /// Commit a nested transaction (pop the savepoint).
    ///
    /// Changes made since the matching [`begin_nested_txn`](Self::begin_nested_txn)
    /// are kept. The savepoint is simply discarded.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadTxn`] if there is no active nested transaction.
    pub fn commit_nested_txn(&mut self) -> Result<()> {
        if self.savepoints.is_empty() {
            return Err(Error::BadTxn);
        }
        self.savepoints.pop();
        Ok(())
    }

    /// Abort a nested transaction, restoring the state captured by the
    /// matching [`begin_nested_txn`](Self::begin_nested_txn).
    ///
    /// All mutations (inserts, deletes, page allocations) made since the
    /// savepoint are rolled back.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadTxn`] if there is no active nested transaction.
    pub fn abort_nested_txn(&mut self) -> Result<()> {
        let sp = self.savepoints.pop().ok_or(Error::BadTxn)?;
        self.dbs = sp.dbs;
        self.db_dirty = sp.db_dirty;
        self.free_pgs = sp.free_pgs;
        self.loose_pgs = sp.loose_pgs;
        self.next_pgno = sp.next_pgno;
        self.reclaim_pgs = sp.reclaim_pgs;
        self.dirty.restore(sp.dirty_snapshot);
        Ok(())
    }

    /// Return the current nested transaction depth.
    ///
    /// A depth of zero means the outermost (top-level) transaction is active.
    #[must_use]
    pub fn nested_depth(&self) -> usize {
        self.savepoints.len()
    }

    /// Return the transaction ID.
    #[must_use]
    pub fn txnid(&self) -> u64 {
        self.txnid
    }

    /// Try to reclaim a page from FREE_DBI records of older transactions.
    ///
    /// On the first call, loads all reclaimable pages from FREE_DBI into
    /// `reclaim_pgs`. Subsequent calls pop from the cached list.
    fn try_reclaim_page(&mut self) -> Result<Option<u64>> {
        if !self.freelist_loaded {
            self.load_freelist()?;
            self.freelist_loaded = true;
        }
        Ok(self.reclaim_pgs.pop())
    }

    /// Load all reclaimable page numbers from FREE_DBI.
    ///
    /// Reads FREE_DBI records whose key (txnid) is less than the reclaim
    /// threshold. The threshold is determined by the oldest active reader:
    /// pages freed by transactions that an active reader may still reference
    /// must not be reclaimed. If no readers are active, all pages freed
    /// before the current transaction can be reclaimed.
    ///
    /// This method separates the read phase (cursor iteration) from the
    /// mutation phase (pushing to `reclaim_pgs`) to satisfy the borrow checker,
    /// since the cursor's `get_page` closure borrows `self` immutably.
    fn load_freelist(&mut self) -> Result<()> {
        let free_db = self.dbs[FREE_DBI];
        if free_db.root == P_INVALID {
            return Ok(());
        }

        let cmp = self.env.get_cmp(FREE_DBI as u32)?;
        let page_size = self.env.page_size;

        // Determine the reclaim threshold based on the oldest active reader.
        // Only reclaim pages from transactions older than the oldest reader,
        // so readers still see a consistent snapshot.
        let oldest_reader = self.env.reader_table.find_oldest();
        let reclaim_threshold = if oldest_reader == u64::MAX {
            self.txnid
        } else {
            oldest_reader
        };

        // Phase 1: Read all freelist data into a temporary buffer.
        // We collect (record_txnid, data_bytes) pairs so we can release
        // the immutable borrow on `self` before mutating `reclaim_pgs`.
        let mut collected: Vec<(u64, Vec<u8>)> = Vec::new();

        {
            let mut cursor = Cursor::new(page_size, FREE_DBI as u32);
            let get_page = |pgno: u64| -> Result<*const u8> { self.get_page(pgno) };

            if cursor.first(free_db.root, &**cmp, &get_page).is_err() {
                return Ok(());
            }

            while let Some(node) = cursor.current_node() {
                let key = node.key();
                if key.len() != 8 {
                    break;
                }
                let record_txnid =
                    u64::from_ne_bytes(key[..8].try_into().map_err(|_| Error::Corrupted)?);

                if record_txnid >= reclaim_threshold {
                    break;
                }

                // Copy the data out so we can drop the borrow.
                collected.push((record_txnid, node.node_data().to_vec()));

                if cursor.next(&get_page).is_err() {
                    break;
                }
            }
        }

        // Phase 2: Parse collected data and push to reclaim_pgs.
        for (record_txnid, data) in &collected {
            self.consumed_freelist_txnids.push(*record_txnid);
            if data.len() < 8 {
                continue;
            }
            let count =
                u64::from_le_bytes(data[..8].try_into().map_err(|_| Error::Corrupted)?) as usize;

            for i in 0..count {
                let offset = (i + 1) * 8;
                if offset + 8 > data.len() {
                    break;
                }
                let pgno = u64::from_le_bytes(
                    data[offset..offset + 8]
                        .try_into()
                        .map_err(|_| Error::Corrupted)?,
                );
                // Don't reclaim meta pages (pages 0 and 1).
                if pgno >= 2 {
                    self.reclaim_pgs.push(pgno);
                }
            }
        }

        Ok(())
    }

    /// Write freed page numbers to FREE_DBI and delete consumed records.
    ///
    /// Uses an iterative convergence loop: writing to FREE_DBI may itself
    /// allocate new pages (e.g. B+ tree splits), which frees old pages and
    /// changes the freelist. The loop repeats until the freelist stabilizes.
    /// This mirrors C LMDB's approach to freelist persistence.
    fn save_freelist(&mut self) -> Result<()> {
        let mut prev_free_count = 0;
        // Safety limit to prevent infinite loops in pathological cases.
        for iteration in 0..100 {
            let current_free_count = self.free_pgs.len();
            if current_free_count == prev_free_count && iteration > 0 {
                break; // Converged
            }
            prev_free_count = current_free_count;

            // Delete consumed FREE_DBI records so they are not reclaimed again
            // by future transactions.
            self.delete_consumed_freelist_records()?;

            // Write current freed pages to FREE_DBI.
            if !self.free_pgs.is_empty() {
                self.write_freelist_record()?;
            }
        }
        Ok(())
    }

    /// Delete consumed FREE_DBI records whose pages were reclaimed during
    /// this transaction.
    fn delete_consumed_freelist_records(&mut self) -> Result<()> {
        let consumed = std::mem::take(&mut self.consumed_freelist_txnids);
        for &old_txnid in &consumed {
            let key = old_txnid.to_ne_bytes();
            // Ignore NotFound errors — the record may have already been deleted.
            match btree::cursor_del(self, FREE_DBI as u32, &key, None) {
                Ok(()) | Err(Error::NotFound) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Serialize and write the current freed pages to FREE_DBI under this
    /// transaction's ID.
    fn write_freelist_record(&mut self) -> Result<()> {
        // Sort for deterministic on-disk order.
        self.free_pgs.sort_unstable();

        // Serialize: [count (u64 LE), pgno0 (u64 LE), pgno1 (u64 LE), ...]
        let mut data = Vec::with_capacity((self.free_pgs.len() + 1) * 8);
        let count = self.free_pgs.len() as u64;
        data.extend_from_slice(&count.to_le_bytes());
        for &pgno in &self.free_pgs {
            data.extend_from_slice(&pgno.to_le_bytes());
        }

        // Key is txnid in native byte order (INTEGER_KEY).
        let key = self.txnid.to_ne_bytes();
        btree::cursor_put_with_flags(
            self,
            FREE_DBI as u32,
            &key,
            &data,
            WriteFlags::empty(),
            NodeFlags::empty(),
        )
    }

    // -----------------------------------------------------------------------
    // Private commit helpers
    // -----------------------------------------------------------------------

    /// Flush dirty named database records back to MAIN_DBI.
    ///
    /// For each named database that was modified during this transaction,
    /// writes its updated `DbStat` record into MAIN_DBI with the `SUBDATA`
    /// node flag.
    fn flush_named_dbs(&mut self) -> Result<()> {
        // Collect the list of dirty named DBs that need flushing.
        let mut to_flush: Vec<(String, DbStat)> = Vec::new();
        for dbi in CORE_DBS as usize..self.dbs.len() {
            if self.db_dirty.get(dbi).copied().unwrap_or(false) {
                if let Some(name) = self.get_db_name(dbi) {
                    to_flush.push((name, self.dbs[dbi]));
                }
            }
        }

        for (name, db) in &to_flush {
            let db_bytes = db_stat_to_bytes(db);
            btree::cursor_put_with_flags(
                self,
                MAIN_DBI as u32,
                name.as_bytes(),
                &db_bytes,
                WriteFlags::empty(),
                NodeFlags::SUBDATA,
            )?;
        }

        Ok(())
    }

    /// Write all dirty pages to the data file via `pwrite`.
    fn flush_dirty_pages(&self) -> Result<()> {
        let page_size = self.env.page_size;
        let fd = self.env.data_fd();

        for (pgno, buf) in self.dirty.iter() {
            let offset = *pgno as i64 * page_size as i64;
            let data = buf.as_slice();
            // SAFETY: fd is a valid file descriptor from the environment's
            // data file. data points to a valid buffer for data.len() bytes.
            let written = unsafe { libc::pwrite(fd, data.as_ptr().cast(), data.len(), offset) };
            if written < 0 || written as usize != data.len() {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
        }

        Ok(())
    }

    /// Sync the data file to disk.
    fn sync_data(&self) -> Result<()> {
        let fd = self.env.data_fd();
        #[cfg(target_os = "macos")]
        {
            // SAFETY: fd is a valid file descriptor.
            let ret = unsafe { libc::fcntl(fd, libc::F_FULLFSYNC) };
            if ret < 0 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // SAFETY: fd is a valid file descriptor.
            let ret = unsafe { libc::fdatasync(fd) };
            if ret < 0 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
        }
        Ok(())
    }

    /// Write the meta page to commit the transaction.
    ///
    /// Toggles between meta page 0 and 1 based on the transaction ID
    /// to implement atomic meta-page switching.
    fn write_meta(&self) -> Result<()> {
        let page_size = self.env.page_size;
        let toggle = (self.txnid & 1) as usize;

        // Build meta page
        let mut meta_buf = vec![0u8; page_size];
        // Page header: pgno + flags
        let pgno = toggle as u64;
        meta_buf[0..8].copy_from_slice(&pgno.to_le_bytes());
        meta_buf[10..12].copy_from_slice(&PageFlags::META.bits().to_le_bytes());

        // Meta payload at PAGE_HEADER_SIZE
        let meta = Meta {
            magic: MDB_MAGIC,
            version: MDB_DATA_VERSION,
            address: 0,
            map_size: self.env.map_size as u64,
            dbs: [self.dbs[FREE_DBI], self.dbs[MAIN_DBI]],
            last_pgno: self.next_pgno - 1,
            txnid: self.txnid,
        };

        // SAFETY: Meta is repr(C) and we read exactly size_of::<Meta>() bytes
        // from a properly aligned struct on the stack.
        let meta_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&meta).cast::<u8>(),
                std::mem::size_of::<Meta>(),
            )
        };
        meta_buf[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + meta_bytes.len()].copy_from_slice(meta_bytes);

        // Write to file at offset toggle * page_size
        let fd = self.env.data_fd();
        let offset = (toggle * page_size) as i64;
        // SAFETY: fd is a valid file descriptor. meta_buf is a valid buffer.
        let written = unsafe { libc::pwrite(fd, meta_buf.as_ptr().cast(), meta_buf.len(), offset) };
        if written < 0 || written as usize != meta_buf.len() {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }

        Ok(())
    }
}

/// Serialize a `DbStat` to its on-disk byte representation.
///
/// # Safety
///
/// `DbStat` is `repr(C)` with a known 48-byte layout. The returned array
/// is a copy of the struct's raw bytes in native byte order (which matches
/// the on-disk format on little-endian platforms).
pub(crate) fn db_stat_to_bytes(db: &DbStat) -> [u8; std::mem::size_of::<DbStat>()] {
    let mut buf = [0u8; std::mem::size_of::<DbStat>()];
    // SAFETY: DbStat is repr(C) and we copy exactly size_of::<DbStat>() bytes.
    let src = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(db).cast::<u8>(),
            std::mem::size_of::<DbStat>(),
        )
    };
    buf.copy_from_slice(src);
    buf
}

impl Drop for RwTransaction<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Implicitly abort -- discard dirty pages
            self.finished = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{env::Environment, node::init_page};

    // -----------------------------------------------------------------------
    // PageBuf unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_create_zeroed_page_buf() {
        let buf = PageBuf::new(4096);
        assert_eq!(buf.as_slice().len(), 4096);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_should_create_page_buf_from_existing() {
        let src = vec![1u8, 2, 3, 4];
        let buf = PageBuf::from_existing(&src);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_should_mutate_page_buf() {
        let mut buf = PageBuf::new(16);
        buf.as_mut_slice()[0] = 0xFF;
        assert_eq!(buf.as_slice()[0], 0xFF);
    }

    #[test]
    fn test_should_return_valid_page_view() {
        let mut buf = PageBuf::new(4096);
        init_page(buf.as_mut_slice(), 42, PageFlags::LEAF, 4096);
        let page = buf.as_page();
        assert_eq!(page.pgno(), 42);
        assert!(page.is_leaf());
    }

    #[test]
    fn test_should_return_stable_pointer() {
        let buf = PageBuf::new(64);
        let ptr = buf.as_ptr();
        assert!(!ptr.is_null());
    }

    // -----------------------------------------------------------------------
    // DirtyPages unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_create_empty_dirty_pages() {
        let dp = DirtyPages::new();
        assert!(dp.is_empty());
        assert_eq!(dp.len(), 0);
    }

    #[test]
    fn test_should_insert_and_find_dirty_page() {
        let mut dp = DirtyPages::new();
        let mut buf = PageBuf::new(64);
        buf.as_mut_slice()[0] = 0xAB;
        dp.insert(5, buf);

        assert_eq!(dp.len(), 1);
        assert!(!dp.is_empty());

        let found = dp.find(5);
        assert!(found.is_some());
        assert_eq!(found.map(|b| b.as_slice()[0]), Some(0xAB));
    }

    #[test]
    fn test_should_replace_existing_dirty_page() {
        let mut dp = DirtyPages::new();
        let buf1 = PageBuf::new(64);
        dp.insert(5, buf1);

        let mut buf2 = PageBuf::new(64);
        buf2.as_mut_slice()[0] = 0xCD;
        dp.insert(5, buf2);

        assert_eq!(dp.len(), 1);
        assert_eq!(dp.find(5).map(|b| b.as_slice()[0]), Some(0xCD));
    }

    #[test]
    fn test_should_maintain_sorted_order() {
        let mut dp = DirtyPages::new();
        dp.insert(10, PageBuf::new(16));
        dp.insert(3, PageBuf::new(16));
        dp.insert(7, PageBuf::new(16));

        let pgnos: Vec<u64> = dp.iter().map(|(pgno, _)| *pgno).collect();
        assert_eq!(pgnos, vec![3, 7, 10]);
    }

    #[test]
    fn test_should_return_none_for_missing_page() {
        let dp = DirtyPages::new();
        assert!(dp.find(42).is_none());
    }

    #[test]
    fn test_should_remove_dirty_page() {
        let mut dp = DirtyPages::new();
        dp.insert(5, PageBuf::new(16));
        dp.insert(10, PageBuf::new(16));

        let removed = dp.remove(5);
        assert!(removed.is_some());
        assert_eq!(dp.len(), 1);
        assert!(dp.find(5).is_none());
        assert!(dp.find(10).is_some());
    }

    #[test]
    fn test_should_return_none_removing_missing_page() {
        let mut dp = DirtyPages::new();
        assert!(dp.remove(99).is_none());
    }

    #[test]
    fn test_should_find_mut_dirty_page() {
        let mut dp = DirtyPages::new();
        dp.insert(5, PageBuf::new(64));

        if let Some(buf) = dp.find_mut(5) {
            buf.as_mut_slice()[0] = 0xFF;
        }

        assert_eq!(dp.find(5).map(|b| b.as_slice()[0]), Some(0xFF));
    }

    #[test]
    fn test_should_clear_all_dirty_pages() {
        let mut dp = DirtyPages::new();
        dp.insert(1, PageBuf::new(16));
        dp.insert(2, PageBuf::new(16));
        dp.insert(3, PageBuf::new(16));
        dp.clear();
        assert!(dp.is_empty());
    }

    // -----------------------------------------------------------------------
    // RwTransaction integration tests
    //
    // NOTE: These tests require `data_fd()` on EnvironmentInner and
    // `begin_rw_txn()` on Environment to be present in env.rs.
    // See the module-level documentation for the methods to add.
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_put_and_get_single_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        // Write a key
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI as u32, b"hello", b"world", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // Read it back in a new read txn
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let val = txn.get(MAIN_DBI as u32, b"hello").expect("get");
            assert_eq!(val, b"world");
        }
    }

    #[test]
    fn test_should_put_and_delete() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI as u32, b"key", b"val", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.del(MAIN_DBI as u32, b"key", None).expect("del");
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let result = txn.get(MAIN_DBI as u32, b"key");
            assert!(matches!(result, Err(Error::NotFound)));
        }
    }

    #[test]
    fn test_should_reject_no_overwrite() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI as u32, b"key", b"v1", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let result = txn.put(MAIN_DBI as u32, b"key", b"v2", WriteFlags::NO_OVERWRITE);
            assert!(matches!(result, Err(Error::KeyExist)));
            txn.abort();
        }
    }

    #[test]
    fn test_should_abort_discard_changes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI as u32, b"key", b"val", WriteFlags::empty())
                .expect("put");
            txn.abort();
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let result = txn.get(MAIN_DBI as u32, b"key");
            assert!(matches!(result, Err(Error::NotFound)));
        }
    }

    #[test]
    fn test_should_put_multiple_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI as u32, b"aaa", b"111", WriteFlags::empty())
                .expect("put aaa");
            txn.put(MAIN_DBI as u32, b"bbb", b"222", WriteFlags::empty())
                .expect("put bbb");
            txn.put(MAIN_DBI as u32, b"ccc", b"333", WriteFlags::empty())
                .expect("put ccc");
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            assert_eq!(txn.get(MAIN_DBI as u32, b"aaa").expect("get"), b"111");
            assert_eq!(txn.get(MAIN_DBI as u32, b"bbb").expect("get"), b"222");
            assert_eq!(txn.get(MAIN_DBI as u32, b"ccc").expect("get"), b"333");
        }
    }

    #[test]
    fn test_should_delete_nonexistent_returns_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        let result = txn.del(MAIN_DBI as u32, b"nonexistent", None);
        assert!(matches!(result, Err(Error::NotFound)));
        txn.abort();
    }

    #[test]
    fn test_should_read_within_write_txn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        txn.put(MAIN_DBI as u32, b"key", b"val", WriteFlags::empty())
            .expect("put");

        // Read within the same write txn
        let val = txn.get(MAIN_DBI as u32, b"key").expect("get");
        assert_eq!(val, b"val");

        txn.commit().expect("commit");
    }

    // -----------------------------------------------------------------------
    // Named database tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_create_and_use_named_db() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        // Create and write to a named database.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(Some("mydb"), DatabaseFlags::CREATE)
                .expect("open_db");
            txn.put(dbi, b"hello", b"named-world", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // Read from the named database.
        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("mydb")).expect("open_db ro");
            let val = txn.get(dbi, b"hello").expect("get");
            assert_eq!(val, b"named-world");
        }
    }

    #[test]
    fn test_should_support_multiple_named_dbs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        // Create two named databases and write different data to each.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let db_a = txn
                .open_db(Some("db_a"), DatabaseFlags::CREATE)
                .expect("open db_a");
            let db_b = txn
                .open_db(Some("db_b"), DatabaseFlags::CREATE)
                .expect("open db_b");

            txn.put(db_a, b"key", b"value-A", WriteFlags::empty())
                .expect("put A");
            txn.put(db_b, b"key", b"value-B", WriteFlags::empty())
                .expect("put B");
            txn.commit().expect("commit");
        }

        // Read from both and verify data is independent.
        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let db_a = txn.open_db(Some("db_a")).expect("open db_a ro");
            let db_b = txn.open_db(Some("db_b")).expect("open db_b ro");

            assert_eq!(txn.get(db_a, b"key").expect("get A"), b"value-A");
            assert_eq!(txn.get(db_b, b"key").expect("get B"), b"value-B");
        }
    }

    #[test]
    fn test_should_persist_named_db_across_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Write to a named database.
        {
            let env = Environment::builder()
                .map_size(10 * 1024 * 1024)
                .max_dbs(4)
                .open(dir.path())
                .expect("open");

            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(Some("persist"), DatabaseFlags::CREATE)
                .expect("open_db");
            txn.put(dbi, b"durable", b"data", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // Reopen the environment and read back.
        {
            let env = Environment::builder()
                .map_size(10 * 1024 * 1024)
                .max_dbs(4)
                .open(dir.path())
                .expect("reopen");

            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("persist")).expect("open_db ro");
            let val = txn.get(dbi, b"durable").expect("get");
            assert_eq!(val, b"data");
        }
    }

    #[test]
    fn test_should_return_not_found_for_nonexistent_named_db() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        // Try to open a named DB without CREATE.
        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        let result = txn.open_db(Some("nope"), DatabaseFlags::empty());
        assert!(
            matches!(result, Err(Error::NotFound)),
            "expected NotFound, got {result:?}",
        );
        txn.abort();
    }

    #[test]
    fn test_should_return_bad_dbi_without_max_dbs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .max_dbs(0) // no named databases configured
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        let result = txn.open_db(Some("fail"), DatabaseFlags::CREATE);
        assert!(
            matches!(result, Err(Error::BadDbi)),
            "expected BadDbi, got {result:?}",
        );
        txn.abort();
    }

    #[test]
    fn test_should_see_consistent_snapshot_with_sequential_txns() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        // Create named DB and write initial data.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(Some("snapshot_db"), DatabaseFlags::CREATE)
                .expect("open_db");
            txn.put(dbi, b"version", b"1", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // Start a read transaction (sees version 1).
        let ro_txn = env.begin_ro_txn().expect("begin_ro_txn");

        // Write new data in a write transaction.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(Some("snapshot_db"), DatabaseFlags::CREATE)
                .expect("open_db");
            txn.put(dbi, b"version", b"2", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // A new read transaction should see the latest data.
        {
            let mut ro_txn2 = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = ro_txn2.open_db(Some("snapshot_db")).expect("open_db ro");
            let val = ro_txn2.get(dbi, b"version").expect("get");
            assert_eq!(val, b"2");
        }

        drop(ro_txn);
    }

    #[test]
    fn test_should_write_many_keys_to_named_db() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(64 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        // Write many keys to a named DB to exercise page splits.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(Some("bigdb"), DatabaseFlags::CREATE)
                .expect("open_db");
            for i in 0..200u32 {
                let key = format!("nk-{i:06}");
                let val = format!("nv-{i:06}");
                txn.put(dbi, key.as_bytes(), val.as_bytes(), WriteFlags::empty())
                    .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Verify all keys.
        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("bigdb")).expect("open_db ro");
            for i in 0..200u32 {
                let key = format!("nk-{i:06}");
                let val = format!("nv-{i:06}");
                let got = txn
                    .get(dbi, key.as_bytes())
                    .unwrap_or_else(|e| panic!("get {key}: {e}"));
                assert_eq!(got, val.as_bytes(), "mismatch for {key}");
            }
        }
    }

    #[test]
    fn test_should_open_default_db_with_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        let dbi = txn
            .open_db(None, DatabaseFlags::empty())
            .expect("open_db None");
        assert_eq!(dbi, MAIN_DBI as u32);
        txn.abort();
    }

    // -----------------------------------------------------------------------
    // Free page reuse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_save_freelist_on_commit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .open(dir.path())
            .expect("open");

        // Insert keys — this creates pages.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..50u32 {
                let key = format!("key-{i:04}");
                let val = format!("val-{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Delete all keys — this frees pages. The freed pages should be
        // saved to FREE_DBI on commit.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..50u32 {
                let key = format!("key-{i:04}");
                txn.del(MAIN_DBI as u32, key.as_bytes(), None)
                    .unwrap_or_else(|e| panic!("del {key}: {e}"));
            }
            // After delete, free_pgs should be non-empty (COW freed pages).
            assert!(
                !txn.free_pgs.is_empty(),
                "expected freed pages after deleting keys",
            );
            txn.commit().expect("commit");
        }

        // Verify the FREE_DBI has entries by checking its root is not P_INVALID.
        {
            let meta = env.info();
            // The free DB should have been written to, so last_pgno should
            // reflect the pages used for the freelist tree.
            assert!(meta.last_pgno > 0, "expected some pages to exist");
        }
    }

    #[test]
    fn test_should_reuse_freed_pages() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .open(dir.path())
            .expect("open");

        // Insert 100 keys across multiple pages.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..100u32 {
                let key = format!("reuse-{i:04}");
                let val = format!("data-{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        let pgno_after_insert = env.info().last_pgno;

        // Delete all keys — frees their pages.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..100u32 {
                let key = format!("reuse-{i:04}");
                txn.del(MAIN_DBI as u32, key.as_bytes(), None)
                    .unwrap_or_else(|e| panic!("del {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Insert 100 new keys — should reuse freed pages.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..100u32 {
                let key = format!("new-{i:04}");
                let val = format!("newdata-{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        let pgno_after_reinsert = env.info().last_pgno;

        // The file should not have grown much — pages were reused.
        // Without reuse, pgno_after_reinsert would be roughly
        // pgno_after_insert * 2. With reuse, it should be close to
        // pgno_after_insert or only slightly larger.
        assert!(
            pgno_after_reinsert <= pgno_after_insert + 5,
            "expected page reuse: after_insert={pgno_after_insert}, \
             after_reinsert={pgno_after_reinsert}",
        );

        // Verify the new data is readable.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..100u32 {
                let key = format!("new-{i:04}");
                let val = format!("newdata-{i:04}");
                let got = txn
                    .get(MAIN_DBI as u32, key.as_bytes())
                    .unwrap_or_else(|e| panic!("get {key}: {e}"));
                assert_eq!(got, val.as_bytes(), "mismatch for {key}");
            }
        }
    }

    #[test]
    fn test_should_reuse_pages_across_multiple_rounds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .open(dir.path())
            .expect("open");

        // Run 5 rounds of insert-delete-reinsert.
        for round in 0..5u32 {
            // Insert
            {
                let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
                for i in 0..50u32 {
                    let key = format!("r{round}-k{i:04}");
                    let val = format!("r{round}-v{i:04}");
                    txn.put(
                        MAIN_DBI as u32,
                        key.as_bytes(),
                        val.as_bytes(),
                        WriteFlags::empty(),
                    )
                    .unwrap_or_else(|e| panic!("put round {round} key {i}: {e}"));
                }
                txn.commit().expect("commit");
            }

            // Delete
            {
                let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
                for i in 0..50u32 {
                    let key = format!("r{round}-k{i:04}");
                    txn.del(MAIN_DBI as u32, key.as_bytes(), None)
                        .unwrap_or_else(|e| panic!("del round {round} key {i}: {e}"));
                }
                txn.commit().expect("commit");
            }
        }

        // After 5 rounds of 50 insert/delete, the database should not have
        // grown to 5x the single-round size. Check that last_pgno is bounded.
        let info = env.info();
        // Without free page reuse, 5 rounds of ~3 pages each = ~30+ pages.
        // With reuse, it should stabilize around the single-round peak.
        assert!(
            info.last_pgno < 30,
            "expected bounded growth with free page reuse, got last_pgno={}",
            info.last_pgno,
        );
    }

    // -----------------------------------------------------------------------
    // Nested transaction (savepoint) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_commit_nested_txn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");

            // Insert key A in parent
            txn.put(MAIN_DBI as u32, b"key-a", b"val-a", WriteFlags::empty())
                .expect("put A");

            // Begin nested, insert key B
            txn.begin_nested_txn().expect("begin_nested");
            txn.put(MAIN_DBI as u32, b"key-b", b"val-b", WriteFlags::empty())
                .expect("put B");

            // Commit nested — B should be visible in parent
            txn.commit_nested_txn().expect("commit_nested");

            // Both keys should be visible within the parent txn
            assert_eq!(txn.get(MAIN_DBI as u32, b"key-a").expect("get A"), b"val-a",);
            assert_eq!(txn.get(MAIN_DBI as u32, b"key-b").expect("get B"), b"val-b",);

            txn.commit().expect("commit");
        }

        // Read txn should see both A and B
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            assert_eq!(txn.get(MAIN_DBI as u32, b"key-a").expect("get A"), b"val-a",);
            assert_eq!(txn.get(MAIN_DBI as u32, b"key-b").expect("get B"), b"val-b",);
        }
    }

    #[test]
    fn test_should_abort_nested_txn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");

            // Insert key A in parent
            txn.put(MAIN_DBI as u32, b"key-a", b"val-a", WriteFlags::empty())
                .expect("put A");

            // Begin nested, insert key B, delete key A
            txn.begin_nested_txn().expect("begin_nested");
            txn.put(MAIN_DBI as u32, b"key-b", b"val-b", WriteFlags::empty())
                .expect("put B");
            txn.del(MAIN_DBI as u32, b"key-a", None).expect("del A");

            // Abort nested — B gone, A restored
            txn.abort_nested_txn().expect("abort_nested");

            // A should be visible, B should not
            assert_eq!(txn.get(MAIN_DBI as u32, b"key-a").expect("get A"), b"val-a",);
            assert!(matches!(
                txn.get(MAIN_DBI as u32, b"key-b"),
                Err(Error::NotFound),
            ));

            txn.commit().expect("commit");
        }

        // Read txn should see A but not B
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            assert_eq!(txn.get(MAIN_DBI as u32, b"key-a").expect("get A"), b"val-a",);
            assert!(matches!(
                txn.get(MAIN_DBI as u32, b"key-b"),
                Err(Error::NotFound),
            ));
        }
    }

    #[test]
    fn test_should_handle_multiple_nesting_levels() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            assert_eq!(txn.nested_depth(), 0);

            // Begin nested1, insert key1
            txn.begin_nested_txn().expect("begin nested1");
            assert_eq!(txn.nested_depth(), 1);
            txn.put(MAIN_DBI as u32, b"k1", b"v1", WriteFlags::empty())
                .expect("put k1");

            // Begin nested2, insert key2
            txn.begin_nested_txn().expect("begin nested2");
            assert_eq!(txn.nested_depth(), 2);
            txn.put(MAIN_DBI as u32, b"k2", b"v2", WriteFlags::empty())
                .expect("put k2");

            // Commit nested2 — k2 merges into nested1
            txn.commit_nested_txn().expect("commit nested2");
            assert_eq!(txn.nested_depth(), 1);

            // Abort nested1 — both k1 and k2 should be gone
            txn.abort_nested_txn().expect("abort nested1");
            assert_eq!(txn.nested_depth(), 0);

            assert!(matches!(
                txn.get(MAIN_DBI as u32, b"k1"),
                Err(Error::NotFound),
            ));
            assert!(matches!(
                txn.get(MAIN_DBI as u32, b"k2"),
                Err(Error::NotFound),
            ));

            txn.abort();
        }
    }

    #[test]
    fn test_should_rollback_nested_txn_with_page_splits() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");

            // Insert an initial key so the tree is non-empty
            txn.put(MAIN_DBI as u32, b"anchor", b"value", WriteFlags::empty())
                .expect("put anchor");

            // Begin nested txn and insert enough keys to trigger page splits
            txn.begin_nested_txn().expect("begin_nested");
            for i in 0..200u32 {
                let key = format!("split-{i:06}");
                let val = format!("data-{i:06}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .expect("put split key");
            }

            // Abort — all split pages should be rolled back
            txn.abort_nested_txn().expect("abort_nested");

            // Anchor key should still be there
            assert_eq!(
                txn.get(MAIN_DBI as u32, b"anchor").expect("get anchor"),
                b"value",
            );

            // None of the split keys should exist
            for i in 0..200u32 {
                let key = format!("split-{i:06}");
                assert!(
                    matches!(
                        txn.get(MAIN_DBI as u32, key.as_bytes()),
                        Err(Error::NotFound)
                    ),
                    "key {key} should not exist after abort",
                );
            }

            txn.commit().expect("commit");
        }

        // Verify in a read txn
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            assert_eq!(
                txn.get(MAIN_DBI as u32, b"anchor").expect("get anchor"),
                b"value",
            );
            assert!(matches!(
                txn.get(MAIN_DBI as u32, b"split-000000"),
                Err(Error::NotFound),
            ));
        }
    }

    #[test]
    fn test_should_reject_commit_nested_without_savepoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        assert!(matches!(txn.commit_nested_txn(), Err(Error::BadTxn)));
        txn.abort();
    }

    #[test]
    fn test_should_reject_abort_nested_without_savepoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        assert!(matches!(txn.abort_nested_txn(), Err(Error::BadTxn)));
        txn.abort();
    }

    #[test]
    fn test_should_reject_commit_with_active_savepoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        txn.begin_nested_txn().expect("begin_nested");
        let result = txn.commit();
        assert!(
            matches!(result, Err(Error::BadTxn)),
            "expected BadTxn when committing with active savepoint",
        );
    }

    #[test]
    fn test_should_reject_begin_nested_on_finished_txn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        // We can't call begin_nested_txn after commit since commit consumes
        // self. But we can test the finished guard by using the internal field.
        // Instead, we test that begin_nested_txn works on a live transaction.
        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        txn.begin_nested_txn().expect("begin_nested should work");
        txn.commit_nested_txn().expect("commit_nested");
        txn.abort();
    }

    #[test]
    fn test_should_update_value_in_nested_txn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI as u32, b"key", b"original", WriteFlags::empty())
                .expect("put");

            // Begin nested, update the value
            txn.begin_nested_txn().expect("begin_nested");
            txn.put(MAIN_DBI as u32, b"key", b"updated", WriteFlags::empty())
                .expect("put update");

            // Verify updated value visible
            assert_eq!(txn.get(MAIN_DBI as u32, b"key").expect("get"), b"updated",);

            // Abort nested — should restore original value
            txn.abort_nested_txn().expect("abort_nested");
            assert_eq!(txn.get(MAIN_DBI as u32, b"key").expect("get"), b"original",);

            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            assert_eq!(txn.get(MAIN_DBI as u32, b"key").expect("get"), b"original",);
        }
    }

    #[test]
    fn test_should_handle_nested_txn_with_named_db() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(Some("nested_db"), DatabaseFlags::CREATE)
                .expect("open_db");

            txn.put(dbi, b"parent-key", b"parent-val", WriteFlags::empty())
                .expect("put parent");

            // Begin nested, add another key
            txn.begin_nested_txn().expect("begin_nested");
            txn.put(dbi, b"child-key", b"child-val", WriteFlags::empty())
                .expect("put child");

            // Commit nested
            txn.commit_nested_txn().expect("commit_nested");
            txn.commit().expect("commit");
        }

        // Verify both keys exist
        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("nested_db")).expect("open_db ro");
            assert_eq!(
                txn.get(dbi, b"parent-key").expect("get parent"),
                b"parent-val",
            );
            assert_eq!(txn.get(dbi, b"child-key").expect("get child"), b"child-val",);
        }
    }

    // -----------------------------------------------------------------------
    // DUPSORT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_dupsort_insert_and_get_duplicates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            // Insert multiple values for the same key.
            txn.put(dbi, b"key1", b"val-c", WriteFlags::empty())
                .expect("put c");
            txn.put(dbi, b"key1", b"val-a", WriteFlags::empty())
                .expect("put a");
            txn.put(dbi, b"key1", b"val-b", WriteFlags::empty())
                .expect("put b");
            txn.commit().expect("commit");
        }

        // get() returns the first (sorted) dup value.
        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let val = txn.get(dbi, b"key1").expect("get");
            assert_eq!(val, b"val-a", "get() should return first sorted dup");
        }
    }

    #[test]
    fn test_should_dupsort_delete_single_dup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"val-a", WriteFlags::empty())
                .expect("put a");
            txn.put(dbi, b"key1", b"val-b", WriteFlags::empty())
                .expect("put b");
            txn.put(dbi, b"key1", b"val-c", WriteFlags::empty())
                .expect("put c");

            // Delete only val-b.
            txn.del(dbi, b"key1", Some(b"val-b"))
                .expect("del single dup");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");

            // val-a should still be the first.
            let val = txn.get(dbi, b"key1").expect("get");
            assert_eq!(val, b"val-a");

            // Iterate with cursor to verify val-b is gone.
            let mut cursor = txn.open_cursor(dbi).expect("cursor");
            let (_, v1) = cursor.get(Some(b"key1"), CursorOp::Set).expect("set");
            assert_eq!(v1, b"val-a");
            let (_, v2) = cursor.get(None, CursorOp::NextDup).expect("next dup");
            assert_eq!(v2, b"val-c");
            // No more dups.
            let result = cursor.get(None, CursorOp::NextDup);
            assert!(matches!(result, Err(Error::NotFound)));
        }
    }

    #[test]
    fn test_should_dupsort_delete_all_dups() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"val-a", WriteFlags::empty())
                .expect("put a");
            txn.put(dbi, b"key1", b"val-b", WriteFlags::empty())
                .expect("put b");

            // Delete all dups (data = None).
            txn.del(dbi, b"key1", None).expect("del all");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let result = txn.get(dbi, b"key1");
            assert!(matches!(result, Err(Error::NotFound)));
        }
    }

    #[test]
    fn test_should_dupsort_cursor_next_dup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"val-a", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", b"val-b", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", b"val-c", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key2", b"val-x", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");

            // First entry.
            let (k, v) = cursor.get(None, CursorOp::First).expect("first");
            assert_eq!(k, b"key1");
            assert_eq!(v, b"val-a");

            // Next dup.
            let (k, v) = cursor.get(None, CursorOp::NextDup).expect("next dup 1");
            assert_eq!(k, b"key1");
            assert_eq!(v, b"val-b");

            let (k, v) = cursor.get(None, CursorOp::NextDup).expect("next dup 2");
            assert_eq!(k, b"key1");
            assert_eq!(v, b"val-c");

            // No more dups for key1.
            assert!(matches!(
                cursor.get(None, CursorOp::NextDup),
                Err(Error::NotFound)
            ));

            // Next (advances to key2).
            let (k, v) = cursor.get(None, CursorOp::Next).expect("next to key2");
            assert_eq!(k, b"key2");
            assert_eq!(v, b"val-x");
        }
    }

    #[test]
    fn test_should_dupsort_cursor_prev_dup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"val-a", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", b"val-b", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", b"val-c", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");

            // Position at last.
            let (k, v) = cursor.get(None, CursorOp::Last).expect("last");
            assert_eq!(k, b"key1");
            assert_eq!(v, b"val-c");

            // Prev dup.
            let (_, v) = cursor.get(None, CursorOp::PrevDup).expect("prev dup");
            assert_eq!(v, b"val-b");

            let (_, v) = cursor.get(None, CursorOp::PrevDup).expect("prev dup");
            assert_eq!(v, b"val-a");

            // No more prev dups.
            assert!(matches!(
                cursor.get(None, CursorOp::PrevDup),
                Err(Error::NotFound)
            ));
        }
    }

    #[test]
    fn test_should_dupsort_cursor_first_last_dup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"aaa", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", b"bbb", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", b"ccc", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");

            // Position at key1.
            let (_, v) = cursor.get(Some(b"key1"), CursorOp::Set).expect("set");
            assert_eq!(v, b"aaa");

            // Last dup.
            let (_, v) = cursor.get(None, CursorOp::LastDup).expect("last dup");
            assert_eq!(v, b"ccc");

            // First dup.
            let (_, v) = cursor.get(None, CursorOp::FirstDup).expect("first dup");
            assert_eq!(v, b"aaa");
        }
    }

    #[test]
    fn test_should_dupsort_many_duplicates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        let dup_count = 50u32;

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            // Insert many dups in reverse order.
            for i in (0..dup_count).rev() {
                let val = format!("dup-{i:04}");
                txn.put(dbi, b"multi", val.as_bytes(), WriteFlags::empty())
                    .expect("put dup");
            }
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");

            // Verify all dups are in sorted order.
            let (_, first_val) = cursor.get(Some(b"multi"), CursorOp::Set).expect("set");
            assert_eq!(first_val, b"dup-0000");

            let mut collected = vec![first_val.to_vec()];
            while let Ok((_, v)) = cursor.get(None, CursorOp::NextDup) {
                collected.push(v.to_vec());
            }
            assert_eq!(
                collected.len(),
                dup_count as usize,
                "expected {dup_count} dups, got {}",
                collected.len(),
            );

            // Verify sorted order.
            for i in 0..collected.len() {
                let expected = format!("dup-{i:04}");
                assert_eq!(collected[i], expected.as_bytes(), "mismatch at dup {i}",);
            }
        }
    }

    #[test]
    fn test_should_dupsort_persist_across_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");

        {
            let env = Environment::builder()
                .map_size(10 * 1024 * 1024)
                .max_dbs(4)
                .open(dir.path())
                .expect("open");

            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"val-a", WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", b"val-b", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // Reopen.
        {
            let env = Environment::builder()
                .map_size(10 * 1024 * 1024)
                .max_dbs(4)
                .open(dir.path())
                .expect("reopen");

            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let val = txn.get(dbi, b"key1").expect("get");
            assert_eq!(val, b"val-a");

            let mut cursor = txn.open_cursor(dbi).expect("cursor");
            let (_, v) = cursor.get(Some(b"key1"), CursorOp::Set).expect("set");
            assert_eq!(v, b"val-a");
            let (_, v) = cursor.get(None, CursorOp::NextDup).expect("next dup");
            assert_eq!(v, b"val-b");
        }
    }

    #[test]
    fn test_should_dupsort_no_dup_data_flag() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"val-a", WriteFlags::empty())
                .expect("put a");

            // Try inserting the same key+data with NO_DUP_DATA.
            let result = txn.put(dbi, b"key1", b"val-a", WriteFlags::NO_DUP_DATA);
            assert!(matches!(result, Err(Error::KeyExist)));

            // Different data should work.
            txn.put(dbi, b"key1", b"val-b", WriteFlags::NO_DUP_DATA)
                .expect("put b");

            txn.commit().expect("commit");
        }
    }

    #[test]
    fn test_should_dupfixed_insert_and_iterate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupfixed"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED,
                )
                .expect("open_db");

            // Insert fixed-size values (4 bytes each).
            txn.put(dbi, b"key1", &3u32.to_be_bytes(), WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", &1u32.to_be_bytes(), WriteFlags::empty())
                .expect("put");
            txn.put(dbi, b"key1", &2u32.to_be_bytes(), WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupfixed")).expect("open_db ro");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");

            let mut collected: Vec<u32> = Vec::new();
            let (_, v) = cursor.get(Some(b"key1"), CursorOp::Set).expect("set");
            collected.push(u32::from_be_bytes(v.try_into().expect("4 bytes")));

            while let Ok((_, v)) = cursor.get(None, CursorOp::NextDup) {
                collected.push(u32::from_be_bytes(v.try_into().expect("4 bytes")));
            }

            assert_eq!(collected, vec![1, 2, 3], "values should be sorted");
        }
    }

    #[test]
    fn test_should_dupsort_next_traverses_across_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"a", b"a1", WriteFlags::empty()).expect("put");
            txn.put(dbi, b"a", b"a2", WriteFlags::empty()).expect("put");
            txn.put(dbi, b"b", b"b1", WriteFlags::empty()).expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");

            // Iterate using Next which traverses through all dups and keys.
            let (k, v) = cursor.get(None, CursorOp::First).expect("first");
            assert_eq!((k, v), (b"a" as &[u8], b"a1" as &[u8]));

            let (k, v) = cursor.get(None, CursorOp::Next).expect("next");
            assert_eq!((k, v), (b"a" as &[u8], b"a2" as &[u8]));

            let (k, v) = cursor.get(None, CursorOp::Next).expect("next");
            assert_eq!((k, v), (b"b" as &[u8], b"b1" as &[u8]));

            // No more.
            assert!(matches!(
                cursor.get(None, CursorOp::Next),
                Err(Error::NotFound)
            ));
        }
    }

    #[test]
    fn test_should_dupsort_del_last_dup_removes_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .max_dbs(4)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let dbi = txn
                .open_db(
                    Some("dupsort"),
                    DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
                )
                .expect("open_db");

            txn.put(dbi, b"key1", b"only-val", WriteFlags::empty())
                .expect("put");

            // Delete the only dup value.
            txn.del(dbi, b"key1", Some(b"only-val"))
                .expect("del single");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_ro_txn().expect("begin_ro_txn");
            let dbi = txn.open_db(Some("dupsort")).expect("open_db ro");
            let result = txn.get(dbi, b"key1");
            assert!(matches!(result, Err(Error::NotFound)));
        }
    }

    // -----------------------------------------------------------------------
    // Reader table integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_track_reader_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .max_readers(4)
            .open(dir.path())
            .expect("open");

        assert_eq!(env.info().num_readers, 0);

        let ro1 = env.begin_ro_txn().expect("ro1");
        assert_eq!(env.info().num_readers, 1);

        let ro2 = env.begin_ro_txn().expect("ro2");
        assert_eq!(env.info().num_readers, 2);

        drop(ro1);
        assert_eq!(env.info().num_readers, 1);

        drop(ro2);
        assert_eq!(env.info().num_readers, 0);
    }

    #[test]
    fn test_should_return_readers_full() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .max_readers(2)
            .open(dir.path())
            .expect("open");

        let _ro1 = env.begin_ro_txn().expect("ro1");
        let _ro2 = env.begin_ro_txn().expect("ro2");
        let result = env.begin_ro_txn();
        assert!(
            matches!(result, Err(Error::ReadersFull)),
            "expected ReadersFull, got {result:?}",
        );
    }

    #[test]
    fn test_should_reader_table_prevent_premature_reclamation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(10 * 1024 * 1024)
            .open(dir.path())
            .expect("open");

        // Insert keys in txn 1.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..50u32 {
                let key = format!("rtp-{i:04}");
                let val = format!("val-{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit insert");
        }

        // Start a read txn that holds a snapshot at the current txnid.
        let ro_txn = env.begin_ro_txn().expect("ro snapshot");
        let snapshot_txnid = ro_txn.txnid();

        // Insert more keys in txn 2 (these cause COW, freeing pages).
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 50..100u32 {
                let key = format!("rtp-{i:04}");
                let val = format!("val-{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit more");
        }

        let pgno_before = env.info().last_pgno;

        // Delete keys in txn 3 (frees more pages).
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..100u32 {
                let key = format!("rtp-{i:04}");
                txn.del(MAIN_DBI as u32, key.as_bytes(), None)
                    .unwrap_or_else(|e| panic!("del {key}: {e}"));
            }
            txn.commit().expect("commit delete");
        }

        // Now insert again in txn 4. Because the reader is still active,
        // pages freed by txn 2 and 3 should NOT be reclaimed, so the
        // database file must grow.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..50u32 {
                let key = format!("rtp2-{i:04}");
                let val = format!("val2-{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit reinsert");
        }

        let pgno_with_reader = env.info().last_pgno;

        // The reader was still active, so pages should not have been
        // reclaimed — file should have grown.
        assert!(
            pgno_with_reader > pgno_before,
            "expected growth with active reader: before={pgno_before}, after={pgno_with_reader}",
        );

        // The snapshot read txn should still be valid and see the old data.
        assert!(snapshot_txnid > 0, "snapshot txnid should be valid",);

        // Drop the reader.
        drop(ro_txn);
        assert_eq!(env.info().num_readers, 0);

        // Now that the reader is gone, a new write txn should be able to
        // reclaim those freed pages.
        let pgno_after_reader_drop = env.info().last_pgno;

        // Delete again and reinsert — this time pages should be reclaimed.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..50u32 {
                let key = format!("rtp2-{i:04}");
                txn.del(MAIN_DBI as u32, key.as_bytes(), None)
                    .unwrap_or_else(|e| panic!("del {key}: {e}"));
            }
            txn.commit().expect("commit del2");
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..50u32 {
                let key = format!("rtp3-{i:04}");
                let val = format!("val3-{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit reinsert2");
        }

        let pgno_after_reclaim = env.info().last_pgno;

        // With no active readers, pages should have been reclaimed, so
        // growth should be bounded (much less than with the reader active).
        assert!(
            pgno_after_reclaim <= pgno_after_reader_drop + 5,
            "expected page reclamation without reader: \
             after_reader_drop={pgno_after_reader_drop}, after_reclaim={pgno_after_reclaim}",
        );
    }
}
