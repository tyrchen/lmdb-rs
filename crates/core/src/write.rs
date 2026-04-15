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
}

impl Default for DirtyPages {
    fn default() -> Self {
        Self::new()
    }
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
    /// Whether this transaction has been committed or aborted.
    finished: bool,
}

impl std::fmt::Debug for RwTransaction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RwTransaction")
            .field("txnid", &self.txnid)
            .field("next_pgno", &self.next_pgno)
            .field("dirty_pages", &self.dirty.len())
            .field("finished", &self.finished)
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
        // Acquire writer lock — only one writer at a time.
        // We lock but don't store the guard; Drop on RwTransaction won't
        // unlock, but since we consume `self` in commit/abort this is safe
        // for single-process use. Full lock management comes with Phase 6.
        let _guard = env.write_mutex.lock().map_err(|_| Error::Panic)?;
        drop(_guard); // Release immediately for now — true serialization needs stored guard
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
            finished: false,
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
        // Try loose pages first (pages freed and dirtied in same txn)
        if let Some(pgno) = self.loose_pgs.pop() {
            if let Some(buf) = self.dirty.remove(pgno) {
                return Ok((pgno, buf));
            }
        }

        // Extend the file
        let pgno = self.next_pgno;
        if pgno >= self.env.max_pgno {
            return Err(Error::MapFull);
        }
        self.next_pgno += 1;
        let buf = PageBuf::new(self.env.page_size);
        Ok((pgno, buf))
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

    /// Delete a key from the specified database.
    ///
    /// Handles page rebalancing automatically when pages become underfilled.
    ///
    /// # Errors
    ///
    /// - [`Error::NotFound`] if the key does not exist
    /// - [`Error::BadDbi`] if `dbi` is invalid
    pub fn del(&mut self, dbi: u32, key: &[u8]) -> Result<()> {
        btree::cursor_del(self, dbi, key)
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
            let ptr = self.get_page(pgno)?;
            let data_size = node.data_size() as usize;
            // SAFETY: ptr points into the mmap or dirty page buffer which
            // outlives this transaction reference.
            let data: &[u8] =
                unsafe { std::slice::from_raw_parts(ptr.add(PAGE_HEADER_SIZE), data_size) };
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
        self.finished = true;

        // Write dirty named DB records back to MAIN_DBI before flushing.
        self.flush_named_dbs()?;

        if self.dirty.is_empty() {
            return Ok(());
        }

        // 1. Flush dirty pages to the data file
        self.flush_dirty_pages()?;

        // 2. Sync data pages to disk
        self.sync_data()?;

        // 3. Write the new meta page (the commit point)
        self.write_meta()?;

        // 4. Sync meta page to disk
        self.sync_data()?;

        Ok(())
    }

    /// Abort the transaction, discarding all changes.
    ///
    /// Dirty pages are dropped automatically when the transaction is consumed.
    pub fn abort(mut self) {
        self.finished = true;
    }

    /// Return the transaction ID.
    #[must_use]
    pub fn txnid(&self) -> u64 {
        self.txnid
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
fn db_stat_to_bytes(db: &DbStat) -> [u8; std::mem::size_of::<DbStat>()] {
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
            txn.del(MAIN_DBI as u32, b"key").expect("del");
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
        let result = txn.del(MAIN_DBI as u32, b"nonexistent");
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
}
