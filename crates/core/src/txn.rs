//! Transaction types for MVCC read/write access.
//!
//! LMDB uses a single-writer, multi-reader concurrency model.
//! [`RoTransaction`] provides zero-copy read access to a consistent snapshot.
//! Write transactions will be added in Phase 2.

use std::{cmp::Ordering, sync::Arc};

use crate::{
    cmp::CmpFn,
    cursor::Cursor,
    env::EnvironmentInner,
    error::{Error, Result},
    page::Page,
    types::*,
};

// ---------------------------------------------------------------------------
// Read-only transaction
// ---------------------------------------------------------------------------

/// A read-only transaction providing a consistent snapshot of the database.
///
/// Data returned from [`get`](Self::get) or cursor operations points directly
/// into the memory-mapped file and is valid for the lifetime of the
/// transaction.
///
/// On creation, a reader slot is acquired from the environment's reader table.
/// The slot is released when the transaction is dropped, allowing the writer
/// to reclaim pages freed after this snapshot's txnid.
pub struct RoTransaction<'env> {
    pub(crate) env: &'env EnvironmentInner,
    txnid: u64,
    /// Snapshot of database metadata for all open databases.
    dbs: Vec<DbStat>,
    /// Reader table slot index, released on drop.
    reader_slot: Option<usize>,
    /// Local snapshot of the env's per-dbi key comparators. Populated at
    /// construction; refreshed when a named database is registered. Lets
    /// the hot path avoid a per-op `RwLock::read` + `Arc::clone`.
    pub(crate) cmp_cache: Vec<Arc<Box<CmpFn>>>,
}

impl<'env> RoTransaction<'env> {
    /// Create a new read-only transaction.
    ///
    /// Acquires a reader slot in the environment's reader table, registering
    /// this transaction's snapshot txnid so the writer knows not to reclaim
    /// pages that this transaction may still reference.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadersFull`] if all reader slots are occupied.
    pub(crate) fn new(env: &'env EnvironmentInner) -> Result<Self> {
        let meta = env.meta();
        let txnid = meta.txnid;
        let slot = env.reader_table.acquire(txnid)?;
        // Snapshot the per-dbi comparator list once — subsequent hot-path
        // lookups then avoid the env RwLock + Arc clone per op.
        let cmp_cache = env.db_cmp.read().map_err(|_| Error::Panic)?.clone();
        Ok(Self {
            env,
            txnid,
            dbs: vec![meta.dbs[0], meta.dbs[1]],
            reader_slot: Some(slot),
            cmp_cache,
        })
    }

    /// Borrow the cached key comparator for `dbi`. Much cheaper than
    /// `env.get_cmp`: no RwLock acquire, no Arc clone.
    #[inline]
    pub(crate) fn cmp_ref(&self, dbi: u32) -> Result<&CmpFn> {
        let arc = self.cmp_cache.get(dbi as usize).ok_or(Error::BadDbi)?;
        Ok(&***arc)
    }

    /// Refresh `cmp_cache` from the env. Call after registering a new DBI.
    fn refresh_cmp_cache(&mut self) -> Result<()> {
        self.cmp_cache = self.env.db_cmp.read().map_err(|_| Error::Panic)?.clone();
        Ok(())
    }

    /// Retrieve data by key from the specified database.
    ///
    /// Returns a reference to the value bytes stored in the memory-mapped
    /// file. The reference is valid for the lifetime of this transaction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the key does not exist.
    /// Returns [`Error::BadDbi`] if `dbi` is invalid.
    pub fn get(&self, dbi: u32, key: &[u8]) -> Result<&[u8]> {
        let db = self.db(dbi)?;
        if db.root == P_INVALID {
            return Err(Error::NotFound);
        }

        let cmp = self.cmp_ref(dbi)?;
        let mut cursor = Cursor::new(self.env.page_size, dbi);
        let get_page = |pgno: u64| self.env.get_page(pgno);

        cursor.page_search(db.root, Some(key), cmp, &get_page)?;

        // The cursor's current_node() returns data from the mmap, which
        // has lifetime 'env. We use transmute to extend the lifetime from
        // the cursor's unbound references to 'self (which is bounded by 'env).
        let node = cursor.current_node().ok_or(Error::NotFound)?;
        let node_key: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.key()) };
        if cmp(key, node_key) != Ordering::Equal {
            return Err(Error::NotFound);
        }

        if node.is_bigdata() {
            let pgno = node.overflow_pgno();
            let ptr = self.env.get_page(pgno)?;
            let data_size = node.data_size() as usize;
            // SAFETY: ptr points into the mmap which lives as long as 'env.
            let data: &[u8] =
                unsafe { std::slice::from_raw_parts(ptr.add(PAGE_HEADER_SIZE), data_size) };
            Ok(data)
        } else if node.is_dupdata() {
            // DUPSORT node with sub-page: return the first dup value.
            let first_val = crate::btree::get_dup_at_index(&node, 0).ok_or(Error::NotFound)?;
            let data: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(first_val) };
            Ok(data)
        } else {
            // SAFETY: node_data() points into the mmap.
            let data: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.node_data()) };
            Ok(data)
        }
    }

    /// Open a read-only cursor on the specified database.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadDbi`] if `dbi` is invalid.
    pub fn open_cursor(&self, dbi: u32) -> Result<RoCursor<'_, 'env>> {
        let _db = self.db(dbi)?;
        let cursor = Cursor::new(self.env.page_size, dbi);
        // Cache cmp + db_flags once at open — hot-path ops (Next on a
        // seq_scan) avoid the env.get_cmp RwLock acquire + Arc clone that
        // would otherwise cost 20-30 ns per call.
        let cmp = self.env.get_cmp(dbi)?;
        let db_flags = self.env.get_db_flags(dbi)?;
        let is_dupsort = db_flags & DatabaseFlags::DUP_SORT.bits() as u16 != 0;
        Ok(RoCursor {
            txn: self,
            cursor,
            cmp,
            is_dupsort,
            dup_idx: 0,
            dup_count: 0,
        })
    }

    /// Open a named database for read-only access.
    ///
    /// If `name` is `None`, returns the handle for the default (main) database.
    /// If `name` is `Some`, looks up the named database in the main database.
    /// Named databases must have been previously created by a write transaction.
    ///
    /// # Errors
    ///
    /// - [`Error::NotFound`] if the named database does not exist
    /// - [`Error::BadDbi`] if the environment was not configured with `max_dbs > 0`
    pub fn open_db(&mut self, name: Option<&str>) -> Result<u32> {
        if let Some(name) = name {
            if self.env.max_dbs == 0 {
                return Err(Error::BadDbi);
            }
            self.find_db(name)
        } else {
            Ok(MAIN_DBI)
        }
    }

    /// Look up a named database in MAIN_DBI (read-only, no creation).
    fn find_db(&mut self, name: &str) -> Result<u32> {
        // Always search MAIN_DBI to get the current DbStat, then check
        // if it's already registered.
        let main_db = self.dbs[MAIN_DBI as usize];
        if main_db.root == P_INVALID {
            return Err(Error::NotFound);
        }

        let cmp = self.env.get_cmp(MAIN_DBI)?;
        let mut cursor = Cursor::new(self.env.page_size, MAIN_DBI);
        let get_page = |pgno: u64| self.env.get_page(pgno);

        cursor.page_search(main_db.root, Some(name.as_bytes()), &*cmp, &get_page)?;

        let node = cursor.current_node().ok_or(Error::NotFound)?;
        if cmp(name.as_bytes(), node.key()) != Ordering::Equal || !node.is_subdata() {
            return Err(Error::NotFound);
        }

        let db_stat = node.sub_db();

        // Check if already registered in the environment.
        {
            let db_names = self.env.db_names.read().map_err(|_| Error::Panic)?;
            for (i, n) in db_names.iter().enumerate() {
                if let Some(n) = n {
                    if n == name {
                        // Already registered — update our local dbs array.
                        while self.dbs.len() <= i {
                            self.dbs.push(DbStat::default());
                        }
                        self.dbs[i] = db_stat;
                        return Ok(i as u32);
                    }
                }
            }
        }

        self.register_db_ro(name, db_stat)
    }

    /// Register a named database for read-only access.
    fn register_db_ro(&mut self, name: &str, db: DbStat) -> Result<u32> {
        use crate::cmp::{default_cmp, default_dcmp};

        let dbi = {
            let db_names = self.env.db_names.read().map_err(|_| Error::Panic)?;
            db_names.len()
        };

        {
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
        }

        // Extend local dbs array.
        while self.dbs.len() <= dbi {
            self.dbs.push(DbStat::default());
        }
        self.dbs[dbi] = db;

        // Keep the local cmp cache in sync with the env.
        self.refresh_cmp_cache()?;

        Ok(dbi as u32)
    }

    /// Return the transaction ID.
    #[must_use]
    pub fn txnid(&self) -> u64 {
        self.txnid
    }

    /// Reset this read-only transaction, releasing its reader slot.
    ///
    /// The transaction handle can be reused later via [`renew`](Self::renew).
    /// This allows recycling transactions without allocation overhead.
    pub fn reset(&mut self) {
        if let Some(slot) = self.reader_slot.take() {
            self.env.reader_table.release(slot);
        }
    }

    /// Renew a previously reset read-only transaction with a fresh snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadTxn`] if the transaction is still active (not reset).
    /// Returns [`Error::ReadersFull`] if no reader slots are available.
    pub fn renew(&mut self) -> Result<()> {
        if self.reader_slot.is_some() {
            return Err(Error::BadTxn);
        }
        let meta = self.env.meta();
        self.txnid = meta.txnid;
        self.dbs = vec![meta.dbs[0], meta.dbs[1]];
        self.reader_slot = Some(self.env.reader_table.acquire(self.txnid)?);
        Ok(())
    }

    /// Return statistics for the specified database.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadDbi`] if `dbi` is invalid.
    pub fn db_stat(&self, dbi: u32) -> Result<Stat> {
        let db = self.db(dbi)?;
        Ok(Stat {
            page_size: self.env.page_size as u32,
            depth: db.depth as u32,
            branch_pages: db.branch_pages,
            leaf_pages: db.leaf_pages,
            overflow_pages: db.overflow_pages,
            entries: db.entries,
        })
    }

    /// Return the database flags for the specified database handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadDbi`] if `dbi` is invalid.
    pub fn dbi_flags(&self, dbi: u32) -> Result<u16> {
        let db = self.db(dbi)?;
        Ok(db.flags)
    }

    /// Resolve a database handle to its metadata.
    fn db(&self, dbi: u32) -> Result<&DbStat> {
        self.dbs.get(dbi as usize).ok_or(Error::BadDbi)
    }
}

impl Drop for RoTransaction<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.reader_slot.take() {
            self.env.reader_table.release(slot);
        }
    }
}

impl std::fmt::Debug for RoTransaction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoTransaction")
            .field("txnid", &self.txnid)
            .field("num_dbs", &self.dbs.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Read-only cursor
// ---------------------------------------------------------------------------

/// A read-only cursor for sequential or positioned access to database entries.
pub struct RoCursor<'txn, 'env> {
    txn: &'txn RoTransaction<'env>,
    cursor: Cursor,
    /// Cached key comparator, populated at `open_cursor`. Avoids a per-op
    /// RwLock acquire + Arc clone on the hot path.
    cmp: Arc<Box<CmpFn>>,
    /// Cached "this DB has DUP_SORT" flag. Lets non-DUPSORT databases skip
    /// dup-count bookkeeping entirely.
    is_dupsort: bool,
    /// Current dup index within a DUPSORT sub-page (0 for non-dup databases).
    dup_idx: usize,
    /// Total dup count at the current key position (1 for non-dup databases).
    dup_count: usize,
}

impl<'txn, 'env> RoCursor<'txn, 'env> {
    /// Position or advance the cursor and return the key/value at the new
    /// position.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the cursor reaches the end or the
    /// requested key is not found.
    pub fn get(&mut self, key: Option<&[u8]>, op: CursorOp) -> Result<(&'txn [u8], &'txn [u8])> {
        // Fast path for the most common op — Next on a non-DUPSORT database.
        // Skips the full match, avoids sync_dup_count + current_node
        // reconstruction. This is the seq_scan / range_scan hot path.
        if !self.is_dupsort && matches!(op, CursorOp::Next) {
            let get_page = |pgno: u64| self.txn.env.get_page(pgno);
            self.cursor.next(&get_page)?;
            return self.current_kv_nondup();
        }

        let db = self.txn.db(self.cursor.dbi)?;
        let root = db.root;
        let cmp: &CmpFn = &**self.cmp;
        let get_page = |pgno: u64| self.txn.env.get_page(pgno);

        match op {
            CursorOp::First => {
                self.cursor.first(root, cmp, &get_page)?;
                self.dup_idx = 0;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
            }
            CursorOp::Last => {
                self.cursor.last(root, cmp, &get_page)?;
                if self.is_dupsort {
                    self.sync_dup_count();
                    self.dup_idx = self.dup_count.saturating_sub(1);
                } else {
                    self.dup_idx = 0;
                }
            }
            CursorOp::Next => {
                // Only reached for DUPSORT (non-dup fast-pathed above).
                if self.dup_count > 1 && self.dup_idx + 1 < self.dup_count {
                    self.dup_idx += 1;
                } else {
                    self.cursor.next(&get_page)?;
                    self.dup_idx = 0;
                    self.sync_dup_count();
                }
            }
            CursorOp::NextNoDup => {
                self.cursor.next(&get_page)?;
                self.dup_idx = 0;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
            }
            CursorOp::NextDup => {
                if self.dup_idx + 1 < self.dup_count {
                    self.dup_idx += 1;
                } else {
                    return Err(Error::NotFound);
                }
            }
            CursorOp::Prev => {
                if self.dup_count > 1 && self.dup_idx > 0 {
                    self.dup_idx -= 1;
                } else {
                    self.cursor.prev(&get_page)?;
                    if self.is_dupsort {
                        self.sync_dup_count();
                        self.dup_idx = self.dup_count.saturating_sub(1);
                    } else {
                        self.dup_idx = 0;
                    }
                }
            }
            CursorOp::PrevNoDup => {
                self.cursor.prev(&get_page)?;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
                self.dup_idx = 0;
            }
            CursorOp::PrevDup => {
                if self.dup_idx > 0 {
                    self.dup_idx -= 1;
                } else {
                    return Err(Error::NotFound);
                }
            }
            CursorOp::FirstDup => {
                if !self.cursor.is_initialized() {
                    return Err(Error::NotFound);
                }
                self.dup_idx = 0;
            }
            CursorOp::LastDup => {
                if !self.cursor.is_initialized() {
                    return Err(Error::NotFound);
                }
                if self.is_dupsort {
                    self.sync_dup_count();
                    self.dup_idx = self.dup_count.saturating_sub(1);
                }
            }
            CursorOp::Set | CursorOp::SetKey => {
                let k = key.ok_or(Error::BadValSize)?;
                self.cursor.set(root, k, cmp, &get_page)?;
                self.dup_idx = 0;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
            }
            CursorOp::SetRange => {
                let k = key.ok_or(Error::BadValSize)?;
                self.cursor.page_search(root, Some(k), cmp, &get_page)?;
                if self.cursor.current_key().is_none() {
                    return Err(Error::NotFound);
                }
                self.dup_idx = 0;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
            }
            CursorOp::GetCurrent => {
                if !self.cursor.is_initialized() {
                    return Err(Error::NotFound);
                }
            }
            CursorOp::GetBoth => {
                let k = key.ok_or(Error::BadValSize)?;
                self.cursor.set(root, k, cmp, &get_page)?;
                // GetBoth is not fully supported without a data parameter;
                // position at the key only.
                self.dup_idx = 0;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
            }
            CursorOp::GetBothRange => {
                let k = key.ok_or(Error::BadValSize)?;
                self.cursor.set(root, k, cmp, &get_page)?;
                self.dup_idx = 0;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
            }
            CursorOp::GetMultiple | CursorOp::NextMultiple => {
                if op == CursorOp::NextMultiple {
                    // Advance to the next key first.
                    self.cursor.next(&get_page)?;
                    self.dup_idx = 0;
                    if self.is_dupsort {
                        self.sync_dup_count();
                    }
                } else if !self.cursor.is_initialized() {
                    return Err(Error::NotFound);
                }
                return self.get_multiple_kv();
            }
            CursorOp::PrevMultiple => {
                self.cursor.prev(&get_page)?;
                self.dup_idx = 0;
                if self.is_dupsort {
                    self.sync_dup_count();
                }
                return self.get_multiple_kv();
            }
        }

        self.current_kv()
    }

    /// Fast-path `current_kv` for non-DUPSORT databases — avoids the
    /// `is_dupdata()` branch (always false) and saves a check per call.
    fn current_kv_nondup(&self) -> Result<(&'txn [u8], &'txn [u8])> {
        let node = self.cursor.current_node().ok_or(Error::NotFound)?;
        // SAFETY: node data comes from mmap owned by env (outlives 'txn).
        let key: &'txn [u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.key()) };

        let data: &'txn [u8] = if node.is_bigdata() {
            let pgno = node.overflow_pgno();
            let ptr = self.txn.env.get_page(pgno)?;
            let data_size = node.data_size() as usize;
            unsafe {
                let start = ptr.add(PAGE_HEADER_SIZE);
                std::mem::transmute::<&[u8], &[u8]>(std::slice::from_raw_parts(start, data_size))
            }
        } else {
            unsafe { std::mem::transmute::<&[u8], &[u8]>(node.node_data()) }
        };

        Ok((key, data))
    }

    /// Return all DUPFIXED values at the current position as a contiguous
    /// byte slice.
    ///
    /// For `DUPFIXED` databases, the sub-page stores values contiguously
    /// without node headers (LEAF2 format). This returns the key and the
    /// raw data area containing all dup values packed together.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Incompatible`] if the current node is not a DUPFIXED
    /// sub-page. Returns [`Error::NotFound`] if the cursor is not positioned.
    fn get_multiple_kv(&self) -> Result<(&'txn [u8], &'txn [u8])> {
        let node = self.cursor.current_node().ok_or(Error::NotFound)?;
        if !node.is_dupdata() {
            return Err(Error::Incompatible);
        }

        let sub_page_data = node.node_data();
        let page = Page::from_raw(sub_page_data);
        if !page.is_leaf2() {
            return Err(Error::Incompatible);
        }

        // SAFETY: node data comes from the mmap which has lifetime >= 'env >= 'txn.
        let key: &'txn [u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.key()) };

        let val_size = page.pad() as usize;
        let num_vals = page.num_keys();
        let data_len = num_vals * val_size;
        let data: &'txn [u8] = unsafe {
            std::mem::transmute::<&[u8], &[u8]>(
                &sub_page_data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + data_len],
            )
        };
        Ok((key, data))
    }

    /// Synchronize the dup_count from the current node.
    fn sync_dup_count(&mut self) {
        if let Some(node) = self.cursor.current_node() {
            self.dup_count = crate::btree::dup_count(&node);
        } else {
            self.dup_count = 1;
        }
    }

    /// Return the current key/value pair.
    ///
    /// # Safety
    ///
    /// The returned references point into mmap'd memory. We transmute the
    /// lifetimes from the cursor's internal raw-pointer-derived references
    /// to `'txn`, which is sound because the mmap outlives both the cursor
    /// and the transaction.
    fn current_kv(&self) -> Result<(&'txn [u8], &'txn [u8])> {
        let node = self.cursor.current_node().ok_or(Error::NotFound)?;

        // SAFETY: node data comes from the mmap which has lifetime >= 'env >= 'txn.
        // The cursor stores raw pointers into the mmap. Transmuting to 'txn is
        // safe because the transaction holds the mmap pinned.
        let key: &'txn [u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.key()) };

        let data: &'txn [u8] = if node.is_bigdata() {
            let pgno = node.overflow_pgno();
            let ptr = self.txn.env.get_page(pgno)?;
            let data_size = node.data_size() as usize;
            unsafe {
                let start = ptr.add(PAGE_HEADER_SIZE);
                std::mem::transmute::<&[u8], &[u8]>(std::slice::from_raw_parts(start, data_size))
            }
        } else if node.is_dupdata() {
            // DUPSORT node: return the dup value at dup_idx.
            let val = crate::btree::get_dup_at_index(&node, self.dup_idx).ok_or(Error::NotFound)?;
            unsafe { std::mem::transmute::<&[u8], &[u8]>(val) }
        } else {
            unsafe { std::mem::transmute::<&[u8], &[u8]>(node.node_data()) }
        };

        Ok((key, data))
    }

    /// Return the number of duplicate values at the current cursor position.
    ///
    /// For non-DUPSORT databases this always returns 1. The cursor must be
    /// positioned at a valid entry (i.e., initialized) before calling this
    /// method.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the cursor is not positioned.
    pub fn count(&self) -> Result<usize> {
        if !self.cursor.is_initialized() {
            return Err(Error::NotFound);
        }
        Ok(self.dup_count)
    }

    /// Create an iterator over all key/value pairs in the database,
    /// starting from the first entry.
    pub fn iter(&mut self) -> CursorIter<'txn, 'env, '_> {
        CursorIter {
            cursor: self,
            started: false,
            op: CursorOp::Next,
            first_op: CursorOp::First,
        }
    }

    /// Create an iterator starting from the given key (or the first key
    /// that is >= the given key).
    ///
    /// # Errors
    ///
    /// Returns an error if the database is empty or the cursor cannot be
    /// positioned.
    pub fn iter_from(&mut self, key: &[u8]) -> Result<CursorIter<'txn, 'env, '_>> {
        let db = self.txn.db(self.cursor.dbi)?;
        let cmp = self.txn.env.get_cmp(self.cursor.dbi)?;
        let get_page = |pgno: u64| self.txn.env.get_page(pgno);
        self.cursor
            .page_search(db.root, Some(key), &*cmp, &get_page)?;
        Ok(CursorIter {
            cursor: self,
            started: true,
            op: CursorOp::Next,
            first_op: CursorOp::GetCurrent,
        })
    }
}

impl std::fmt::Debug for RoCursor<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoCursor")
            .field("cursor", &self.cursor)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Cursor iterator
// ---------------------------------------------------------------------------

/// An iterator over cursor positions, yielding `(key, value)` pairs.
pub struct CursorIter<'txn, 'env, 'cursor> {
    cursor: &'cursor mut RoCursor<'txn, 'env>,
    started: bool,
    op: CursorOp,
    first_op: CursorOp,
}

impl<'txn, 'env, 'cursor> Iterator for CursorIter<'txn, 'env, 'cursor> {
    type Item = Result<(&'txn [u8], &'txn [u8])>;

    fn next(&mut self) -> Option<Self::Item> {
        let op = if self.started {
            self.op
        } else {
            self.started = true;
            self.first_op
        };

        match self.cursor.get(None, op) {
            Ok(kv) => Some(Ok(kv)),
            Err(Error::NotFound) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
