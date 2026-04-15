//! Transaction types for MVCC read/write access.
//!
//! LMDB uses a single-writer, multi-reader concurrency model.
//! [`RoTransaction`] provides zero-copy read access to a consistent snapshot.
//! Write transactions will be added in Phase 2.

use std::cmp::Ordering;

use crate::{
    cursor::Cursor,
    env::EnvironmentInner,
    error::{Error, Result},
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
pub struct RoTransaction<'env> {
    pub(crate) env: &'env EnvironmentInner,
    txnid: u64,
    /// Snapshot of database metadata for all open databases.
    dbs: Vec<DbStat>,
}

impl<'env> RoTransaction<'env> {
    /// Create a new read-only transaction.
    pub(crate) fn new(env: &'env EnvironmentInner) -> Result<Self> {
        let meta = env.meta();
        Ok(Self {
            env,
            txnid: meta.txnid,
            dbs: vec![meta.dbs[0], meta.dbs[1]],
        })
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

        let cmp = self.env.get_cmp(dbi)?;
        let mut cursor = Cursor::new(self.env.page_size, dbi);
        let get_page = |pgno: u64| self.env.get_page(pgno);

        cursor.page_search(db.root, Some(key), &*cmp, &get_page)?;

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
        Ok(RoCursor { txn: self, cursor })
    }

    /// Return the transaction ID.
    #[must_use]
    pub fn txnid(&self) -> u64 {
        self.txnid
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

    /// Resolve a database handle to its metadata.
    fn db(&self, dbi: u32) -> Result<&DbStat> {
        self.dbs.get(dbi as usize).ok_or(Error::BadDbi)
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
        let db = self.txn.db(self.cursor.dbi)?;
        let root = db.root;
        let cmp = self.txn.env.get_cmp(self.cursor.dbi)?;
        let get_page = |pgno: u64| self.txn.env.get_page(pgno);

        match op {
            CursorOp::First => self.cursor.first(root, &*cmp, &get_page)?,
            CursorOp::Last => self.cursor.last(root, &*cmp, &get_page)?,
            CursorOp::Next | CursorOp::NextNoDup => self.cursor.next(&get_page)?,
            CursorOp::Prev | CursorOp::PrevNoDup => self.cursor.prev(&get_page)?,
            CursorOp::Set | CursorOp::SetKey => {
                let k = key.ok_or(Error::BadValSize)?;
                self.cursor.set(root, k, &*cmp, &get_page)?;
            }
            CursorOp::SetRange => {
                let k = key.ok_or(Error::BadValSize)?;
                self.cursor.page_search(root, Some(k), &*cmp, &get_page)?;
                if self.cursor.current_key().is_none() {
                    return Err(Error::NotFound);
                }
            }
            CursorOp::GetCurrent => {
                if !self.cursor.is_initialized() {
                    return Err(Error::NotFound);
                }
            }
            // DUPSORT operations will be implemented in Phase 5
            _ => return Err(Error::Incompatible),
        }

        self.current_kv()
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
        } else {
            unsafe { std::mem::transmute::<&[u8], &[u8]>(node.node_data()) }
        };

        Ok((key, data))
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
