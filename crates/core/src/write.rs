//! Write transaction: COW B+ tree mutations, page allocation, and commit.
//!
//! Provides [`RwTransaction`] for read-write access to the database. Only one
//! write transaction may be active at a time. All page modifications use
//! copy-on-write: the original pages in the mmap remain intact for concurrent
//! readers, and dirty copies are flushed to disk on commit.

use std::cmp::Ordering;

use crate::{
    cursor::Cursor,
    env::EnvironmentInner,
    error::{Error, Result},
    node::{init_page, node_add, node_del},
    page::Page,
    types::*,
};

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
    entries: Vec<(u64, PageBuf)>,
}

impl DirtyPages {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

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
    /// Dirty page buffers (modified pages).
    pub(crate) dirty: DirtyPages,
    /// Per-database metadata snapshots.
    pub(crate) dbs: Vec<DbStat>,
    /// Per-database dirty flags.
    pub(crate) db_dirty: Vec<bool>,
    /// Whether this transaction has been committed or aborted.
    finished: bool,
}

impl<'env> RwTransaction<'env> {
    /// Create a new write transaction.
    pub(crate) fn new(env: &'env EnvironmentInner) -> Result<Self> {
        let meta = env.meta();
        let txnid = meta.txnid + 1;
        let next_pgno = meta.last_pgno + 1;
        let num_dbs = CORE_DBS as usize;

        Ok(Self {
            env,
            txnid,
            next_pgno,
            free_pgs: Vec::new(),
            dirty: DirtyPages::new(),
            dbs: vec![meta.dbs[0], meta.dbs[1]],
            db_dirty: vec![false; num_dbs],
            finished: false,
        })
    }

    /// Resolve a page number to a page pointer.
    ///
    /// Checks the dirty list first, then falls back to the mmap.
    pub(crate) fn get_page(&self, pgno: u64) -> Result<*const u8> {
        if let Some(buf) = self.dirty.find(pgno) {
            return Ok(buf.as_ptr());
        }
        self.env.get_page(pgno)
    }

    /// Allocate a new page. Returns the page number and an owned buffer.
    fn page_alloc(&mut self) -> Result<(u64, PageBuf)> {
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
    fn page_touch(&mut self, pgno: u64) -> Result<u64> {
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
        let src = unsafe { std::slice::from_raw_parts(src_ptr, self.env.page_size) };
        new_buf.as_mut_slice().copy_from_slice(src);

        // Update pgno in the new page
        new_buf.as_mut_slice()[0..8].copy_from_slice(&new_pgno.to_le_bytes());

        // Set DIRTY flag
        let flags_raw = u16::from_le_bytes([new_buf.as_slice()[10], new_buf.as_slice()[11]]);
        let new_flags = flags_raw | PageFlags::DIRTY.bits();
        new_buf.as_mut_slice()[10..12].copy_from_slice(&new_flags.to_le_bytes());

        self.dirty.insert(new_pgno, new_buf);
        Ok(new_pgno)
    }

    /// Insert a key/value pair into the specified database.
    ///
    /// # Errors
    ///
    /// - [`Error::BadValSize`] if key exceeds max key size
    /// - [`Error::KeyExist`] if `NO_OVERWRITE` is set and key exists
    /// - [`Error::PageFull`] if the leaf page is full (requires page split, implemented in Phase 3)
    /// - [`Error::MapFull`] if no more pages can be allocated
    pub fn put(&mut self, dbi: u32, key: &[u8], data: &[u8], flags: WriteFlags) -> Result<()> {
        if key.len() > self.env.max_key_size || key.is_empty() {
            return Err(Error::BadValSize);
        }

        let db = *self.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
        let cmp = self.env.get_cmp(dbi)?;
        let page_size = self.env.page_size;

        if db.root == P_INVALID {
            // Empty database — create new root leaf page
            let (root_pgno, mut root_buf) = self.page_alloc()?;
            init_page(
                root_buf.as_mut_slice(),
                root_pgno,
                PageFlags::LEAF | PageFlags::DIRTY,
                page_size,
            );

            node_add(
                root_buf.as_mut_slice(),
                page_size,
                0,
                key,
                data,
                0,
                NodeFlags::empty(),
            )?;

            self.dirty.insert(root_pgno, root_buf);

            let db_mut = self.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = root_pgno;
            db_mut.depth = 1;
            db_mut.leaf_pages = 1;
            db_mut.entries = 1;
            self.db_dirty[dbi as usize] = true;

            return Ok(());
        }

        // COW the root page
        let root_pgno = self.page_touch(db.root)?;

        // Search for the insertion point on the dirty page
        let mut cursor = Cursor::new(page_size, dbi);
        let get_page = |pgno: u64| -> Result<*const u8> { self.get_page(pgno) };
        cursor.page_search(root_pgno, Some(key), &*cmp, &get_page)?;
        let insert_idx = cursor.current_index();

        // Check for existing key
        let mut overwrite = false;
        if let Some(node) = cursor.current_node() {
            if cmp(key, node.key()) == Ordering::Equal {
                if flags.contains(WriteFlags::NO_OVERWRITE) {
                    return Err(Error::KeyExist);
                }
                overwrite = true;
            }
        }

        let dirty_buf = self.dirty.find_mut(root_pgno).ok_or(Error::Corrupted)?;

        if overwrite {
            node_del(dirty_buf.as_mut_slice(), page_size, insert_idx);
        }

        node_add(
            dirty_buf.as_mut_slice(),
            page_size,
            insert_idx,
            key,
            data,
            0,
            NodeFlags::empty(),
        )?;

        let db_mut = self.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = root_pgno;
        if !overwrite {
            db_mut.entries += 1;
        }
        self.db_dirty[dbi as usize] = true;

        Ok(())
    }

    /// Delete a key from the specified database.
    ///
    /// # Errors
    ///
    /// - [`Error::NotFound`] if the key does not exist
    pub fn del(&mut self, dbi: u32, key: &[u8]) -> Result<()> {
        let db = *self.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
        if db.root == P_INVALID {
            return Err(Error::NotFound);
        }

        let cmp = self.env.get_cmp(dbi)?;
        let page_size = self.env.page_size;

        // COW the root
        let root_pgno = self.page_touch(db.root)?;

        // Search for the key
        let mut cursor = Cursor::new(page_size, dbi);
        let get_page = |pgno: u64| -> Result<*const u8> { self.get_page(pgno) };
        cursor.page_search(root_pgno, Some(key), &*cmp, &get_page)?;

        // Verify exact match
        let node = cursor.current_node().ok_or(Error::NotFound)?;
        if cmp(key, node.key()) != Ordering::Equal {
            return Err(Error::NotFound);
        }

        let idx = cursor.current_index();
        let dirty_buf = self.dirty.find_mut(root_pgno).ok_or(Error::Corrupted)?;
        node_del(dirty_buf.as_mut_slice(), page_size, idx);

        let db_mut = self.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = root_pgno;
        db_mut.entries = db_mut.entries.saturating_sub(1);
        self.db_dirty[dbi as usize] = true;

        // If page is now empty, mark database as empty
        let page = Page::from_raw(dirty_buf.as_slice());
        if page.num_keys() == 0 {
            db_mut.root = P_INVALID;
            db_mut.depth = 0;
            db_mut.leaf_pages = 0;
        }

        Ok(())
    }

    /// Read a value within a write transaction.
    ///
    /// Checks dirty pages first, falls back to the mmap.
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
        // SAFETY: data is from mmap or dirty pages, both live long enough.
        let node_key: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.key()) };
        if cmp(key, node_key) != Ordering::Equal {
            return Err(Error::NotFound);
        }

        if node.is_bigdata() {
            let pgno = node.overflow_pgno();
            let ptr = self.get_page(pgno)?;
            let data_size = node.data_size() as usize;
            let data: &[u8] =
                unsafe { std::slice::from_raw_parts(ptr.add(PAGE_HEADER_SIZE), data_size) };
            Ok(data)
        } else {
            let data: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(node.node_data()) };
            Ok(data)
        }
    }

    /// Commit the transaction: flush dirty pages and write the meta page.
    ///
    /// After a successful commit, the transaction is consumed and all changes
    /// become visible to subsequent transactions.
    pub fn commit(mut self) -> Result<()> {
        if self.finished {
            return Err(Error::BadTxn);
        }
        self.finished = true;

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

    /// Write all dirty pages to the data file via `pwrite`.
    fn flush_dirty_pages(&self) -> Result<()> {
        let page_size = self.env.page_size;
        let fd = self.env.data_fd();

        for (pgno, buf) in self.dirty.iter() {
            let offset = *pgno as i64 * page_size as i64;
            let data = buf.as_slice();
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
            let ret = unsafe { libc::fcntl(fd, libc::F_FULLFSYNC) };
            if ret < 0 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let ret = unsafe { libc::fdatasync(fd) };
            if ret < 0 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
        }
        Ok(())
    }

    /// Write the meta page to commit the transaction.
    fn write_meta(&self) -> Result<()> {
        let page_size = self.env.page_size;
        let toggle = (self.txnid & 1) as usize;

        let mut meta_buf = vec![0u8; page_size];
        // Page header: pgno + flags
        let pgno = toggle as u64;
        meta_buf[0..8].copy_from_slice(&pgno.to_le_bytes());
        meta_buf[10..12].copy_from_slice(&PageFlags::META.bits().to_le_bytes());

        // Meta payload
        let meta = Meta {
            magic: MDB_MAGIC,
            version: MDB_DATA_VERSION,
            address: 0,
            map_size: self.env.map_size as u64,
            dbs: [self.dbs[FREE_DBI as usize], self.dbs[MAIN_DBI as usize]],
            last_pgno: self.next_pgno - 1,
            txnid: self.txnid,
        };

        let meta_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&meta).cast::<u8>(),
                std::mem::size_of::<Meta>(),
            )
        };
        meta_buf[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + meta_bytes.len()].copy_from_slice(meta_bytes);

        let fd = self.env.data_fd();
        let offset = (toggle * page_size) as i64;
        let written = unsafe { libc::pwrite(fd, meta_buf.as_ptr().cast(), meta_buf.len(), offset) };
        if written < 0 || written as usize != meta_buf.len() {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }

        Ok(())
    }
}

impl Drop for RwTransaction<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Implicitly abort — discard dirty pages
            self.finished = true;
        }
    }
}

impl std::fmt::Debug for RwTransaction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RwTransaction")
            .field("txnid", &self.txnid)
            .field("next_pgno", &self.next_pgno)
            .field("dirty_pages", &self.dirty.len())
            .field("finished", &self.finished)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::{
        env::Environment,
        types::{MAIN_DBI, WriteFlags},
    };

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
            txn.put(MAIN_DBI, b"hello", b"world", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // Read it back in a new read txn
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let val = txn.get(MAIN_DBI, b"hello").expect("get");
            assert_eq!(val, b"world");
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
            txn.put(MAIN_DBI, b"aaa", b"111", WriteFlags::empty())
                .expect("put aaa");
            txn.put(MAIN_DBI, b"bbb", b"222", WriteFlags::empty())
                .expect("put bbb");
            txn.put(MAIN_DBI, b"ccc", b"333", WriteFlags::empty())
                .expect("put ccc");
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            assert_eq!(txn.get(MAIN_DBI, b"aaa").expect("get"), b"111");
            assert_eq!(txn.get(MAIN_DBI, b"bbb").expect("get"), b"222");
            assert_eq!(txn.get(MAIN_DBI, b"ccc").expect("get"), b"333");
        }
    }

    #[test]
    fn test_should_overwrite_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI, b"key", b"v1", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI, b"key", b"v2", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            assert_eq!(txn.get(MAIN_DBI, b"key").expect("get"), b"v2");
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
            txn.put(MAIN_DBI, b"key", b"v1", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let result = txn.put(MAIN_DBI, b"key", b"v2", WriteFlags::NO_OVERWRITE);
            assert!(matches!(result, Err(crate::error::Error::KeyExist)));
            txn.abort();
        }
    }

    #[test]
    fn test_should_delete_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI, b"key", b"val", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.del(MAIN_DBI, b"key").expect("del");
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let result = txn.get(MAIN_DBI, b"key");
            assert!(matches!(result, Err(crate::error::Error::NotFound)));
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
            txn.put(MAIN_DBI, b"key", b"val", WriteFlags::empty())
                .expect("put");
            txn.abort();
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let result = txn.get(MAIN_DBI, b"key");
            assert!(matches!(result, Err(crate::error::Error::NotFound)));
        }
    }

    #[test]
    fn test_should_read_within_write_txn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        txn.put(MAIN_DBI, b"key", b"val", WriteFlags::empty())
            .expect("put");

        // Read within the same write txn
        let val = txn.get(MAIN_DBI, b"key").expect("get");
        assert_eq!(val, b"val");

        txn.commit().expect("commit");
    }

    #[test]
    fn test_should_delete_nonexistent_returns_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        let result = txn.del(MAIN_DBI, b"missing");
        assert!(matches!(result, Err(crate::error::Error::NotFound)));
        txn.abort();
    }

    #[test]
    fn test_should_persist_across_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Write data
        {
            let env = Environment::builder()
                .map_size(1024 * 1024)
                .open(dir.path())
                .expect("open");
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI, b"persist", b"test", WriteFlags::empty())
                .expect("put");
            txn.commit().expect("commit");
        }

        // Reopen and verify
        {
            let env = Environment::builder()
                .map_size(1024 * 1024)
                .open(dir.path())
                .expect("reopen");
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let val = txn.get(MAIN_DBI, b"persist").expect("get");
            assert_eq!(val, b"test");
        }
    }
}
