//! Database environment: file management, memory mapping, and configuration.
//!
//! The [`Environment`] type is the main entry point for opening an LMDB database.
//! It manages the memory-mapped data file, tracks per-database comparison
//! functions, and enforces single-writer semantics via a mutex.
//!
//! # Examples
//!
//! ```no_run
//! use lmdb_rs_core::env::Environment;
//!
//! let env = Environment::builder()
//!     .map_size(10 * 1024 * 1024)
//!     .open("/tmp/my-lmdb")
//!     .expect("failed to open environment");
//! ```

use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    mem,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
};

use memmap2::{MmapOptions, MmapRaw};

use crate::{
    cmp::{CmpFn, default_cmp, default_dcmp},
    error::{Error, Result},
    page::Page,
    types::{DbStat, MDB_DATA_VERSION, MDB_MAGIC, Meta, P_INVALID, PAGE_HEADER_SIZE, PageFlags},
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of meta pages at the start of the data file.
const NUM_METAS: usize = 2;

/// Core databases: `FREE_DBI` (freelist) and `MAIN_DBI` (root namespace).
const CORE_DBS: u32 = 2;

/// Database index for the free-page list.
const FREE_DBI: u32 = 0;

/// Database index for the main (root) B+ tree namespace.
const MAIN_DBI: u32 = 1;

/// Default memory-map size (1 GiB).
const DEFAULT_MAPSIZE: usize = 1 << 30;

/// Default maximum number of reader slots.
const DEFAULT_READERS: u32 = 126;

/// Maximum key size (511 bytes, matching LMDB default for 4096-byte pages).
const MAX_KEY_SIZE: usize = 511;

// Re-export types used in the public API from their canonical location.
pub use crate::types::{EnvFlags, EnvInfo, Stat};

// ---------------------------------------------------------------------------
// EnvironmentBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and opening an [`Environment`].
///
/// # Examples
///
/// ```no_run
/// use lmdb_rs_core::env::EnvironmentBuilder;
///
/// let env = EnvironmentBuilder::new()
///     .map_size(64 * 1024 * 1024)
///     .max_dbs(4)
///     .open("/tmp/my-lmdb")
///     .expect("open failed");
/// ```
#[derive(Debug, Clone)]
pub struct EnvironmentBuilder {
    map_size: usize,
    max_readers: u32,
    max_dbs: u32,
    flags: EnvFlags,
}

impl EnvironmentBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            map_size: DEFAULT_MAPSIZE,
            max_readers: DEFAULT_READERS,
            max_dbs: 0,
            flags: EnvFlags::empty(),
        }
    }

    /// Set the memory-map size in bytes.
    pub fn map_size(mut self, size: usize) -> Self {
        self.map_size = size;
        self
    }

    /// Set the maximum number of concurrent reader slots.
    pub fn max_readers(mut self, readers: u32) -> Self {
        self.max_readers = readers;
        self
    }

    /// Set the maximum number of named databases.
    pub fn max_dbs(mut self, dbs: u32) -> Self {
        self.max_dbs = dbs;
        self
    }

    /// Set environment flags.
    pub fn flags(mut self, flags: EnvFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Open the environment at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not exist, the data file is invalid,
    /// or memory mapping fails.
    pub fn open<P: AsRef<Path>>(self, path: P) -> Result<Environment> {
        Environment::open_with_config(path.as_ref(), self)
    }
}

impl Default for EnvironmentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReaderTable — tracks active read transaction snapshots
// ---------------------------------------------------------------------------

/// Tracks active reader transaction IDs for correct page reclamation.
///
/// Each read transaction registers its snapshot txnid. The writer scans
/// the table to find the oldest active reader, and only reclaims pages
/// freed by transactions older than that.
///
/// The sentinel value `u64::MAX` indicates a free (unused) slot.
pub(crate) struct ReaderTable {
    /// Slots for reader txnids. `u64::MAX` means the slot is free.
    slots: Vec<AtomicU64>,
}

/// Sentinel value indicating a free reader slot.
const READER_SLOT_FREE: u64 = u64::MAX;

impl ReaderTable {
    /// Create a new reader table with `max_readers` slots, all initially free.
    pub(crate) fn new(max_readers: u32) -> Self {
        let mut slots = Vec::with_capacity(max_readers as usize);
        for _ in 0..max_readers {
            slots.push(AtomicU64::new(READER_SLOT_FREE));
        }
        Self { slots }
    }

    /// Acquire a reader slot and set the txnid.
    ///
    /// Returns the slot index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadersFull`] if no slots are available.
    pub(crate) fn acquire(&self, txnid: u64) -> Result<usize> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot
                .compare_exchange(
                    READER_SLOT_FREE,
                    txnid,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                )
                .is_ok()
            {
                return Ok(i);
            }
        }
        Err(Error::ReadersFull)
    }

    /// Release a reader slot, marking it as free.
    pub(crate) fn release(&self, slot_idx: usize) {
        if slot_idx < self.slots.len() {
            self.slots[slot_idx].store(READER_SLOT_FREE, AtomicOrdering::Release);
        }
    }

    /// Find the oldest (minimum) active reader txnid.
    ///
    /// Returns `u64::MAX` if no readers are active.
    pub(crate) fn find_oldest(&self) -> u64 {
        let mut oldest = READER_SLOT_FREE;
        for slot in &self.slots {
            let txnid = slot.load(AtomicOrdering::Acquire);
            if txnid < oldest {
                oldest = txnid;
            }
        }
        oldest
    }

    /// Count the number of active readers.
    pub(crate) fn active_count(&self) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.load(AtomicOrdering::Relaxed) != READER_SLOT_FREE)
            .count() as u32
    }
}

impl std::fmt::Debug for ReaderTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReaderTable")
            .field("total_slots", &self.slots.len())
            .field("active_count", &self.active_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Environment / EnvironmentInner
// ---------------------------------------------------------------------------

/// A handle to an LMDB environment (database directory + memory-mapped file).
///
/// Internally reference-counted via [`Arc`] so cloning is cheap and all
/// clones share the same underlying mmap.
#[derive(Debug, Clone)]
pub struct Environment {
    inner: Arc<EnvironmentInner>,
}

/// Shared, interior state of an environment.
#[allow(dead_code)] // Fields used in later phases
pub(crate) struct EnvironmentInner {
    /// Memory-mapped data file (raw, allows both read and write views).
    mmap: MmapRaw,
    /// Path to the database directory (or data file if `NO_SUB_DIR`).
    path: PathBuf,
    /// Data file handle — kept open for the lifetime of the environment.
    _data_file: File,
    /// Page size in bytes (detected from meta pages).
    pub(crate) page_size: usize,
    /// Configured (or detected) map size.
    pub(crate) map_size: usize,
    /// Maximum valid page number (`map_size / page_size - 1`).
    pub(crate) max_pgno: u64,
    /// Maximum number of named databases.
    pub(crate) max_dbs: u32,
    /// Maximum number of reader slots.
    pub(crate) max_readers: u32,
    /// Environment flags.
    pub(crate) flags: EnvFlags,
    /// Maximum key size in bytes.
    pub(crate) max_key_size: usize,
    /// Maximum node size (key + data) that fits on a page without overflow.
    pub(crate) node_max: usize,
    /// Maximum number of free-list entries per overflow page.
    pub(crate) max_free_per_page: usize,
    /// Per-database key comparison functions.
    pub(crate) db_cmp: RwLock<Vec<Arc<Box<CmpFn>>>>,
    /// Per-database data comparison functions (for `DUPSORT`).
    pub(crate) db_dcmp: RwLock<Vec<Arc<Box<CmpFn>>>>,
    /// Per-database names (`None` for core DBs).
    pub(crate) db_names: RwLock<Vec<Option<String>>>,
    /// Per-database flags.
    pub(crate) db_flags: RwLock<Vec<u16>>,
    /// Writer mutex — only one write transaction at a time (in-process).
    pub(crate) write_mutex: Mutex<()>,
    /// In-process reader table tracking active read transaction snapshots.
    pub(crate) reader_table: ReaderTable,
    /// Lock file for cross-process writer serialization.
    pub(crate) lock_file: Option<File>,
}

// SAFETY: The raw mmap pointer is only accessed through methods that uphold
// Rust's aliasing rules (read-only references for read transactions, exclusive
// access for the single writer).
unsafe impl Send for EnvironmentInner {}
// SAFETY: All mutable state is protected by RwLock / Mutex.
unsafe impl Sync for EnvironmentInner {}

impl std::fmt::Debug for EnvironmentInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvironmentInner")
            .field("path", &self.path)
            .field("page_size", &self.page_size)
            .field("map_size", &self.map_size)
            .field("max_pgno", &self.max_pgno)
            .field("max_dbs", &self.max_dbs)
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// EnvironmentInner helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)] // Methods used in later phases
impl EnvironmentInner {
    /// Return a raw pointer to the start of page `pgno` inside the mmap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PageNotFound`] if `pgno` exceeds [`max_pgno`](Self::max_pgno).
    pub(crate) fn get_page(&self, pgno: u64) -> Result<*const u8> {
        if pgno > self.max_pgno {
            return Err(Error::PageNotFound);
        }
        let offset = pgno as usize * self.page_size;
        // SAFETY: `pgno <= max_pgno` guarantees the offset is within the mapped
        // region. The caller must not hold the returned pointer past the
        // lifetime of `self`.
        let ptr = unsafe { self.mmap.as_ptr().add(offset) };
        Ok(ptr)
    }

    /// Return a [`Page`] view for the given page number.
    pub(crate) fn page(&self, pgno: u64) -> Result<Page<'_>> {
        let ptr = self.get_page(pgno)?;
        // SAFETY: the pointer is valid for `page_size` bytes within the mmap.
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.page_size) };
        Ok(Page::from_raw(slice))
    }

    /// Return the [`Meta`] structure from the meta page with the highest
    /// committed transaction ID.
    ///
    /// Both meta pages (0 and 1) are compared; the one with the larger
    /// `txnid` is considered current.
    pub(crate) fn meta(&self) -> Meta {
        let m0 = self.meta_at(0);
        let m1 = self.meta_at(1);
        if m1.txnid > m0.txnid { m1 } else { m0 }
    }

    /// Read the [`Meta`] structure from meta page `idx` (0 or 1).
    pub(crate) fn meta_at(&self, idx: usize) -> Meta {
        debug_assert!(idx < NUM_METAS, "meta index out of range");
        let pgno = idx as u64;
        // Meta pages are always pages 0 and 1; they must be valid.
        let ptr = unsafe { self.mmap.as_ptr().add(pgno as usize * self.page_size) };
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.page_size) };
        let page = Page::from_raw(slice);
        page.meta()
    }

    /// Return the page size.
    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }

    /// Clone the key comparison function for `dbi`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadDbi`] if `dbi` is out of range.
    pub(crate) fn get_cmp(&self, dbi: u32) -> Result<Arc<Box<CmpFn>>> {
        let guard = self.db_cmp.read().map_err(|_| Error::Panic)?;
        guard.get(dbi as usize).cloned().ok_or(Error::BadDbi)
    }

    /// Clone the data comparison function for `dbi` (used with `DUPSORT`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadDbi`] if `dbi` is out of range.
    pub(crate) fn get_dcmp(&self, dbi: u32) -> Result<Arc<Box<CmpFn>>> {
        let guard = self.db_dcmp.read().map_err(|_| Error::Panic)?;
        guard.get(dbi as usize).cloned().ok_or(Error::BadDbi)
    }

    /// Return the database flags for `dbi`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadDbi`] if `dbi` is out of range.
    pub(crate) fn get_db_flags(&self, dbi: u32) -> Result<u16> {
        let guard = self.db_flags.read().map_err(|_| Error::Panic)?;
        guard.get(dbi as usize).copied().ok_or(Error::BadDbi)
    }

    /// Return the raw file descriptor of the data file.
    #[cfg(unix)]
    pub(crate) fn data_fd(&self) -> std::os::fd::RawFd {
        use std::os::fd::AsRawFd;
        self._data_file.as_raw_fd()
    }

    /// Return the raw mmap pointer (for WRITEMAP mode).
    ///
    /// # Safety
    ///
    /// The caller must hold the exclusive writer lock and ensure the mmap
    /// was created with write permissions (WRITE_MAP flag).
    pub(crate) fn mmap_mut_ptr(&self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    /// Return the mmap size in bytes.
    pub(crate) fn mmap_len(&self) -> usize {
        self.map_size
    }
}

// ---------------------------------------------------------------------------
// Environment implementation
// ---------------------------------------------------------------------------

impl Environment {
    /// Create a new [`EnvironmentBuilder`].
    pub fn builder() -> EnvironmentBuilder {
        EnvironmentBuilder::new()
    }

    /// Open the environment described by `config` at `path`.
    fn open_with_config(path: &Path, config: EnvironmentBuilder) -> Result<Self> {
        // ------------------------------------------------------------------
        // 1. Resolve data-file path
        // ------------------------------------------------------------------
        let data_path = if config.flags.contains(EnvFlags::NO_SUB_DIR) {
            path.to_path_buf()
        } else {
            if !path.is_dir() {
                fs::create_dir_all(path)?;
            }
            path.join("data.mdb")
        };

        // ------------------------------------------------------------------
        // 2. Open the data file
        // ------------------------------------------------------------------
        let read_only = config.flags.contains(EnvFlags::READ_ONLY);
        let data_file = if read_only {
            OpenOptions::new().read(true).open(&data_path)?
        } else {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&data_path)?
        };

        let file_len = data_file.metadata()?.len() as usize;

        // ------------------------------------------------------------------
        // 3. Determine page size from meta pages (or OS default for new DBs)
        // ------------------------------------------------------------------
        let page_size: usize;
        let meta0: Meta;
        let meta1: Meta;

        if file_len < PAGE_HEADER_SIZE + mem::size_of::<Meta>() {
            // File is empty or too small to contain a valid meta page.
            // For Phase 1 (read-only access to existing databases) we require
            // at least one valid meta page.
            if read_only {
                return Err(Error::Invalid);
            }
            // New database — use the OS page size.
            page_size = os_page_size();
            // Synthesize empty meta pages so the rest of the init path works.
            meta0 = Meta {
                magic: MDB_MAGIC,
                version: MDB_DATA_VERSION,
                address: 0,
                map_size: config.map_size as u64,
                dbs: [
                    empty_free_dbstat(page_size as u32),
                    empty_dbstat(page_size as u32),
                ],
                last_pgno: NUM_METAS as u64 - 1,
                txnid: 0,
            };
            meta1 = meta0;
            // Write initial meta pages to the file so the mmap has backing.
            write_initial_meta(&data_file, page_size, &meta0)?;
        } else {
            // Read the first meta page to discover page_size.
            // The page size is stored in `dbs[0].pad` (FREE_DBI's pad field
            // holds the page size in LMDB).
            let first_meta = read_meta_from_file(&data_file, 0, PAGE_HEADER_SIZE)?;
            if first_meta.magic != MDB_MAGIC {
                return Err(Error::Invalid);
            }
            if first_meta.version != MDB_DATA_VERSION {
                return Err(Error::VersionMismatch);
            }

            page_size = first_meta.dbs[FREE_DBI as usize].pad as usize;
            if page_size == 0 || !page_size.is_power_of_two() || page_size < 512 {
                return Err(Error::Invalid);
            }

            meta0 = first_meta;

            // Read the second meta page at offset `page_size`.
            if file_len >= page_size * 2 {
                let m1 = read_meta_from_file(&data_file, page_size, PAGE_HEADER_SIZE)?;
                if m1.magic != MDB_MAGIC {
                    return Err(Error::Invalid);
                }
                meta1 = m1;
            } else {
                // Only one meta page exists (very unusual). Duplicate meta0.
                meta1 = meta0;
            }
        }

        // ------------------------------------------------------------------
        // 4. Pick the current meta (higher txnid)
        // ------------------------------------------------------------------
        let current_meta = if meta1.txnid > meta0.txnid {
            &meta1
        } else {
            &meta0
        };

        // ------------------------------------------------------------------
        // 5. Compute derived values
        // ------------------------------------------------------------------
        let map_size = if config.map_size > 0 && config.map_size > file_len {
            config.map_size
        } else if current_meta.map_size as usize > file_len {
            current_meta.map_size as usize
        } else if file_len > 0 {
            file_len
        } else {
            config.map_size
        };

        let max_pgno = (map_size / page_size).saturating_sub(1) as u64;
        let max_key_size = max_key_for_page(page_size);
        // Node max: a node (key + data + header) that must fit within a
        // single page alongside the header and at least one pointer slot.
        let node_max = page_size - PAGE_HEADER_SIZE - 2 - 8; // conservative
        let max_free_per_page = (page_size - PAGE_HEADER_SIZE) / 8;

        let total_dbs = CORE_DBS + config.max_dbs;

        // ------------------------------------------------------------------
        // 6a. Open/create the lock file for cross-process writer locking
        // ------------------------------------------------------------------
        let lock_file = if !config.flags.contains(EnvFlags::NO_LOCK) && !read_only {
            let lock_path = if config.flags.contains(EnvFlags::NO_SUB_DIR) {
                format!("{}-lock", data_path.display())
            } else {
                path.join("lock.mdb").to_string_lossy().to_string()
            };
            Some(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&lock_path)?,
            )
        } else {
            None
        };

        // ------------------------------------------------------------------
        // 6. Memory-map the data file
        // ------------------------------------------------------------------
        let mmap = if read_only {
            MmapOptions::new()
                .len(map_size)
                .map_raw_read_only(&data_file)
                .map_err(Error::Io)?
        } else {
            MmapOptions::new()
                .len(map_size)
                .map_raw(&data_file)
                .map_err(Error::Io)?
        };

        // ------------------------------------------------------------------
        // 7. Initialise per-database comparison functions
        // ------------------------------------------------------------------
        let mut cmp_vec: Vec<Arc<Box<CmpFn>>> = Vec::with_capacity(total_dbs as usize);
        let mut dcmp_vec: Vec<Arc<Box<CmpFn>>> = Vec::with_capacity(total_dbs as usize);
        let mut names_vec: Vec<Option<String>> = Vec::with_capacity(total_dbs as usize);
        let mut flags_vec: Vec<u16> = Vec::with_capacity(total_dbs as usize);

        for i in 0..total_dbs {
            let db_flags = if (i as usize) < current_meta.dbs.len() {
                current_meta.dbs[i as usize].flags
            } else {
                0
            };
            cmp_vec.push(Arc::new(default_cmp(db_flags)));
            dcmp_vec.push(Arc::new(default_dcmp(db_flags)));
            names_vec.push(None);
            flags_vec.push(db_flags);
        }

        // ------------------------------------------------------------------
        // 8. Construct the environment
        // ------------------------------------------------------------------
        let env_path = if config.flags.contains(EnvFlags::NO_SUB_DIR) {
            data_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        } else {
            path.to_path_buf()
        };

        let inner = EnvironmentInner {
            mmap,
            path: env_path,
            _data_file: data_file,
            page_size,
            map_size,
            max_pgno,
            max_dbs: config.max_dbs,
            max_readers: config.max_readers,
            flags: config.flags,
            max_key_size,
            node_max,
            max_free_per_page,
            db_cmp: RwLock::new(cmp_vec),
            db_dcmp: RwLock::new(dcmp_vec),
            db_names: RwLock::new(names_vec),
            db_flags: RwLock::new(flags_vec),
            write_mutex: Mutex::new(()),
            reader_table: ReaderTable::new(config.max_readers),
            lock_file,
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Begin a read-only transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment is in a fatal-error state.
    pub fn begin_ro_txn(&self) -> Result<crate::txn::RoTransaction<'_>> {
        crate::txn::RoTransaction::new(&self.inner)
    }

    /// Begin a read-write transaction.
    ///
    /// Only one write transaction may be active at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment is read-only or in a fatal state.
    pub fn begin_rw_txn(&self) -> Result<crate::write::RwTransaction<'_>> {
        if self.inner.flags.contains(EnvFlags::READ_ONLY) {
            return Err(Error::Incompatible);
        }
        crate::write::RwTransaction::new(&self.inner)
    }

    /// Return statistics for the main database.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment is in a fatal-error state.
    pub fn stat(&self) -> Result<Stat> {
        let meta = self.inner.meta();
        let db = &meta.dbs[MAIN_DBI as usize];
        Ok(Stat {
            page_size: self.inner.page_size as u32,
            depth: db.depth as u32,
            branch_pages: db.branch_pages,
            leaf_pages: db.leaf_pages,
            overflow_pages: db.overflow_pages,
            entries: db.entries,
        })
    }

    /// Return environment information.
    pub fn info(&self) -> EnvInfo {
        let meta = self.inner.meta();
        EnvInfo {
            map_size: self.inner.map_size,
            last_pgno: meta.last_pgno,
            last_txnid: meta.txnid,
            max_readers: self.inner.max_readers,
            num_readers: self.inner.reader_table.active_count(),
        }
    }

    /// Path to the environment directory (or data file parent if
    /// `NO_SUB_DIR`).
    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Maximum key size in bytes.
    pub fn max_key_size(&self) -> usize {
        self.inner.max_key_size
    }

    /// Environment flags.
    pub fn flags(&self) -> EnvFlags {
        self.inner.flags
    }

    /// Close a named database handle.
    ///
    /// This releases the slot occupied by the named database so it can be
    /// reused by a future [`open_db`](crate::write::RwTransaction::open_db)
    /// call. Core databases (FREE_DBI and MAIN_DBI) cannot be closed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Incompatible`] if `dbi` refers to a core database.
    /// Returns [`Error::Panic`] if an internal lock is poisoned.
    pub fn close_db(&self, dbi: u32) -> Result<()> {
        if dbi < CORE_DBS {
            return Err(Error::Incompatible);
        }
        let mut db_names = self.inner.db_names.write().map_err(|_| Error::Panic)?;
        if let Some(slot) = db_names.get_mut(dbi as usize) {
            *slot = None;
        }
        Ok(())
    }

    /// Sync the data file to disk.
    ///
    /// If `force` is true, a synchronous flush is performed even when
    /// `NO_SYNC` is set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the sync fails.
    pub fn sync(&self, force: bool) -> Result<()> {
        if !force && self.inner.flags.contains(EnvFlags::NO_SYNC) {
            return Ok(());
        }
        let fd = self.inner.data_fd();
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

    /// Check for stale readers and return the number cleared.
    ///
    /// This is currently a stub that always returns 0. Full reader-table
    /// management will be implemented in a future phase.
    pub fn check_readers(&self) -> Result<u32> {
        Ok(0)
    }

    /// Create a plain (non-compacting) backup copy of the database.
    ///
    /// Opens a read transaction to get a consistent snapshot, then copies
    /// all pages to the destination file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if file operations fail.
    pub fn copy<P: AsRef<Path>>(&self, dest: P) -> Result<()> {
        use std::io::Write as IoWrite;

        let _txn = self.begin_ro_txn()?;
        let meta = self.inner.meta();
        let page_size = self.inner.page_size;
        let num_pages = (meta.last_pgno + 1) as usize;

        let mut file = fs::File::create(dest.as_ref())?;
        for pgno in 0..num_pages {
            let ptr = self.inner.get_page(pgno as u64)?;
            let page_data = unsafe { std::slice::from_raw_parts(ptr, page_size) };
            file.write_all(page_data)?;
        }
        file.sync_all()?;
        Ok(())
    }

    /// Access the shared inner state (for internal use by transactions).
    #[allow(dead_code)] // Used in later phases
    pub(crate) fn inner(&self) -> &Arc<EnvironmentInner> {
        &self.inner
    }

    /// Create a compacting copy of the database.
    ///
    /// Walks the B+ tree depth-first, assigns new sequential page numbers,
    /// and writes the pages out with updated child pointers. Free pages are
    /// eliminated, producing a minimal output file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if file operations fail.
    pub fn copy_compact<P: AsRef<Path>>(&self, dest: P) -> Result<()> {
        use std::io::{Seek, SeekFrom, Write as IoWrite};

        let _txn = self.begin_ro_txn()?;
        let meta = self.inner.meta();
        let page_size = self.inner.page_size;

        let mut file = fs::File::create(dest.as_ref())?;

        // Map old page numbers to new sequential page numbers.
        let mut pgno_map: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        let mut next_pgno = NUM_METAS as u64; // Skip meta pages 0 and 1.
        let mut pages_out: Vec<(u64, Vec<u8>)> = Vec::new(); // (new_pgno, page_data)

        let main_root = meta.dbs[MAIN_DBI as usize].root;
        let free_root = meta.dbs[FREE_DBI as usize].root;

        // Walk FREE_DBI tree (we include it for correctness, but it will be
        // empty in the compacted output since all pages are renumbered).
        // Actually, for compacting copy, we skip the free list entirely
        // since the compacted file has no free pages.

        // Walk MAIN_DBI tree depth-first.
        if main_root != P_INVALID {
            self.walk_tree_compact(
                main_root,
                page_size,
                &mut pgno_map,
                &mut next_pgno,
                &mut pages_out,
            )?;
        }

        // Also walk FREE_DBI so we don't lose its structure if non-empty.
        // For a compacting copy, the free list should be empty.
        // We skip it intentionally.
        let _ = free_root;

        // Fix up page pointers to use new page numbers.
        // We collect fixups first, then apply them, to avoid borrow conflicts.
        for (_, page_data) in &mut pages_out {
            let mut fixups: Vec<(usize, Vec<u8>)> = Vec::new();
            {
                let page = Page::from_raw(page_data);
                if page.is_branch() {
                    let nkeys = page.num_keys();
                    for i in 0..nkeys {
                        let node_offset = page.ptr_at(i) as usize;
                        let lo = u16::from_le_bytes([
                            page_data[node_offset],
                            page_data[node_offset + 1],
                        ]);
                        let hi = u16::from_le_bytes([
                            page_data[node_offset + 2],
                            page_data[node_offset + 3],
                        ]);
                        let flags_raw = u16::from_le_bytes([
                            page_data[node_offset + 4],
                            page_data[node_offset + 5],
                        ]);
                        let old_child =
                            u64::from(lo) | (u64::from(hi) << 16) | (u64::from(flags_raw) << 32);

                        if let Some(&new_child) = pgno_map.get(&old_child) {
                            let new_lo = (new_child & 0xFFFF) as u16;
                            let new_hi = ((new_child >> 16) & 0xFFFF) as u16;
                            let new_flags = ((new_child >> 32) & 0xFFFF) as u16;
                            let mut bytes = Vec::with_capacity(6);
                            bytes.extend_from_slice(&new_lo.to_le_bytes());
                            bytes.extend_from_slice(&new_hi.to_le_bytes());
                            bytes.extend_from_slice(&new_flags.to_le_bytes());
                            fixups.push((node_offset, bytes));
                        }
                    }
                } else if page.is_leaf() && !page.is_leaf2() {
                    let nkeys = page.num_keys();
                    for i in 0..nkeys {
                        let node_offset = page.ptr_at(i) as usize;
                        let node_flags_raw = u16::from_le_bytes([
                            page_data[node_offset + 4],
                            page_data[node_offset + 5],
                        ]);
                        let is_bigdata = node_flags_raw & 0x01 != 0;
                        let is_subdata = node_flags_raw & 0x02 != 0;
                        let key_size = u16::from_le_bytes([
                            page_data[node_offset + 6],
                            page_data[node_offset + 7],
                        ]) as usize;
                        let data_offset = node_offset + 8 + key_size;

                        if is_bigdata && data_offset + 8 <= page_data.len() {
                            let mut old_pgno_bytes = [0u8; 8];
                            old_pgno_bytes
                                .copy_from_slice(&page_data[data_offset..data_offset + 8]);
                            let old_ovf_pgno = u64::from_le_bytes(old_pgno_bytes);
                            if let Some(&new_ovf_pgno) = pgno_map.get(&old_ovf_pgno) {
                                fixups.push((data_offset, new_ovf_pgno.to_le_bytes().to_vec()));
                            }
                        }

                        if is_subdata
                            && data_offset + std::mem::size_of::<crate::types::DbStat>()
                                <= page_data.len()
                        {
                            // DbStat.root is at offset 40 within the struct.
                            let root_offset = data_offset + 40;
                            if root_offset + 8 <= page_data.len() {
                                let mut old_root_bytes = [0u8; 8];
                                old_root_bytes
                                    .copy_from_slice(&page_data[root_offset..root_offset + 8]);
                                let old_root = u64::from_le_bytes(old_root_bytes);
                                if old_root != P_INVALID {
                                    if let Some(&new_root) = pgno_map.get(&old_root) {
                                        fixups.push((root_offset, new_root.to_le_bytes().to_vec()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Apply collected fixups.
            for (offset, bytes) in fixups {
                page_data[offset..offset + bytes.len()].copy_from_slice(&bytes);
            }
        }

        // Write meta pages.
        let new_main_root = if main_root != P_INVALID {
            pgno_map.get(&main_root).copied().unwrap_or(P_INVALID)
        } else {
            P_INVALID
        };

        // The output file size is next_pgno * page_size.
        let output_size = next_pgno as usize * page_size;
        let compact_map_size = if output_size > 0 {
            output_size
        } else {
            NUM_METAS * page_size
        };

        let new_meta = Meta {
            magic: MDB_MAGIC,
            version: MDB_DATA_VERSION,
            address: 0,
            map_size: compact_map_size as u64,
            dbs: [
                empty_free_dbstat(page_size as u32),
                crate::types::DbStat {
                    root: new_main_root,
                    ..meta.dbs[MAIN_DBI as usize]
                },
            ],
            last_pgno: if next_pgno > 0 {
                next_pgno - 1
            } else {
                NUM_METAS as u64 - 1
            },
            txnid: meta.txnid,
        };

        write_initial_meta(&file, page_size, &new_meta)?;

        // Ensure the file is large enough for all pages.
        file.set_len(compact_map_size as u64)?;

        // Write all renumbered pages.
        for (new_pgno, page_data) in &pages_out {
            let offset = *new_pgno * page_size as u64;
            file.seek(SeekFrom::Start(offset))?;

            // Update the page number in the page header.
            let mut data = page_data.clone();
            data[0..8].copy_from_slice(&new_pgno.to_le_bytes());
            // Clear DIRTY flag.
            let flags_raw = u16::from_le_bytes([data[10], data[11]]);
            let clean_flags = flags_raw & !PageFlags::DIRTY.bits();
            data[10..12].copy_from_slice(&clean_flags.to_le_bytes());

            file.write_all(&data)?;
        }

        file.sync_all()?;
        Ok(())
    }

    /// Walk a B+ tree depth-first, collecting all reachable pages and assigning
    /// new sequential page numbers.
    fn walk_tree_compact(
        &self,
        pgno: u64,
        page_size: usize,
        pgno_map: &mut std::collections::HashMap<u64, u64>,
        next_pgno: &mut u64,
        pages_out: &mut Vec<(u64, Vec<u8>)>,
    ) -> Result<()> {
        if pgno == P_INVALID || pgno_map.contains_key(&pgno) {
            return Ok(());
        }

        let ptr = self.inner.get_page(pgno)?;
        let page_data = unsafe { std::slice::from_raw_parts(ptr, page_size) };
        let page = Page::from_raw(page_data);

        if page.is_overflow() {
            // Overflow page: copy all contiguous pages.
            let num_pages = page.overflow_pages() as usize;
            let total_size = num_pages * page_size;
            let all_data = unsafe { std::slice::from_raw_parts(ptr, total_size) };

            let new_pgno = *next_pgno;
            pgno_map.insert(pgno, new_pgno);
            *next_pgno += num_pages as u64;

            // Map intermediate pages too.
            for i in 1..num_pages as u64 {
                pgno_map.insert(pgno + i, new_pgno + i);
            }

            pages_out.push((new_pgno, all_data.to_vec()));
            return Ok(());
        }

        if page.is_branch() {
            // Recurse into children first (depth-first).
            let nkeys = page.num_keys();
            for i in 0..nkeys {
                let child_pgno = page.node(i).child_pgno();
                self.walk_tree_compact(child_pgno, page_size, pgno_map, next_pgno, pages_out)?;
            }
        } else if page.is_leaf() && !page.is_leaf2() {
            // Check for overflow pages and sub-database roots.
            let nkeys = page.num_keys();
            for i in 0..nkeys {
                let node = page.node(i);
                if node.is_bigdata() {
                    let ovf_pgno = node.overflow_pgno();
                    self.walk_tree_compact(ovf_pgno, page_size, pgno_map, next_pgno, pages_out)?;
                }
                if node.is_subdata() {
                    let sub_db = node.sub_db();
                    if sub_db.root != P_INVALID {
                        self.walk_tree_compact(
                            sub_db.root,
                            page_size,
                            pgno_map,
                            next_pgno,
                            pages_out,
                        )?;
                    }
                }
            }
        }

        // Assign new page number for this page.
        let new_pgno = *next_pgno;
        pgno_map.insert(pgno, new_pgno);
        *next_pgno += 1;
        pages_out.push((new_pgno, page_data.to_vec()));

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers (private)
// ---------------------------------------------------------------------------

/// Read a [`Meta`] structure from `file` at the given byte offset.
///
/// `hdr_size` is the page-header size to skip before the meta payload.
fn read_meta_from_file(file: &File, page_offset: usize, hdr_size: usize) -> Result<Meta> {
    use std::io::{Read, Seek, SeekFrom};

    let meta_size = mem::size_of::<Meta>();
    let mut buf = vec![0u8; meta_size];

    let seek_pos = page_offset + hdr_size;
    let mut reader = std::io::BufReader::new(file);
    reader.seek(SeekFrom::Start(seek_pos as u64))?;
    reader.read_exact(&mut buf)?;

    // SAFETY: `Meta` is `repr(C)` and we read exactly `size_of::<Meta>()` bytes.
    // `read_unaligned` handles potential alignment issues.
    let meta = unsafe { std::ptr::read_unaligned(buf.as_ptr().cast::<Meta>()) };
    Ok(meta)
}

/// Return the OS page size.
fn os_page_size() -> usize {
    // SAFETY: sysconf is always safe to call with _SC_PAGESIZE.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size <= 0 { 4096 } else { size as usize }
}

/// Compute the maximum key size for a given page size.
///
/// LMDB formula: `((page_size - PAGE_HEADER_SIZE) / (MIN_KEYS * 3)) - 8`
/// where `MIN_KEYS = 2` and `8` is `NODE_HEADER_SIZE`.
fn max_key_for_page(page_size: usize) -> usize {
    let usable = page_size - PAGE_HEADER_SIZE;
    // Each key needs: 2 bytes pointer + NODE_HEADER_SIZE + key bytes.
    // With MIN_KEYS=2, each branch page must hold at least 2 nodes.
    // node_space = usable / (2 * 3) (LMDB uses a factor of 3 for safety)
    let node_space = usable / 6;
    let max = node_space.saturating_sub(8);
    max.min(MAX_KEY_SIZE)
}

/// Write two initial meta pages to a new data file.
fn write_initial_meta(file: &File, page_size: usize, meta: &Meta) -> Result<()> {
    use std::io::{Seek, SeekFrom};

    let meta_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(meta).cast::<u8>(),
            mem::size_of::<Meta>(),
        )
    };

    let mut writer = std::io::BufWriter::new(file);

    // Page 0: page header + meta
    let mut page0 = vec![0u8; page_size];
    // Set page flags to META (offset 10-11)
    page0[10] = 0x08; // PageFlags::META
    // Copy meta payload at offset PAGE_HEADER_SIZE
    page0[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + meta_bytes.len()].copy_from_slice(meta_bytes);

    // Page 1: same content but pgno = 1
    let mut page1 = page0.clone();
    // Set pgno = 1 (little-endian u64 at offset 0)
    page1[0] = 1;

    writer.seek(SeekFrom::Start(0))?;
    writer.write_all(&page0)?;
    writer.write_all(&page1)?;
    writer.flush()?;

    Ok(())
}

/// Create an empty [`DbStat`] with the given page size stored in `pad`.
fn empty_dbstat(page_size: u32) -> DbStat {
    DbStat {
        pad: page_size,
        flags: 0,
        depth: 0,
        branch_pages: 0,
        leaf_pages: 0,
        overflow_pages: 0,
        entries: 0,
        root: P_INVALID,
    }
}

/// Create an empty [`DbStat`] for FREE_DBI with `INTEGER_KEY` flag set.
///
/// The free-page database uses native-byte-order unsigned integer keys
/// (transaction IDs), matching LMDB's `MDB_INTEGERKEY` convention.
fn empty_free_dbstat(page_size: u32) -> DbStat {
    DbStat {
        pad: page_size,
        flags: 0x08, // INTEGER_KEY
        depth: 0,
        branch_pages: 0,
        leaf_pages: 0,
        overflow_pages: 0,
        entries: 0,
        root: P_INVALID,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_create_default_builder() {
        let b = EnvironmentBuilder::new();
        assert_eq!(b.map_size, DEFAULT_MAPSIZE);
        assert_eq!(b.max_readers, DEFAULT_READERS);
        assert_eq!(b.max_dbs, 0);
        assert_eq!(b.flags, EnvFlags::empty());
    }

    #[test]
    fn test_should_configure_builder() {
        let b = EnvironmentBuilder::new()
            .map_size(1024 * 1024)
            .max_readers(64)
            .max_dbs(8)
            .flags(EnvFlags::READ_ONLY | EnvFlags::NO_SUB_DIR);

        assert_eq!(b.map_size, 1024 * 1024);
        assert_eq!(b.max_readers, 64);
        assert_eq!(b.max_dbs, 8);
        assert!(b.flags.contains(EnvFlags::READ_ONLY));
        assert!(b.flags.contains(EnvFlags::NO_SUB_DIR));
    }

    #[test]
    fn test_should_compute_max_key_size() {
        // For a 4096-byte page the max key should be <= MAX_KEY_SIZE.
        let max = max_key_for_page(4096);
        assert!(max > 0);
        assert!(max <= MAX_KEY_SIZE);
    }

    #[test]
    fn test_should_create_empty_dbstat() {
        let db = empty_dbstat(4096);
        assert_eq!(db.pad, 4096);
        assert_eq!(db.root, P_INVALID);
        assert_eq!(db.entries, 0);
    }

    #[test]
    fn test_should_reject_nonexistent_readonly_file() {
        let result = EnvironmentBuilder::new()
            .flags(EnvFlags::READ_ONLY | EnvFlags::NO_SUB_DIR)
            .open("/tmp/nonexistent_lmdb_file_for_test_42.mdb");
        assert!(result.is_err());
    }

    #[test]
    fn test_should_open_new_readwrite_env() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path());
        // A new (empty) environment should open successfully.
        assert!(env.is_ok(), "failed to open new env: {:?}", env.err());
        let env = env.expect("env");
        assert_eq!(env.path(), dir.path());
        assert!(env.max_key_size() > 0);
    }

    #[test]
    fn test_should_return_stat_and_info() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");
        let stat = env.stat().expect("stat");
        assert!(stat.page_size > 0);
        let info = env.info();
        assert_eq!(info.map_size, 1024 * 1024);
    }

    // -------------------------------------------------------------------
    // Feature: Compacting copy
    // -------------------------------------------------------------------

    #[test]
    fn test_should_compact_copy_empty_db() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(1024 * 1024)
            .open(dir.path())
            .expect("open");

        let dest_dir = tempfile::tempdir().expect("dest_dir");
        let dest = dest_dir.path().join("compact.mdb");
        env.copy_compact(&dest).expect("copy_compact");
        assert!(dest.exists(), "compacted file should exist");

        // Drop the source env to release the lock file.
        drop(env);

        // Open the compacted copy.
        let file_len = std::fs::metadata(&dest).expect("meta").len() as usize;
        let env2 = Environment::builder()
            .map_size(file_len)
            .flags(EnvFlags::NO_SUB_DIR | EnvFlags::READ_ONLY)
            .open(&dest)
            .expect("open compact");
        let stat = env2.stat().expect("stat");
        assert_eq!(stat.entries, 0);
    }

    #[test]
    fn test_should_compact_copy_with_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(2 * 1024 * 1024)
            .open(dir.path())
            .expect("open");

        // Insert data.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..100 {
                let key = format!("key_{i:04}");
                let val = format!("val_{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    crate::types::WriteFlags::empty(),
                )
                .expect("put");
            }
            txn.commit().expect("commit");
        }

        let dest = dir.path().join("compact.mdb");
        env.copy_compact(&dest).expect("copy_compact");

        // Open the compacted copy. Use file size as map_size to avoid SIGBUS.
        let file_len = std::fs::metadata(&dest).expect("meta").len() as usize;
        let env2 = Environment::builder()
            .map_size(file_len)
            .flags(EnvFlags::NO_SUB_DIR | EnvFlags::READ_ONLY)
            .open(&dest)
            .expect("open compact");

        let txn = env2.begin_ro_txn().expect("begin_ro_txn");
        for i in 0..100 {
            let key = format!("key_{i:04}");
            let expected = format!("val_{i:04}");
            let val = txn.get(MAIN_DBI as u32, key.as_bytes()).expect("get");
            assert_eq!(val, expected.as_bytes(), "mismatch for {key}");
        }
    }

    #[test]
    fn test_should_compact_copy_smaller_than_plain_copy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = Environment::builder()
            .map_size(2 * 1024 * 1024)
            .open(dir.path())
            .expect("open");

        // Insert and then delete half the data to create free pages.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..200 {
                let key = format!("key_{i:04}");
                let val = format!("val_{i:04}");
                txn.put(
                    MAIN_DBI as u32,
                    key.as_bytes(),
                    val.as_bytes(),
                    crate::types::WriteFlags::empty(),
                )
                .expect("put");
            }
            txn.commit().expect("commit");
        }
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..100 {
                let key = format!("key_{i:04}");
                txn.del(MAIN_DBI as u32, key.as_bytes(), None).expect("del");
            }
            txn.commit().expect("commit");
        }

        let plain_dest = dir.path().join("plain.mdb");
        env.copy(&plain_dest).expect("copy");

        let compact_dest = dir.path().join("compact.mdb");
        env.copy_compact(&compact_dest).expect("copy_compact");

        let plain_size = std::fs::metadata(&plain_dest).expect("meta").len();
        let compact_size = std::fs::metadata(&compact_dest).expect("meta").len();

        assert!(
            compact_size <= plain_size,
            "compact ({compact_size}) should be <= plain ({plain_size})",
        );
    }
}
