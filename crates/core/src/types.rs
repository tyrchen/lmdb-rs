//! Core types for the LMDB on-disk format.
//!
//! This module defines the fundamental constants, bitflags, and `repr(C)`
//! structures that mirror the LMDB binary format. All multi-byte fields are
//! stored in little-endian order on disk.

use bitflags::bitflags;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of a page header in bytes (on 64-bit systems).
pub const PAGE_HEADER_SIZE: usize = 16;

/// Size of a node header in bytes.
pub const NODE_HEADER_SIZE: usize = 8;

/// Sentinel value for an invalid / unset page number.
pub const P_INVALID: u64 = u64::MAX;

/// LMDB magic number stored in every meta page.
pub const MDB_MAGIC: u32 = 0xBEEF_C0DE;

/// LMDB data format version.
pub const MDB_DATA_VERSION: u32 = 1;

/// Handle for the free-page database.
pub const FREE_DBI: u32 = 0;

/// Handle for the main (default) database.
pub const MAIN_DBI: u32 = 1;

/// Number of built-in databases (free + main).
pub const CORE_DBS: u32 = 2;

/// Number of meta pages.
pub const NUM_METAS: u32 = 2;

/// Maximum cursor stack depth (B+ tree depth).
pub const CURSOR_STACK: usize = 32;

/// Default memory map size (1 MB).
pub const DEFAULT_MAPSIZE: usize = 1_048_576;

/// Default number of reader slots.
pub const DEFAULT_READERS: u32 = 126;

/// Maximum IDL size for database operations.
pub const IDL_DB_SIZE: usize = 1 << 16;

/// Maximum IDL size for "unlimited" operations (dirty list).
pub const IDL_UM_SIZE: usize = 1 << 17;

/// Maximum index in an unlimited IDL.
pub const IDL_UM_MAX: usize = IDL_UM_SIZE - 1;

/// Minimum number of keys per page.
pub const MIN_KEYS: usize = 2;

/// Page fill threshold in tenths of a percent (25.0%).
pub const FILL_THRESHOLD: usize = 250;

/// Default maximum key size in bytes.
pub const MAX_KEY_SIZE: usize = 511;

/// Maximum pages per `writev()` call.
pub const COMMIT_PAGES: usize = 64;

// ---------------------------------------------------------------------------
// Bitflags
// ---------------------------------------------------------------------------

bitflags! {
    /// Page-level flags stored in the page header (bytes 10-11).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PageFlags: u16 {
        /// Internal (branch) B+ tree node.
        const BRANCH   = 0x01;
        /// Leaf B+ tree node.
        const LEAF     = 0x02;
        /// Overflow page for large values.
        const OVERFLOW = 0x04;
        /// Meta page.
        const META     = 0x08;
        /// Page has been modified (COW copy).
        const DIRTY    = 0x10;
        /// Compact leaf for `DUPFIXED` databases.
        const LEAF2    = 0x20;
        /// Inline sub-page for duplicate data.
        const SUBPAGE  = 0x40;
        /// Freed in the same transaction, eligible for reuse.
        const LOOSE    = 0x4000;
        /// Do not spill this page during page spilling.
        const KEEP     = 0x8000;
    }
}

bitflags! {
    /// Node-level flags stored in the node header (bytes 4-5).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeFlags: u16 {
        /// Data is stored on an overflow page.
        const BIGDATA  = 0x01;
        /// Data is a sub-database ([`DbStat`]).
        const SUBDATA  = 0x02;
        /// Node has duplicate data.
        const DUPDATA  = 0x04;
    }
}

// ---------------------------------------------------------------------------
// repr(C) on-disk structures
// ---------------------------------------------------------------------------

/// Database statistics / record (48 bytes), corresponding to LMDB's
/// `MDB_db`.
///
/// Each named database (plus `FREE_DBI` and `MAIN_DBI`) is described by one
/// of these records stored inside the meta page or as node data.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DbStat {
    /// Key size for `LEAF2` pages; page size for `FREE_DBI`.
    pub pad: u32,
    /// Database flags.
    pub flags: u16,
    /// B+ tree depth.
    pub depth: u16,
    /// Number of branch pages.
    pub branch_pages: u64,
    /// Number of leaf pages.
    pub leaf_pages: u64,
    /// Number of overflow pages.
    pub overflow_pages: u64,
    /// Total number of data entries.
    pub entries: u64,
    /// Root page number (`P_INVALID` if empty).
    pub root: u64,
}

/// Meta page payload, corresponding to LMDB's `MDB_meta`.
///
/// Located at offset `PAGE_HEADER_SIZE` within meta pages (pages 0 and 1).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Meta {
    /// Magic number — must be [`MDB_MAGIC`].
    pub magic: u32,
    /// Data format version.
    pub version: u32,
    /// Fixed mapping address (0 = dynamic).
    pub address: u64,
    /// Memory-map region size.
    pub map_size: u64,
    /// Database records for `FREE_DBI` and `MAIN_DBI`.
    pub dbs: [DbStat; 2],
    /// Last used page number.
    pub last_pgno: u64,
    /// Transaction ID that committed this meta page.
    pub txnid: u64,
}

bitflags! {
    /// Database flags for `open_db`.
    ///
    /// The lower 16 bits match the on-disk `DbStat.flags` (u16).
    /// `CREATE` is an open-time-only flag and is not stored on disk.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DatabaseFlags: u32 {
        /// Compare keys in reverse byte order.
        const REVERSE_KEY  = 0x02;
        /// Allow duplicate keys with sorted data items.
        const DUP_SORT     = 0x04;
        /// Keys are native-byte-order unsigned integers.
        const INTEGER_KEY  = 0x08;
        /// With `DUP_SORT`: all dup data items are fixed-size.
        const DUP_FIXED    = 0x10;
        /// With `DUP_SORT`: dup data items are integers.
        const INTEGER_DUP  = 0x20;
        /// With `DUP_SORT`: compare dup data in reverse byte order.
        const REVERSE_DUP  = 0x40;
        /// Create the database if it doesn't exist (open-time only).
        const CREATE       = 0x40000;
    }
}

bitflags! {
    /// Flags for write operations (`put` / `cursor_put`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct WriteFlags: u32 {
        /// Don't write if key already exists.
        const NO_OVERWRITE = 0x10;
        /// For `DUP_SORT`: don't write if key+data pair exists.
        const NO_DUP_DATA  = 0x20;
        /// Overwrite current key/data pair (cursor put).
        const CURRENT      = 0x40;
        /// Reserve space, don't copy data.
        const RESERVE      = 0x10000;
        /// Append: keys must be in order.
        const APPEND       = 0x20000;
        /// Append dup data items in order.
        const APPEND_DUP   = 0x40000;
        /// Store multiple contiguous data items (DUPFIXED only).
        const MULTIPLE     = 0x80000;
    }
}

bitflags! {
    /// Environment flags for `Environment::open`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EnvFlags: u32 {
        /// Path is the data file, not a directory.
        const NO_SUB_DIR    = 0x4000;
        /// Don't fsync after commit.
        const NO_SYNC       = 0x10000;
        /// Read-only mode.
        const READ_ONLY     = 0x20000;
        /// Don't fsync meta page after commit.
        const NO_META_SYNC  = 0x40000;
        /// Use writable mmap.
        const WRITE_MAP     = 0x80000;
        /// Use async msync with `WRITE_MAP`.
        const MAP_ASYNC     = 0x100000;
        /// Don't use thread-local reader slots.
        const NO_TLS        = 0x200000;
        /// Caller manages all locking.
        const NO_LOCK       = 0x400000;
        /// Disable OS readahead.
        const NO_READAHEAD  = 0x800000;
        /// Don't zero malloc'd pages before writing.
        const NO_MEM_INIT   = 0x1000000;
    }
}

/// Cursor positioning operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorOp {
    /// Position at first key/data item.
    First,
    /// Position at first data item of current key (DUPSORT).
    FirstDup,
    /// Position at exact key/data pair (DUPSORT).
    GetBoth,
    /// Position at key, nearest data >= given (DUPSORT).
    GetBothRange,
    /// Return key/data at current position.
    GetCurrent,
    /// Return up to a page of dup data (DUPFIXED).
    GetMultiple,
    /// Position at last key/data item.
    Last,
    /// Position at last data item of current key (DUPSORT).
    LastDup,
    /// Position at next data item.
    Next,
    /// Position at next dup data item (DUPSORT).
    NextDup,
    /// Return next page of dup data (DUPFIXED).
    NextMultiple,
    /// Position at first data item of next key.
    NextNoDup,
    /// Position at previous data item.
    Prev,
    /// Position at previous dup data item (DUPSORT).
    PrevDup,
    /// Position at last data item of previous key.
    PrevNoDup,
    /// Position at specified key.
    Set,
    /// Position at specified key, return key + data.
    SetKey,
    /// Position at first key >= specified key.
    SetRange,
    /// Previous page of dup data (DUPFIXED).
    PrevMultiple,
}

// ---------------------------------------------------------------------------
// Public-facing result types
// ---------------------------------------------------------------------------

/// Database statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stat {
    /// Size of a database page in bytes.
    pub page_size: u32,
    /// B+ tree depth.
    pub depth: u32,
    /// Number of branch pages.
    pub branch_pages: u64,
    /// Number of leaf pages.
    pub leaf_pages: u64,
    /// Number of overflow pages.
    pub overflow_pages: u64,
    /// Total number of data items.
    pub entries: u64,
}

/// Environment information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvInfo {
    /// Size of the data memory map.
    pub map_size: usize,
    /// Last used page number.
    pub last_pgno: u64,
    /// Last committed transaction ID.
    pub last_txnid: u64,
    /// Maximum reader slots.
    pub max_readers: u32,
    /// Reader slots currently in use.
    pub num_readers: u32,
}

// ---------------------------------------------------------------------------
// Default implementations
// ---------------------------------------------------------------------------

impl Default for DbStat {
    fn default() -> Self {
        Self {
            pad: 0,
            flags: 0,
            depth: 0,
            branch_pages: 0,
            leaf_pages: 0,
            overflow_pages: 0,
            entries: 0,
            root: P_INVALID,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

impl Meta {
    /// Check whether this meta page has valid magic and version.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.magic == MDB_MAGIC && self.version == MDB_DATA_VERSION
    }

    /// Return the page size stored in this meta page.
    ///
    /// The page size is aliased to `dbs[FREE_DBI].pad`.
    #[must_use]
    pub fn page_size(&self) -> u32 {
        self.dbs[FREE_DBI as usize].pad
    }
}

impl DbStat {
    /// Return `true` if this database is empty (root is `P_INVALID`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root == P_INVALID
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::mem;

    use super::*;

    #[test]
    fn test_db_stat_size() {
        assert_eq!(mem::size_of::<DbStat>(), 48);
    }

    #[test]
    fn test_meta_size() {
        assert_eq!(mem::size_of::<Meta>(), 136);
    }

    #[test]
    fn test_db_stat_default() {
        let db = DbStat::default();
        assert_eq!(db.root, P_INVALID);
        assert!(db.is_empty());
    }

    #[test]
    fn test_meta_valid() {
        let mut meta = Meta {
            magic: MDB_MAGIC,
            version: MDB_DATA_VERSION,
            address: 0,
            map_size: 0,
            dbs: [DbStat::default(); 2],
            last_pgno: 0,
            txnid: 0,
        };
        assert!(meta.is_valid());

        meta.magic = 0;
        assert!(!meta.is_valid());
    }

    #[test]
    fn test_page_flags() {
        let flags = PageFlags::BRANCH | PageFlags::DIRTY;
        assert!(flags.contains(PageFlags::BRANCH));
        assert!(!flags.contains(PageFlags::LEAF));
    }

    #[test]
    fn test_constants() {
        assert_eq!(PAGE_HEADER_SIZE, 16);
        assert_eq!(NODE_HEADER_SIZE, 8);
        assert_eq!(FREE_DBI, 0);
        assert_eq!(MAIN_DBI, 1);
        assert_eq!(CORE_DBS, 2);
        assert_eq!(IDL_UM_SIZE, 131_072);
        assert_eq!(IDL_UM_MAX, 131_071);
    }
}
