//! Page types and zero-copy page parsing.
//!
//! This module provides zero-copy access to LMDB pages backed by `&[u8]` slices
//! (typically from memory-mapped files). All accessors read directly from the
//! underlying byte slice using little-endian byte order, avoiding allocations.
//!
//! # Page Layout (16-byte header)
//!
//! ```text
//! Offset  Size  Field
//! 0       8     pgno (page number)
//! 8       2     pad (key size for LEAF2)
//! 10      2     flags (PageFlags)
//! 12      2     lower (end of pointer array)
//! 14      2     upper (start of node data)
//! ```
//!
//! For overflow pages, bytes 12-15 are reinterpreted as a single `u32`
//! representing the number of overflow pages.

use std::mem;

use crate::types::{DbStat, Meta, NODE_HEADER_SIZE, NodeFlags, PAGE_HEADER_SIZE, PageFlags};

/// Round up to the nearest even number (LMDB's EVEN macro).
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::page::even;
/// assert_eq!(even(0), 0);
/// assert_eq!(even(1), 2);
/// assert_eq!(even(2), 2);
/// assert_eq!(even(3), 4);
/// ```
#[inline]
#[must_use]
pub fn even(n: usize) -> usize {
    (n + 1) & !1
}

/// A read-only view of a database page backed by a byte slice.
///
/// `Page` provides zero-copy access to page headers, node pointers, and
/// inline data. The underlying slice must be at least `PAGE_HEADER_SIZE`
/// bytes long.
#[derive(Clone, Copy, Debug)]
pub struct Page<'a> {
    data: &'a [u8],
}

impl<'a> Page<'a> {
    /// Wrap a raw byte slice as a page.
    ///
    /// The caller must ensure `data` is at least `PAGE_HEADER_SIZE` bytes.
    #[inline]
    #[must_use]
    pub fn from_raw(data: &'a [u8]) -> Self {
        debug_assert!(
            data.len() >= PAGE_HEADER_SIZE,
            "page data too short: {} < {PAGE_HEADER_SIZE}",
            data.len(),
        );
        Self { data }
    }

    /// Read the page number (bytes 0..8, little-endian u64).
    #[inline]
    #[must_use]
    pub fn pgno(&self) -> u64 {
        let bytes: [u8; 8] = self.read_array(0);
        u64::from_le_bytes(bytes)
    }

    /// Read the pad / key-size field (bytes 8..10, little-endian u16).
    ///
    /// For `LEAF2` pages this stores the fixed key size.
    #[inline]
    #[must_use]
    pub fn pad(&self) -> u16 {
        let bytes: [u8; 2] = self.read_array(8);
        u16::from_le_bytes(bytes)
    }

    /// Read the page flags (bytes 10..12, little-endian u16).
    #[inline]
    #[must_use]
    pub fn flags(&self) -> PageFlags {
        let bytes: [u8; 2] = self.read_array(10);
        PageFlags::from_bits_truncate(u16::from_le_bytes(bytes))
    }

    /// Read `lower` -- end of the pointer array (bytes 12..14).
    #[inline]
    #[must_use]
    pub fn lower(&self) -> u16 {
        let bytes: [u8; 2] = self.read_array(12);
        u16::from_le_bytes(bytes)
    }

    /// Read `upper` -- start of node data area (bytes 14..16).
    #[inline]
    #[must_use]
    pub fn upper(&self) -> u16 {
        let bytes: [u8; 2] = self.read_array(14);
        u16::from_le_bytes(bytes)
    }

    /// For overflow pages, reinterpret `lower`+`upper` as a single `u32`
    /// representing the number of overflow pages.
    #[inline]
    #[must_use]
    pub fn overflow_pages(&self) -> u32 {
        let bytes: [u8; 4] = self.read_array(12);
        u32::from_le_bytes(bytes)
    }

    /// Returns `true` if this is a branch (internal B+ tree) page.
    #[inline]
    #[must_use]
    pub fn is_branch(&self) -> bool {
        self.flags().contains(PageFlags::BRANCH)
    }

    /// Returns `true` if this is a leaf page.
    #[inline]
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.flags().contains(PageFlags::LEAF)
    }

    /// Returns `true` if this is a compact `LEAF2` page (fixed-size keys, no
    /// data).
    #[inline]
    #[must_use]
    pub fn is_leaf2(&self) -> bool {
        self.flags().contains(PageFlags::LEAF2)
    }

    /// Returns `true` if this is an overflow page.
    #[inline]
    #[must_use]
    pub fn is_overflow(&self) -> bool {
        self.flags().contains(PageFlags::OVERFLOW)
    }

    /// Returns `true` if this is an inline sub-page.
    #[inline]
    #[must_use]
    pub fn is_subpage(&self) -> bool {
        self.flags().contains(PageFlags::SUBPAGE)
    }

    /// Number of keys (node pointers) stored in this page.
    ///
    /// Computed from the pointer array: each entry is 2 bytes, starting
    /// after the page header.
    #[inline]
    #[must_use]
    pub fn num_keys(&self) -> usize {
        let lower = usize::from(self.lower());
        if lower < PAGE_HEADER_SIZE {
            return 0;
        }
        (lower - PAGE_HEADER_SIZE) / 2
    }

    /// Free space between the end of the pointer array and the start of node
    /// data.
    #[inline]
    #[must_use]
    pub fn free_space(&self) -> usize {
        let upper = usize::from(self.upper());
        let lower = usize::from(self.lower());
        upper.saturating_sub(lower)
    }

    /// Used space on this page (total minus header minus free gap).
    #[inline]
    #[must_use]
    pub fn used_space(&self) -> usize {
        self.data
            .len()
            .saturating_sub(PAGE_HEADER_SIZE)
            .saturating_sub(self.free_space())
    }

    /// Read the `idx`-th entry in the node-pointer array.
    ///
    /// Pointers are stored as little-endian `u16` values starting at offset
    /// `PAGE_HEADER_SIZE`.
    #[inline]
    #[must_use]
    pub fn ptr_at(&self, idx: usize) -> u16 {
        let offset = PAGE_HEADER_SIZE + idx * 2;
        let bytes: [u8; 2] = self.read_array(offset);
        u16::from_le_bytes(bytes)
    }

    /// Parse the node at pointer-array index `idx`.
    ///
    /// For non-`LEAF2` pages the node starts at the offset stored in the
    /// pointer array.
    #[inline]
    #[must_use]
    pub fn node(&self, idx: usize) -> Node<'a> {
        let offset = usize::from(self.ptr_at(idx));
        Node::from_raw(&self.data[offset..])
    }

    /// For `LEAF2` pages, return the fixed-size key at position `idx`.
    ///
    /// Keys are packed contiguously starting at `PAGE_HEADER_SIZE`.
    #[inline]
    #[must_use]
    pub fn leaf2_key(&self, idx: usize, key_size: usize) -> &'a [u8] {
        let start = PAGE_HEADER_SIZE + idx * key_size;
        &self.data[start..start + key_size]
    }

    /// Interpret this page as a meta page and return a copy of the
    /// [`Meta`] structure.
    ///
    /// # Safety
    ///
    /// The caller must ensure this is actually a meta page. The cast is
    /// performed via `std::ptr::read_unaligned` to handle potential
    /// alignment issues on the mmap'd region.
    #[inline]
    #[must_use]
    pub fn meta(&self) -> Meta {
        let src = &self.data[PAGE_HEADER_SIZE..];
        debug_assert!(
            src.len() >= mem::size_of::<Meta>(),
            "page too small for Meta",
        );
        // SAFETY: Meta is repr(C) and we only read from a properly sized
        // byte slice. `read_unaligned` handles any alignment mismatch.
        unsafe { std::ptr::read_unaligned(src.as_ptr().cast::<Meta>()) }
    }

    /// Return the raw backing bytes.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Internal helper: read a fixed-size byte array at `offset`.
    #[inline]
    fn read_array<const N: usize>(&self, offset: usize) -> [u8; N] {
        let mut buf = [0u8; N];
        buf.copy_from_slice(&self.data[offset..offset + N]);
        buf
    }
}

/// A read-only view of a node within a page.
///
/// # Node header layout (8 bytes)
///
/// ```text
/// Offset  Size  Field
/// 0       2     lo   -- low 16 bits of data size or child pgno
/// 2       2     hi   -- high 16 bits
/// 4       2     flags (NodeFlags)
/// 6       2     ksize (key length in bytes)
/// ```
///
/// Key bytes immediately follow the header; data (or child page number)
/// follows the key.
#[derive(Clone, Copy, Debug)]
pub struct Node<'a> {
    data: &'a [u8],
}

impl<'a> Node<'a> {
    /// Wrap a raw byte slice as a node. The slice must start at the node
    /// header.
    #[inline]
    #[must_use]
    pub fn from_raw(data: &'a [u8]) -> Self {
        debug_assert!(
            data.len() >= NODE_HEADER_SIZE,
            "node data too short: {} < {NODE_HEADER_SIZE}",
            data.len(),
        );
        Self { data }
    }

    /// Low 16 bits -- part of data size (leaf) or child pgno (branch).
    #[inline]
    #[must_use]
    pub fn lo(&self) -> u16 {
        u16::from_le_bytes(self.read_array(0))
    }

    /// High 16 bits -- part of data size (leaf) or child pgno (branch).
    #[inline]
    #[must_use]
    pub fn hi(&self) -> u16 {
        u16::from_le_bytes(self.read_array(2))
    }

    /// Node flags.
    #[inline]
    #[must_use]
    pub fn flags(&self) -> NodeFlags {
        NodeFlags::from_bits_truncate(u16::from_le_bytes(self.read_array(4)))
    }

    /// Key size in bytes.
    #[inline]
    #[must_use]
    pub fn key_size(&self) -> u16 {
        u16::from_le_bytes(self.read_array(6))
    }

    /// The key bytes (immediately after the 8-byte header).
    #[inline]
    #[must_use]
    pub fn key(&self) -> &'a [u8] {
        let ks = usize::from(self.key_size());
        &self.data[NODE_HEADER_SIZE..NODE_HEADER_SIZE + ks]
    }

    /// Data size for leaf nodes: `lo | (hi << 16)`.
    #[inline]
    #[must_use]
    pub fn data_size(&self) -> u32 {
        u32::from(self.lo()) | (u32::from(self.hi()) << 16)
    }

    /// Child page number for branch nodes.
    ///
    /// Encoded as: `lo | (hi << 16) | (flags_raw << 32)`, giving a 48-bit
    /// page number.
    #[inline]
    #[must_use]
    pub fn child_pgno(&self) -> u64 {
        let lo = u64::from(self.lo());
        let hi = u64::from(self.hi());
        let flags_raw = u64::from(u16::from_le_bytes(self.read_array(4)));
        lo | (hi << 16) | (flags_raw << 32)
    }

    /// Data bytes for leaf nodes.
    ///
    /// Starts at `NODE_HEADER_SIZE + key_size`. If the `BIGDATA` flag is set
    /// the data is an 8-byte overflow page number rather than inline data.
    #[inline]
    #[must_use]
    pub fn node_data(&self) -> &'a [u8] {
        let start = NODE_HEADER_SIZE + usize::from(self.key_size());
        let len = if self.is_bigdata() {
            8 // u64 pgno
        } else {
            self.data_size() as usize
        };
        &self.data[start..start + len]
    }

    /// When `BIGDATA` is set, read the overflow page number from `node_data`.
    #[inline]
    #[must_use]
    pub fn overflow_pgno(&self) -> u64 {
        let d = self.node_data();
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&d[..8]);
        u64::from_le_bytes(buf)
    }

    /// When `SUBDATA` is set, read the [`DbStat`] from `node_data`.
    ///
    /// Uses `read_unaligned` because mmap'd data may not be naturally aligned.
    #[inline]
    #[must_use]
    pub fn sub_db(&self) -> DbStat {
        let d = self.node_data();
        debug_assert!(
            d.len() >= mem::size_of::<DbStat>(),
            "node_data too small for DbStat",
        );
        // SAFETY: DbStat is repr(C) and we read from a byte slice of
        // sufficient length. `read_unaligned` handles alignment.
        unsafe { std::ptr::read_unaligned(d.as_ptr().cast::<DbStat>()) }
    }

    /// When `DUPDATA` is set (but not `SUBDATA`), the data is an inline
    /// sub-page.
    #[inline]
    #[must_use]
    pub fn sub_page(&self) -> Page<'a> {
        Page::from_raw(self.node_data())
    }

    /// Returns `true` if the node references an overflow page (`BIGDATA`).
    #[inline]
    #[must_use]
    pub fn is_bigdata(&self) -> bool {
        self.flags().contains(NodeFlags::BIGDATA)
    }

    /// Returns `true` if the node contains a sub-database record (`SUBDATA`).
    #[inline]
    #[must_use]
    pub fn is_subdata(&self) -> bool {
        self.flags().contains(NodeFlags::SUBDATA)
    }

    /// Returns `true` if the node has duplicate data (`DUPDATA`).
    #[inline]
    #[must_use]
    pub fn is_dupdata(&self) -> bool {
        self.flags().contains(NodeFlags::DUPDATA)
    }

    /// Total size of this node on disk, including header, key, and data,
    /// rounded up to the nearest even byte boundary.
    #[inline]
    #[must_use]
    pub fn total_size(&self) -> usize {
        even(NODE_HEADER_SIZE + usize::from(self.key_size()) + self.data_size() as usize)
    }

    /// Internal helper: read a fixed-size byte array at `offset`.
    #[inline]
    fn read_array<const N: usize>(&self, offset: usize) -> [u8; N] {
        let mut buf = [0u8; N];
        buf.copy_from_slice(&self.data[offset..offset + N]);
        buf
    }
}

/// A mutable view of a database page backed by a `&mut [u8]` slice.
///
/// Provides write access for constructing or modifying pages during
/// transactions. Read access is available via [`MutablePage::as_page`].
#[derive(Debug)]
pub struct MutablePage<'a> {
    data: &'a mut [u8],
}

impl<'a> MutablePage<'a> {
    /// Wrap a mutable byte slice as a page.
    #[inline]
    #[must_use]
    pub fn from_raw(data: &'a mut [u8]) -> Self {
        debug_assert!(
            data.len() >= PAGE_HEADER_SIZE,
            "page data too short: {} < {PAGE_HEADER_SIZE}",
            data.len(),
        );
        Self { data }
    }

    /// Write the page number (bytes 0..8, little-endian).
    #[inline]
    pub fn set_pgno(&mut self, pgno: u64) {
        self.data[0..8].copy_from_slice(&pgno.to_le_bytes());
    }

    /// Write the page flags (bytes 10..12, little-endian).
    #[inline]
    pub fn set_flags(&mut self, flags: PageFlags) {
        self.data[10..12].copy_from_slice(&flags.bits().to_le_bytes());
    }

    /// Write `lower` (bytes 12..14, little-endian).
    #[inline]
    pub fn set_lower(&mut self, lower: u16) {
        self.data[12..14].copy_from_slice(&lower.to_le_bytes());
    }

    /// Write `upper` (bytes 14..16, little-endian).
    #[inline]
    pub fn set_upper(&mut self, upper: u16) {
        self.data[14..16].copy_from_slice(&upper.to_le_bytes());
    }

    /// Return a read-only [`Page`] view of this mutable page.
    #[inline]
    #[must_use]
    pub fn as_page(&self) -> Page<'_> {
        Page::from_raw(self.data)
    }

    /// Return the underlying mutable byte slice.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeFlags, PAGE_HEADER_SIZE, PageFlags};

    /// Build a minimal page buffer with the given header fields.
    fn make_page_buf(
        pgno: u64,
        pad: u16,
        flags: PageFlags,
        lower: u16,
        upper: u16,
        extra: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(PAGE_HEADER_SIZE + extra.len());
        buf.extend_from_slice(&pgno.to_le_bytes()); // 0..8
        buf.extend_from_slice(&pad.to_le_bytes()); // 8..10
        buf.extend_from_slice(&flags.bits().to_le_bytes()); // 10..12
        buf.extend_from_slice(&lower.to_le_bytes()); // 12..14
        buf.extend_from_slice(&upper.to_le_bytes()); // 14..16
        buf.extend_from_slice(extra);
        buf
    }

    /// Build a node: header + key + data.
    fn make_node(lo: u16, hi: u16, flags: NodeFlags, key: &[u8], data: &[u8]) -> Vec<u8> {
        let klen = u16::try_from(key.len()).expect("key too long for test");
        let mut buf = Vec::new();
        buf.extend_from_slice(&lo.to_le_bytes());
        buf.extend_from_slice(&hi.to_le_bytes());
        buf.extend_from_slice(&flags.bits().to_le_bytes());
        buf.extend_from_slice(&klen.to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(data);
        buf
    }

    #[test]
    fn test_even() {
        assert_eq!(even(0), 0);
        assert_eq!(even(1), 2);
        assert_eq!(even(2), 2);
        assert_eq!(even(3), 4);
        assert_eq!(even(4), 4);
        assert_eq!(even(5), 6);
        assert_eq!(even(100), 100);
        assert_eq!(even(101), 102);
    }

    #[test]
    fn test_should_read_leaf_page_header() {
        let num_keys = 2u16;
        let lower = PAGE_HEADER_SIZE as u16 + num_keys * 2; // 16 + 4 = 20
        let upper = 4096u16;
        let buf = make_page_buf(42, 0, PageFlags::LEAF, lower, upper, &[0u8; 128]);
        let page = Page::from_raw(&buf);

        assert_eq!(page.pgno(), 42);
        assert_eq!(page.pad(), 0);
        assert!(page.is_leaf());
        assert!(!page.is_branch());
        assert!(!page.is_overflow());
        assert!(!page.is_leaf2());
        assert!(!page.is_subpage());
        assert_eq!(page.lower(), lower);
        assert_eq!(page.upper(), upper);
        assert_eq!(page.num_keys(), 2);
        assert_eq!(page.free_space(), usize::from(upper - lower));
    }

    #[test]
    fn test_should_read_branch_page_header() {
        let lower = PAGE_HEADER_SIZE as u16 + 2; // 1 key
        let upper = 100u16;
        let buf = make_page_buf(7, 0, PageFlags::BRANCH, lower, upper, &[0u8; 128]);
        let page = Page::from_raw(&buf);

        assert!(page.is_branch());
        assert!(!page.is_leaf());
        assert_eq!(page.num_keys(), 1);
        assert_eq!(page.pgno(), 7);
    }

    #[test]
    fn test_should_parse_leaf_node() {
        let key = b"hello";
        let val = b"world";
        let node_bytes = make_node(
            val.len() as u16, // lo = data size low
            0,                // hi = 0
            NodeFlags::empty(),
            key,
            val,
        );

        // Place node at some offset within the page
        let node_offset = 64u16;
        let lower = PAGE_HEADER_SIZE as u16 + 2; // 1 pointer
        let mut extra = vec![0u8; 256];
        // Write node pointer
        extra[0..2].copy_from_slice(&node_offset.to_le_bytes());
        // Write node data at offset (relative to page start, but our extra
        // starts at PAGE_HEADER_SIZE so subtract that)
        let node_start = usize::from(node_offset) - PAGE_HEADER_SIZE;
        extra[node_start..node_start + node_bytes.len()].copy_from_slice(&node_bytes);

        let buf = make_page_buf(1, 0, PageFlags::LEAF, lower, 200, &extra);
        let page = Page::from_raw(&buf);

        assert_eq!(page.num_keys(), 1);
        assert_eq!(page.ptr_at(0), node_offset);

        let node = page.node(0);
        assert_eq!(node.key_size(), 5);
        assert_eq!(node.key(), b"hello");
        assert_eq!(node.data_size(), 5);
        assert_eq!(node.node_data(), b"world");
        assert!(!node.is_bigdata());
        assert!(!node.is_subdata());
        assert!(!node.is_dupdata());
    }

    #[test]
    fn test_should_parse_branch_node() {
        let key = b"abc";
        // Branch node: child_pgno = lo | (hi << 16) | (flags_raw << 32)
        let child_lo: u16 = 0x1234;
        let child_hi: u16 = 0x0056;
        // For branch nodes, flags field holds upper bits of pgno
        let flags_raw: u16 = 0x0000;

        let node_bytes = make_node(
            child_lo,
            child_hi,
            NodeFlags::from_bits_truncate(flags_raw),
            key,
            &[], // branch nodes have no inline data
        );

        let node = Node::from_raw(&node_bytes);
        assert_eq!(node.key(), b"abc");
        let expected_pgno = u64::from(child_lo) | (u64::from(child_hi) << 16);
        assert_eq!(node.child_pgno(), expected_pgno);
    }

    #[test]
    fn test_should_access_leaf2_keys() {
        let key_size = 4usize;
        let keys: &[u8] = &[
            0x01, 0x02, 0x03, 0x04, // key 0
            0x05, 0x06, 0x07, 0x08, // key 1
            0x09, 0x0A, 0x0B, 0x0C, // key 2
        ];

        let lower = PAGE_HEADER_SIZE as u16; // no pointer array for LEAF2
        let upper = (PAGE_HEADER_SIZE + keys.len()) as u16;
        let buf = make_page_buf(
            10,
            key_size as u16,
            PageFlags::LEAF | PageFlags::LEAF2,
            lower,
            upper,
            keys,
        );
        let page = Page::from_raw(&buf);

        assert!(page.is_leaf2());
        assert_eq!(page.pad(), key_size as u16);
        assert_eq!(page.leaf2_key(0, key_size), &[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(page.leaf2_key(1, key_size), &[0x05, 0x06, 0x07, 0x08]);
        assert_eq!(page.leaf2_key(2, key_size), &[0x09, 0x0A, 0x0B, 0x0C]);
    }

    #[test]
    fn test_should_parse_overflow_page() {
        let num_overflow: u32 = 5;
        let lower = (num_overflow & 0xFFFF) as u16;
        let upper = ((num_overflow >> 16) & 0xFFFF) as u16;
        let buf = make_page_buf(99, 0, PageFlags::OVERFLOW, lower, upper, &[0u8; 32]);
        let page = Page::from_raw(&buf);

        assert!(page.is_overflow());
        assert_eq!(page.overflow_pages(), 5);
        assert_eq!(page.pgno(), 99);
    }

    #[test]
    fn test_should_read_bigdata_node() {
        let key = b"big";
        let overflow_pgno: u64 = 0x0000_0000_DEAD_BEEF;
        let lo = 8u16; // data_size = 8 (size of the pgno reference, but BIGDATA overrides)
        let node_bytes = make_node(lo, 0, NodeFlags::BIGDATA, key, &overflow_pgno.to_le_bytes());

        let node = Node::from_raw(&node_bytes);
        assert!(node.is_bigdata());
        assert_eq!(node.key(), b"big");
        assert_eq!(node.overflow_pgno(), overflow_pgno);
    }

    #[test]
    fn test_should_compute_node_total_size() {
        // key=5 bytes, data=3 bytes => header(8) + 5 + 3 = 16, even(16) = 16
        let node_bytes = make_node(3, 0, NodeFlags::empty(), b"hello", b"abc");
        let node = Node::from_raw(&node_bytes);
        assert_eq!(node.total_size(), 16);

        // key=5 bytes, data=4 bytes => header(8) + 5 + 4 = 17, even(17) = 18
        let node_bytes2 = make_node(4, 0, NodeFlags::empty(), b"hello", b"abcd");
        let node2 = Node::from_raw(&node_bytes2);
        assert_eq!(node2.total_size(), 18);
    }

    #[test]
    fn test_should_roundtrip_mutable_page() {
        let mut buf = vec![0u8; 4096];
        {
            let mut mp = MutablePage::from_raw(&mut buf);
            mp.set_pgno(123);
            mp.set_flags(PageFlags::LEAF | PageFlags::DIRTY);
            mp.set_lower(20);
            mp.set_upper(4000);

            let page = mp.as_page();
            assert_eq!(page.pgno(), 123);
            assert!(page.is_leaf());
            assert!(page.flags().contains(PageFlags::DIRTY));
            assert_eq!(page.lower(), 20);
            assert_eq!(page.upper(), 4000);
        }

        // Verify via fresh Page view
        let page = Page::from_raw(&buf);
        assert_eq!(page.pgno(), 123);
        assert_eq!(page.lower(), 20);
        assert_eq!(page.upper(), 4000);
    }

    #[test]
    fn test_should_report_used_space() {
        let total_size = 256usize;
        let lower = 24u16; // header(16) + 4 pointers
        let upper = 200u16;
        let extra = vec![0u8; total_size - PAGE_HEADER_SIZE];
        let buf = make_page_buf(0, 0, PageFlags::LEAF, lower, upper, &extra);
        let page = Page::from_raw(&buf);

        let free = usize::from(upper - lower);
        let used = total_size - PAGE_HEADER_SIZE - free;
        assert_eq!(page.free_space(), free);
        assert_eq!(page.used_space(), used);
    }
}
