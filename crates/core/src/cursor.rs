//! B+ tree cursor for traversing and positioning within the database.
//!
//! A [`Cursor`] holds a stack of page pointers from the root to the current
//! leaf position in the B+ tree. For read-only operations, the page data
//! points into the memory-mapped region and is guaranteed to outlive the
//! cursor (the mmap lives as long as the transaction).
//!
//! # Safety
//!
//! The only `unsafe` blocks are in [`Cursor::current_page`] and
//! [`Cursor::page_at`], which reconstruct a `&[u8]` slice from a raw
//! pointer into mmap'd memory. This is safe because the mmap outlives
//! the cursor and the transaction holds the pages pinned.

use std::cmp::Ordering;

use bitflags::bitflags;

use crate::{
    cmp::CmpFn,
    error::{Error, Result},
    page::{Node, Page},
    types::CURSOR_STACK,
};

/// Maximum depth of the B+ tree cursor stack.
const STACK_DEPTH: usize = CURSOR_STACK;

bitflags! {
    /// Flags tracking cursor state.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CursorFlags: u32 {
        /// Cursor has been positioned at a valid entry.
        const INITIALIZED = 0x01;
        /// Cursor is past the last entry.
        const EOF = 0x02;
        /// Cursor operates on a sub-database (DUPSORT inner cursor).
        const SUB = 0x04;
        /// Last operation was a delete.
        const DEL = 0x08;
    }
}

/// B+ tree cursor for navigating database pages.
///
/// The cursor maintains a stack of raw page pointers from the root down to
/// the current leaf, along with the key index at each level. This allows
/// efficient traversal (next/prev) without re-searching from the root.
pub struct Cursor {
    /// Stack of page raw pointers from root to current position.
    /// Each entry points to the start of a page in mmap'd memory.
    pages: [*const u8; STACK_DEPTH],
    /// Page size in bytes (needed to reconstruct `Page` from raw pointer).
    page_size: usize,
    /// Key index at each level of the stack.
    indices: [u16; STACK_DEPTH],
    /// Number of pages pushed on the stack (depth from root).
    snum: u16,
    /// Top of stack index (`snum - 1`), or 0 if empty.
    top: u16,
    /// Cursor state flags.
    flags: CursorFlags,
    /// Database handle index.
    pub(crate) dbi: u32,
}

// SAFETY: The raw pointers in `pages` point into mmap'd memory that is
// shared and read-only for the lifetime of the transaction. The cursor
// itself does not provide mutable access through these pointers.
unsafe impl Send for Cursor {}

impl std::fmt::Debug for Cursor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cursor")
            .field("page_size", &self.page_size)
            .field("snum", &self.snum)
            .field("top", &self.top)
            .field("flags", &self.flags)
            .field("dbi", &self.dbi)
            .finish_non_exhaustive()
    }
}

impl Cursor {
    /// Creates a new cursor for the given database handle.
    ///
    /// The cursor starts uninitialized — call one of the positioning methods
    /// ([`first`](Self::first), [`last`](Self::last), [`set`](Self::set),
    /// [`page_search`](Self::page_search)) before reading data.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cursor = Cursor::new(4096, 0);
    /// assert!(!cursor.is_initialized());
    /// ```
    pub fn new(page_size: usize, dbi: u32) -> Self {
        Self {
            pages: [std::ptr::null(); STACK_DEPTH],
            page_size,
            indices: [0; STACK_DEPTH],
            snum: 0,
            top: 0,
            flags: CursorFlags::empty(),
            dbi,
        }
    }

    /// Returns `true` if the cursor has been positioned at a valid entry.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.flags.contains(CursorFlags::INITIALIZED)
    }

    /// Returns `true` if the cursor is past the last entry.
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.flags.contains(CursorFlags::EOF)
    }

    /// Returns the current cursor state flags.
    #[inline]
    pub fn cursor_flags(&self) -> CursorFlags {
        self.flags
    }

    /// Returns the page at the top of the stack.
    ///
    /// The returned `Page` has an unbound lifetime because the underlying
    /// data lives in the mmap (not in `self`). The caller must ensure the
    /// mmap outlives any use of the returned page.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the stack is non-empty.
    fn current_page<'a>(&self) -> Page<'a> {
        debug_assert!(self.snum > 0, "cursor stack is empty");
        // SAFETY: The pointer was obtained from the mmap which outlives the
        // cursor. We use an unbound lifetime because the data is not owned
        // by self — it's in the memory-mapped region.
        unsafe {
            let ptr = self.pages[self.top as usize];
            let slice = std::slice::from_raw_parts(ptr, self.page_size);
            Page::from_raw(slice)
        }
    }

    /// Returns the page at the given stack level.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `level` is within the stack bounds.
    #[allow(dead_code)] // Used in later phases (page split, COW)
    fn page_at<'a>(&self, level: usize) -> Page<'a> {
        debug_assert!(level < self.snum as usize, "level out of stack bounds");
        // SAFETY: Same as `current_page` — pointer into mmap, pinned by txn.
        unsafe {
            let ptr = self.pages[level];
            let slice = std::slice::from_raw_parts(ptr, self.page_size);
            Page::from_raw(slice)
        }
    }

    /// Pushes a page pointer onto the cursor stack.
    fn push_page(&mut self, page_ptr: *const u8) {
        debug_assert!((self.snum as usize) < STACK_DEPTH, "cursor stack overflow");
        let level = self.snum as usize;
        self.pages[level] = page_ptr;
        self.indices[level] = 0;
        self.snum += 1;
        self.top = self.snum - 1;
    }

    /// Pops the top page from the cursor stack.
    fn pop_page(&mut self) {
        if self.snum > 0 {
            self.snum -= 1;
            self.top = if self.snum > 0 { self.snum - 1 } else { 0 };
        }
    }

    /// Binary search for a key within a single page.
    ///
    /// Returns `(index, exact_match)` where `index` is the position at which
    /// the key was found (if exact) or should be inserted. On branch pages
    /// index 0 is skipped because the first branch node has an implicit
    /// empty key (it serves as the leftmost child pointer).
    pub fn node_search(&self, page: &Page<'_>, key: &[u8], cmp: &CmpFn) -> (usize, bool) {
        let nkeys = page.num_keys();
        if nkeys == 0 {
            return (0, false);
        }

        // Branch pages: index 0 holds the leftmost child pointer with an
        // implicit empty key, so binary search starts at index 1.
        let low = if page.is_branch() { 1 } else { 0 };
        let mut lo = low;
        let mut hi = nkeys;
        let mut exact = false;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let node_key = if page.is_leaf2() {
                page.leaf2_key(mid, page.pad() as usize)
            } else {
                page.node(mid).key()
            };
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

    /// Search from root to leaf for a key, positioning the cursor.
    ///
    /// `get_page` resolves a page number to a raw pointer into the mmap.
    /// After a successful call the cursor stack traces the path from root
    /// to the leaf containing (or nearest to) the requested key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if `root_pgno` is invalid.
    /// Returns [`Error::Corrupted`] if a non-branch, non-leaf page is
    /// encountered during traversal.
    pub fn page_search<F>(
        &mut self,
        root_pgno: u64,
        key: Option<&[u8]>,
        cmp: &CmpFn,
        get_page: &F,
    ) -> Result<()>
    where
        F: Fn(u64) -> Result<*const u8>,
    {
        if root_pgno == crate::types::P_INVALID {
            return Err(Error::NotFound);
        }

        // Reset the stack.
        self.snum = 0;
        self.top = 0;
        self.flags.remove(CursorFlags::EOF);

        let root_ptr = get_page(root_pgno)?;
        self.push_page(root_ptr);

        loop {
            let page = self.current_page();

            if page.is_leaf() {
                if let Some(k) = key {
                    let (idx, _exact) = self.node_search(&page, k, cmp);
                    self.indices[self.top as usize] = idx as u16;
                }
                self.flags.insert(CursorFlags::INITIALIZED);
                return Ok(());
            }

            if !page.is_branch() {
                return Err(Error::Corrupted);
            }

            // Search for the child to descend into.
            let idx = if let Some(k) = key {
                let (idx, exact) = self.node_search(&page, k, cmp);
                if exact {
                    idx
                } else if idx > 0 {
                    // Branch keys are separator keys — if the search key is
                    // not an exact match, the correct child is at idx - 1.
                    idx - 1
                } else {
                    // Key is smaller than all separators — go to leftmost child.
                    0
                }
            } else {
                // No key specified: go to the leftmost child.
                0
            };

            self.indices[self.top as usize] = idx as u16;

            let child_pgno = page.node(idx).child_pgno();
            let child_ptr = get_page(child_pgno)?;
            self.push_page(child_ptr);
        }
    }

    /// Position the cursor at the first key/data item.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the database is empty.
    pub fn first<F>(&mut self, root_pgno: u64, cmp: &CmpFn, get_page: &F) -> Result<()>
    where
        F: Fn(u64) -> Result<*const u8>,
    {
        let _ = cmp; // Not needed for first — always go leftmost.
        self.snum = 0;
        self.top = 0;
        self.flags.remove(CursorFlags::EOF);

        if root_pgno == crate::types::P_INVALID {
            return Err(Error::NotFound);
        }

        let root_ptr = get_page(root_pgno)?;
        self.push_page(root_ptr);

        loop {
            let page = self.current_page();
            self.indices[self.top as usize] = 0;

            if page.is_leaf() {
                if page.num_keys() == 0 {
                    return Err(Error::NotFound);
                }
                self.flags.insert(CursorFlags::INITIALIZED);
                return Ok(());
            }

            // Branch page: follow leftmost child.
            let child_pgno = page.node(0).child_pgno();
            let child_ptr = get_page(child_pgno)?;
            self.push_page(child_ptr);
        }
    }

    /// Position the cursor at the last key/data item.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the database is empty.
    pub fn last<F>(&mut self, root_pgno: u64, cmp: &CmpFn, get_page: &F) -> Result<()>
    where
        F: Fn(u64) -> Result<*const u8>,
    {
        let _ = cmp; // Not needed for last — always go rightmost.
        self.snum = 0;
        self.top = 0;
        self.flags.remove(CursorFlags::EOF);

        if root_pgno == crate::types::P_INVALID {
            return Err(Error::NotFound);
        }

        let root_ptr = get_page(root_pgno)?;
        self.push_page(root_ptr);

        loop {
            let page = self.current_page();
            let nkeys = page.num_keys();
            if nkeys == 0 {
                return Err(Error::NotFound);
            }

            self.indices[self.top as usize] = (nkeys - 1) as u16;

            if page.is_leaf() {
                self.flags.insert(CursorFlags::INITIALIZED);
                return Ok(());
            }

            // Branch page: follow rightmost child.
            let child_pgno = page.node(nkeys - 1).child_pgno();
            let child_ptr = get_page(child_pgno)?;
            self.push_page(child_ptr);
        }
    }

    /// Move to a sibling page (left or right).
    ///
    /// This ascends the tree until a parent with a suitable neighbor is found,
    /// then descends into that neighbor. If `right` is `true`, moves to the
    /// next sibling; otherwise moves to the previous one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if there is no sibling in the requested
    /// direction (already at the first/last page).
    pub fn sibling<F>(&mut self, right: bool, get_page: &F) -> Result<()>
    where
        F: Fn(u64) -> Result<*const u8>,
    {
        if self.snum < 2 {
            return Err(Error::NotFound);
        }

        // Pop current leaf/child page.
        self.pop_page();

        let idx = self.indices[self.top as usize] as usize;
        let page = self.current_page();
        let nkeys = page.num_keys();

        if right {
            if idx + 1 >= nkeys {
                // Parent has no more children to the right — ascend further.
                return self.sibling(right, get_page);
            }
            self.indices[self.top as usize] = (idx + 1) as u16;
        } else {
            if idx == 0 {
                // Parent has no more children to the left — ascend further.
                return self.sibling(right, get_page);
            }
            self.indices[self.top as usize] = (idx - 1) as u16;
        }

        let new_idx = self.indices[self.top as usize] as usize;
        let child_pgno = page.node(new_idx).child_pgno();
        let child_ptr = get_page(child_pgno)?;
        self.push_page(child_ptr);

        // Position at the first or last key of the new sibling page.
        let new_page = self.current_page();
        let nk = new_page.num_keys();
        self.indices[self.top as usize] = if right {
            0
        } else if nk > 0 {
            (nk - 1) as u16
        } else {
            0
        };

        Ok(())
    }

    /// Move cursor to the next key/data item.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the cursor is not initialized or
    /// there is no next item.
    pub fn next<F>(&mut self, get_page: &F) -> Result<()>
    where
        F: Fn(u64) -> Result<*const u8>,
    {
        if !self.is_initialized() {
            return Err(Error::NotFound);
        }

        let page = self.current_page();
        let idx = self.indices[self.top as usize] as usize;

        if idx + 1 < page.num_keys() {
            self.indices[self.top as usize] = (idx + 1) as u16;
            self.flags.remove(CursorFlags::EOF);
            return Ok(());
        }

        // Need to move to the next sibling page.
        self.sibling(true, get_page)
    }

    /// Move cursor to the previous key/data item.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the cursor is not initialized or
    /// there is no previous item.
    pub fn prev<F>(&mut self, get_page: &F) -> Result<()>
    where
        F: Fn(u64) -> Result<*const u8>,
    {
        if !self.is_initialized() {
            return Err(Error::NotFound);
        }

        let idx = self.indices[self.top as usize] as usize;

        if idx > 0 {
            self.indices[self.top as usize] = (idx - 1) as u16;
            self.flags.remove(CursorFlags::EOF);
            return Ok(());
        }

        // Need to move to the previous sibling page.
        self.sibling(false, get_page)
    }

    /// Position the cursor at the specified key.
    ///
    /// Returns `Ok(true)` for an exact match, `Ok(false)` when positioned
    /// at the nearest key >= the requested key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the key is beyond all entries.
    pub fn set<F>(&mut self, root_pgno: u64, key: &[u8], cmp: &CmpFn, get_page: &F) -> Result<bool>
    where
        F: Fn(u64) -> Result<*const u8>,
    {
        self.page_search(root_pgno, Some(key), cmp, get_page)?;

        let page = self.current_page();
        let idx = self.indices[self.top as usize] as usize;
        let nkeys = page.num_keys();

        if idx >= nkeys {
            self.flags.insert(CursorFlags::EOF);
            return Err(Error::NotFound);
        }

        // Check for exact match.
        let node_key = if page.is_leaf2() {
            page.leaf2_key(idx, page.pad() as usize)
        } else {
            page.node(idx).key()
        };
        let exact = cmp(key, node_key) == Ordering::Equal;

        Ok(exact)
    }

    /// Returns the key at the current cursor position, or `None` if the
    /// cursor is not initialized or at EOF.
    pub fn current_key(&self) -> Option<&[u8]> {
        if !self.is_initialized() || self.is_eof() {
            return None;
        }
        let page = self.current_page();
        let idx = self.indices[self.top as usize] as usize;
        if idx >= page.num_keys() {
            return None;
        }
        if page.is_leaf2() {
            Some(page.leaf2_key(idx, page.pad() as usize))
        } else {
            Some(page.node(idx).key())
        }
    }

    /// Returns the [`Node`] at the current cursor position, or `None` if the
    /// cursor is not initialized, at EOF, or on a `LEAF2` page.
    pub fn current_node(&self) -> Option<Node<'_>> {
        if !self.is_initialized() || self.is_eof() {
            return None;
        }
        let page = self.current_page();
        let idx = self.indices[self.top as usize] as usize;
        if idx >= page.num_keys() || page.is_leaf2() {
            return None;
        }
        Some(page.node(idx))
    }

    /// Returns the key index at the top of the cursor stack.
    #[inline]
    pub fn current_index(&self) -> usize {
        self.indices[self.top as usize] as usize
    }

    /// Returns the stack depth (number of pages from root to current position).
    #[inline]
    pub fn depth(&self) -> usize {
        self.snum as usize
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;
    use crate::types::{NODE_HEADER_SIZE, PAGE_HEADER_SIZE, PageFlags};

    // -----------------------------------------------------------------------
    // Test helpers — build minimal page buffers
    // -----------------------------------------------------------------------

    const TEST_PAGE_SIZE: usize = 4096;

    /// Build a leaf page with the given keys.
    ///
    /// Layout (simplified, matching the Page / Node on-disk format):
    ///   bytes 0..2:   page number low (u64 LE, we only write 8 bytes)
    ///   bytes 8..10:  pad (u16 LE)
    ///   bytes 10..12: flags (u16 LE) — PageFlags::LEAF
    ///   bytes 12..14: num_keys (u16 LE, called `lower` internally)
    ///   bytes 14..16: free space pointer / upper bound
    ///
    /// For each key, a node pointer (u16) is stored at the end of the header
    /// area, and the actual node is stored from the end of the page growing
    /// backwards.
    ///
    /// Node layout:
    ///   bytes 0..4: data size (u32 LE)
    ///   bytes 4..6: key size (u16 LE)
    ///   bytes 6..8: flags (u16 LE)
    ///   bytes 8..:  key bytes, then data bytes
    fn build_leaf_page(keys: &[&[u8]], data: &[&[u8]]) -> Vec<u8> {
        assert_eq!(keys.len(), data.len());
        let mut buf = vec![0u8; TEST_PAGE_SIZE];

        // Page header: pgno (8 bytes) + pad (2) + flags (2) + lower (2) + upper (2)
        // We write pgno = 0.
        let flags = PageFlags::LEAF.bits();
        let nkeys = keys.len() as u16;

        // flags at offset 10
        buf[10] = (flags & 0xFF) as u8;
        buf[11] = (flags >> 8) as u8;

        // We need to lay out:
        //   - node offset table: starts at PAGE_HEADER_SIZE, 2 bytes per key
        //   - nodes: packed from the end of the page backwards
        let offsets_start = PAGE_HEADER_SIZE;
        let offsets_end = offsets_start + 2 * keys.len();

        // lower = offsets_end (marks end of the offset table)
        buf[12] = (offsets_end & 0xFF) as u8;
        buf[13] = ((offsets_end >> 8) & 0xFF) as u8;

        // Place nodes from the end of the page
        let mut upper = TEST_PAGE_SIZE;
        for (i, (key, val)) in keys.iter().zip(data.iter()).enumerate() {
            let node_size = NODE_HEADER_SIZE + key.len() + val.len();
            upper -= node_size;

            // Write offset into the offset table
            let offset_pos = offsets_start + 2 * i;
            buf[offset_pos] = (upper & 0xFF) as u8;
            buf[offset_pos + 1] = ((upper >> 8) & 0xFF) as u8;

            // Write node header: lo(2) hi(2) flags(2) ksize(2)
            // For leaf nodes: data_size = lo | (hi << 16)
            let ds = val.len() as u32;
            let lo = (ds & 0xFFFF) as u16;
            let hi = ((ds >> 16) & 0xFFFF) as u16;
            buf[upper..upper + 2].copy_from_slice(&lo.to_le_bytes());
            buf[upper + 2..upper + 4].copy_from_slice(&hi.to_le_bytes());
            // node flags = 0
            buf[upper + 4..upper + 6].copy_from_slice(&0u16.to_le_bytes());
            // key size
            let ks = key.len() as u16;
            buf[upper + 6..upper + 8].copy_from_slice(&ks.to_le_bytes());

            // key bytes
            buf[upper + NODE_HEADER_SIZE..upper + NODE_HEADER_SIZE + key.len()]
                .copy_from_slice(key);
            // data bytes
            buf[upper + NODE_HEADER_SIZE + key.len()
                ..upper + NODE_HEADER_SIZE + key.len() + val.len()]
                .copy_from_slice(val);
        }

        // upper at offset 14
        buf[14] = (upper & 0xFF) as u8;
        buf[15] = ((upper >> 8) & 0xFF) as u8;

        // num_keys is derived from (lower - PAGE_HEADER_SIZE) / 2 by Page,
        // which equals nkeys — verified by the offsets_end calculation above.
        let _ = nkeys;

        buf
    }

    /// Build a branch page with child page numbers and separator keys.
    ///
    /// Branch nodes encode the child page number in the node header fields:
    ///   `child_pgno = lo | (hi << 16) | (flags_raw << 32)`
    /// Index 0 has an empty key (implicit leftmost pointer).
    fn build_branch_page(child_pgnos: &[u64], separator_keys: &[&[u8]]) -> Vec<u8> {
        // child_pgnos.len() == separator_keys.len() + 1 conceptually:
        //   node[0] = empty key, child = child_pgnos[0]
        //   node[i] = separator_keys[i-1], child = child_pgnos[i]  for i >= 1
        assert_eq!(child_pgnos.len(), separator_keys.len() + 1);

        let nkeys = child_pgnos.len();
        let mut buf = vec![0u8; TEST_PAGE_SIZE];

        // flags = BRANCH
        let flags = PageFlags::BRANCH.bits();
        buf[10] = (flags & 0xFF) as u8;
        buf[11] = (flags >> 8) as u8;

        let offsets_start = PAGE_HEADER_SIZE;
        let offsets_end = offsets_start + 2 * nkeys;

        // lower
        buf[12] = (offsets_end & 0xFF) as u8;
        buf[13] = ((offsets_end >> 8) & 0xFF) as u8;

        let mut upper = TEST_PAGE_SIZE;
        for i in 0..nkeys {
            let key: &[u8] = if i == 0 { b"" } else { separator_keys[i - 1] };
            // Branch nodes have no data portion — the child pgno is encoded
            // in the header fields (lo, hi, flags).
            let node_size = NODE_HEADER_SIZE + key.len();
            upper -= node_size;

            let offset_pos = offsets_start + 2 * i;
            buf[offset_pos] = (upper & 0xFF) as u8;
            buf[offset_pos + 1] = ((upper >> 8) & 0xFF) as u8;

            // Encode child_pgno into lo (bytes 0..2), hi (bytes 2..4),
            // flags (bytes 4..6) of the node header.
            let pgno = child_pgnos[i];
            let lo = (pgno & 0xFFFF) as u16;
            let hi = ((pgno >> 16) & 0xFFFF) as u16;
            let flags_raw = ((pgno >> 32) & 0xFFFF) as u16;

            buf[upper] = (lo & 0xFF) as u8;
            buf[upper + 1] = (lo >> 8) as u8;
            buf[upper + 2] = (hi & 0xFF) as u8;
            buf[upper + 3] = (hi >> 8) as u8;
            buf[upper + 4] = (flags_raw & 0xFF) as u8;
            buf[upper + 5] = (flags_raw >> 8) as u8;

            let ks = key.len() as u16;
            buf[upper + 6] = (ks & 0xFF) as u8;
            buf[upper + 7] = ((ks >> 8) & 0xFF) as u8;

            buf[upper + NODE_HEADER_SIZE..upper + NODE_HEADER_SIZE + key.len()]
                .copy_from_slice(key);
        }

        buf[14] = (upper & 0xFF) as u8;
        buf[15] = ((upper >> 8) & 0xFF) as u8;

        buf
    }

    fn lexicographic_cmp(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_create_uninitialized_cursor() {
        let cursor = Cursor::new(4096, 0);
        assert!(!cursor.is_initialized());
        assert!(!cursor.is_eof());
        assert_eq!(cursor.depth(), 0);
        assert_eq!(cursor.current_index(), 0);
    }

    #[test]
    fn test_should_return_none_for_current_key_when_uninitialized() {
        let cursor = Cursor::new(4096, 0);
        assert!(cursor.current_key().is_none());
    }

    #[test]
    fn test_should_return_none_for_current_node_when_uninitialized() {
        let cursor = Cursor::new(4096, 0);
        assert!(cursor.current_node().is_none());
    }

    #[test]
    fn test_should_search_leaf_page_for_existing_key() {
        let page_buf = build_leaf_page(&[b"aaa", b"bbb", b"ccc"], &[b"v1", b"v2", b"v3"]);
        let page = Page::from_raw(&page_buf);
        let cursor = Cursor::new(TEST_PAGE_SIZE, 0);

        let (idx, exact) = cursor.node_search(&page, b"bbb", &lexicographic_cmp);
        assert!(exact);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_should_search_leaf_page_for_missing_key() {
        let page_buf = build_leaf_page(&[b"aaa", b"ccc", b"eee"], &[b"v1", b"v2", b"v3"]);
        let page = Page::from_raw(&page_buf);
        let cursor = Cursor::new(TEST_PAGE_SIZE, 0);

        let (idx, exact) = cursor.node_search(&page, b"bbb", &lexicographic_cmp);
        assert!(!exact);
        // "bbb" should be inserted at index 1 (between "aaa" and "ccc")
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_should_search_leaf_page_for_key_before_first() {
        let page_buf = build_leaf_page(&[b"bbb", b"ccc"], &[b"v1", b"v2"]);
        let page = Page::from_raw(&page_buf);
        let cursor = Cursor::new(TEST_PAGE_SIZE, 0);

        let (idx, exact) = cursor.node_search(&page, b"aaa", &lexicographic_cmp);
        assert!(!exact);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_should_search_leaf_page_for_key_after_last() {
        let page_buf = build_leaf_page(&[b"aaa", b"bbb"], &[b"v1", b"v2"]);
        let page = Page::from_raw(&page_buf);
        let cursor = Cursor::new(TEST_PAGE_SIZE, 0);

        let (idx, exact) = cursor.node_search(&page, b"zzz", &lexicographic_cmp);
        assert!(!exact);
        assert_eq!(idx, 2); // past all keys
    }

    #[test]
    fn test_should_search_empty_leaf_page() {
        let page_buf = build_leaf_page(&[], &[]);
        let page = Page::from_raw(&page_buf);
        let cursor = Cursor::new(TEST_PAGE_SIZE, 0);

        let (idx, exact) = cursor.node_search(&page, b"anything", &lexicographic_cmp);
        assert!(!exact);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_should_position_first_on_single_leaf() {
        let page_buf = build_leaf_page(&[b"aaa", b"bbb", b"ccc"], &[b"v1", b"v2", b"v3"]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        cursor.first(0, &lexicographic_cmp, &get_page).ok();
        assert!(cursor.is_initialized());
        assert_eq!(cursor.current_index(), 0);
        assert_eq!(cursor.depth(), 1);

        let key = cursor.current_key();
        assert_eq!(key, Some(b"aaa".as_slice()));
    }

    #[test]
    fn test_should_position_last_on_single_leaf() {
        let page_buf = build_leaf_page(&[b"aaa", b"bbb", b"ccc"], &[b"v1", b"v2", b"v3"]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        cursor.last(0, &lexicographic_cmp, &get_page).ok();
        assert!(cursor.is_initialized());
        assert_eq!(cursor.current_index(), 2);

        let key = cursor.current_key();
        assert_eq!(key, Some(b"ccc".as_slice()));
    }

    #[test]
    fn test_should_navigate_next_on_single_leaf() {
        let page_buf = build_leaf_page(&[b"aaa", b"bbb", b"ccc"], &[b"v1", b"v2", b"v3"]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        cursor.first(0, &lexicographic_cmp, &get_page).ok();
        assert_eq!(cursor.current_key(), Some(b"aaa".as_slice()));

        assert!(cursor.next(&get_page).is_ok());
        assert_eq!(cursor.current_key(), Some(b"bbb".as_slice()));

        assert!(cursor.next(&get_page).is_ok());
        assert_eq!(cursor.current_key(), Some(b"ccc".as_slice()));

        // No more items — single leaf, no sibling
        assert!(cursor.next(&get_page).is_err());
    }

    #[test]
    fn test_should_navigate_prev_on_single_leaf() {
        let page_buf = build_leaf_page(&[b"aaa", b"bbb", b"ccc"], &[b"v1", b"v2", b"v3"]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        cursor.last(0, &lexicographic_cmp, &get_page).ok();
        assert_eq!(cursor.current_key(), Some(b"ccc".as_slice()));

        assert!(cursor.prev(&get_page).is_ok());
        assert_eq!(cursor.current_key(), Some(b"bbb".as_slice()));

        assert!(cursor.prev(&get_page).is_ok());
        assert_eq!(cursor.current_key(), Some(b"aaa".as_slice()));

        // No more items
        assert!(cursor.prev(&get_page).is_err());
    }

    #[test]
    fn test_should_set_exact_key() {
        let page_buf = build_leaf_page(&[b"aaa", b"bbb", b"ccc"], &[b"v1", b"v2", b"v3"]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        let result = cursor.set(0, b"bbb", &lexicographic_cmp, &get_page);
        assert!(result.is_ok());
        assert!(result.ok() == Some(true));
        assert_eq!(cursor.current_key(), Some(b"bbb".as_slice()));
    }

    #[test]
    fn test_should_set_range_key() {
        let page_buf = build_leaf_page(&[b"aaa", b"ccc", b"eee"], &[b"v1", b"v2", b"v3"]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        // "bbb" is not in the page; cursor should be at "ccc" (next >= key)
        let result = cursor.set(0, b"bbb", &lexicographic_cmp, &get_page);
        assert!(result.is_ok());
        assert!(result.ok() == Some(false)); // not exact
        assert_eq!(cursor.current_key(), Some(b"ccc".as_slice()));
    }

    #[test]
    fn test_should_return_not_found_for_invalid_root() {
        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(std::ptr::null()) };

        let result = cursor.first(crate::types::P_INVALID, &lexicographic_cmp, &get_page);
        assert!(result.is_err());
    }

    #[test]
    fn test_should_return_not_found_for_empty_leaf() {
        let page_buf = build_leaf_page(&[], &[]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        let result = cursor.first(0, &lexicographic_cmp, &get_page);
        assert!(result.is_err());
    }

    #[test]
    fn test_should_return_not_found_when_next_on_uninitialized() {
        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(std::ptr::null()) };
        assert!(cursor.next(&get_page).is_err());
    }

    #[test]
    fn test_should_return_not_found_when_prev_on_uninitialized() {
        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(std::ptr::null()) };
        assert!(cursor.prev(&get_page).is_err());
    }

    #[test]
    fn test_should_push_and_pop_pages() {
        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        assert_eq!(cursor.depth(), 0);

        let buf = vec![0u8; TEST_PAGE_SIZE];
        cursor.push_page(buf.as_ptr());
        assert_eq!(cursor.depth(), 1);
        assert_eq!(cursor.top, 0);

        let buf2 = vec![0u8; TEST_PAGE_SIZE];
        cursor.push_page(buf2.as_ptr());
        assert_eq!(cursor.depth(), 2);
        assert_eq!(cursor.top, 1);

        cursor.pop_page();
        assert_eq!(cursor.depth(), 1);
        assert_eq!(cursor.top, 0);

        cursor.pop_page();
        assert_eq!(cursor.depth(), 0);
        assert_eq!(cursor.top, 0);

        // Popping empty stack is a no-op
        cursor.pop_page();
        assert_eq!(cursor.depth(), 0);
    }

    #[test]
    fn test_should_get_current_node() {
        let page_buf = build_leaf_page(&[b"key1", b"key2"], &[b"val1", b"val2"]);
        let page_ptr = page_buf.as_ptr();

        let mut cursor = Cursor::new(TEST_PAGE_SIZE, 0);
        let get_page = |_pgno: u64| -> Result<*const u8> { Ok(page_ptr) };

        cursor.first(0, &lexicographic_cmp, &get_page).ok();
        let node = cursor.current_node();
        assert!(node.is_some());
        let node = node.expect("node should exist");
        assert_eq!(node.key(), b"key1");
    }

    #[test]
    fn test_should_debug_format_cursor() {
        let cursor = Cursor::new(4096, 5);
        let debug_str = format!("{cursor:?}");
        assert!(debug_str.contains("Cursor"));
        assert!(debug_str.contains("dbi: 5"));
    }

    #[test]
    fn test_should_search_branch_page_skipping_index_zero() {
        // Branch page: child_pgnos[0]=10, separator="mmm", child_pgnos[1]=20
        let page_buf = build_branch_page(&[10, 20], &[b"mmm"]);
        let page = Page::from_raw(&page_buf);
        let cursor = Cursor::new(TEST_PAGE_SIZE, 0);

        // Key "aaa" < "mmm" => should go to index 0 (but binary search
        // starts at 1 for branches, so lo stays at 1, not exact).
        // The caller (page_search) then uses idx-1 = 0.
        let (idx, exact) = cursor.node_search(&page, b"aaa", &lexicographic_cmp);
        assert!(!exact);
        assert_eq!(idx, 1); // binary search returns 1 (insertion point)

        // Exact match on separator
        let (idx, exact) = cursor.node_search(&page, b"mmm", &lexicographic_cmp);
        assert!(exact);
        assert_eq!(idx, 1);

        // Key "zzz" > "mmm"
        let (idx, exact) = cursor.node_search(&page, b"zzz", &lexicographic_cmp);
        assert!(!exact);
        assert_eq!(idx, 2); // past all keys
    }
}
