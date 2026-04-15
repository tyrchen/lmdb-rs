//! Node-level operations for adding and removing entries in B+ tree pages.

use crate::{
    error::{Error, Result},
    page::even,
    types::{NODE_HEADER_SIZE, NodeFlags, PAGE_HEADER_SIZE, PageFlags},
};

/// Initialize a page buffer with header fields.
///
/// Sets the page number, flags, `lower` to `PAGE_HEADER_SIZE` (no pointers),
/// and `upper` to `page_size` (no nodes). The pad field is zeroed.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::node::init_page;
/// use lmdb_rs_core::types::{PageFlags, PAGE_HEADER_SIZE};
/// use lmdb_rs_core::page::Page;
///
/// let mut buf = vec![0u8; 4096];
/// init_page(&mut buf, 42, PageFlags::LEAF, 4096);
/// let page = Page::from_raw(&buf);
/// assert_eq!(page.pgno(), 42);
/// assert!(page.is_leaf());
/// assert_eq!(page.lower(), PAGE_HEADER_SIZE as u16);
/// assert_eq!(page.upper(), 4096);
/// ```
pub fn init_page(page: &mut [u8], pgno: u64, flags: PageFlags, page_size: usize) {
    page[0..8].copy_from_slice(&pgno.to_le_bytes());
    page[8..10].copy_from_slice(&0u16.to_le_bytes());
    page[10..12].copy_from_slice(&flags.bits().to_le_bytes());
    page[12..14].copy_from_slice(&(PAGE_HEADER_SIZE as u16).to_le_bytes());
    page[14..16].copy_from_slice(&(page_size as u16).to_le_bytes());
}

/// Calculate the total space needed for a leaf node including its pointer.
///
/// Returns `2 + even(NODE_HEADER_SIZE + key.len() + data.len())`.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::node::leaf_size;
/// assert_eq!(leaf_size(b"key", b"val"), 2 + 14);
/// ```
#[must_use]
pub fn leaf_size(key: &[u8], data: &[u8]) -> usize {
    2 + even(NODE_HEADER_SIZE + key.len() + data.len())
}

/// Calculate the total space needed for a branch node including its pointer.
///
/// Returns `2 + even(NODE_HEADER_SIZE + key.len())`.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::node::branch_size;
/// assert_eq!(branch_size(b"key"), 2 + 12);
/// ```
#[must_use]
pub fn branch_size(key: &[u8]) -> usize {
    2 + even(NODE_HEADER_SIZE + key.len())
}

/// Add a node at position `idx` in the page.
///
/// For branch pages the `pgno` argument encodes the child page number into
/// the node header. For leaf pages `data` is stored inline and `flags`
/// records node-level flags such as `BIGDATA` or `SUBDATA`.
///
/// # Errors
///
/// Returns [`Error::PageFull`] if there is not enough space on the page
/// for the new pointer and node data.
pub fn node_add(
    page: &mut [u8],
    _page_size: usize,
    idx: usize,
    key: &[u8],
    data: &[u8],
    pgno: u64,
    flags: NodeFlags,
) -> Result<()> {
    let is_branch = {
        let pf = u16::from_le_bytes([page[10], page[11]]);
        PageFlags::from_bits_truncate(pf).contains(PageFlags::BRANCH)
    };

    let lower = u16::from_le_bytes([page[12], page[13]]) as usize;
    let upper = u16::from_le_bytes([page[14], page[15]]) as usize;

    let node_size = if is_branch {
        NODE_HEADER_SIZE + key.len()
    } else {
        NODE_HEADER_SIZE + key.len() + data.len()
    };

    // Need 2 bytes for the pointer + node_size bytes for the node
    if upper - lower < 2 + node_size {
        return Err(Error::PageFull);
    }

    let num_keys = (lower - PAGE_HEADER_SIZE) / 2;

    // Shift pointers at positions >= idx to make room
    if idx < num_keys {
        let src = PAGE_HEADER_SIZE + idx * 2;
        let dst = PAGE_HEADER_SIZE + (idx + 1) * 2;
        let len = (num_keys - idx) * 2;
        page.copy_within(src..src + len, dst);
    }

    // New node position: just below current upper
    let node_offset = upper - node_size;

    // Write pointer at idx
    let ptr_offset = PAGE_HEADER_SIZE + idx * 2;
    page[ptr_offset..ptr_offset + 2].copy_from_slice(&(node_offset as u16).to_le_bytes());

    // Write node header
    if is_branch {
        // Encode pgno: lo = pgno & 0xFFFF, hi = (pgno >> 16) & 0xFFFF,
        // flags = (pgno >> 32) & 0xFFFF
        let lo = (pgno & 0xFFFF) as u16;
        let hi = ((pgno >> 16) & 0xFFFF) as u16;
        let flags_raw = ((pgno >> 32) & 0xFFFF) as u16;
        page[node_offset..node_offset + 2].copy_from_slice(&lo.to_le_bytes());
        page[node_offset + 2..node_offset + 4].copy_from_slice(&hi.to_le_bytes());
        page[node_offset + 4..node_offset + 6].copy_from_slice(&flags_raw.to_le_bytes());
    } else {
        // Encode data size
        let ds = data.len() as u32;
        let lo = (ds & 0xFFFF) as u16;
        let hi = ((ds >> 16) & 0xFFFF) as u16;
        page[node_offset..node_offset + 2].copy_from_slice(&lo.to_le_bytes());
        page[node_offset + 2..node_offset + 4].copy_from_slice(&hi.to_le_bytes());
        page[node_offset + 4..node_offset + 6].copy_from_slice(&flags.bits().to_le_bytes());
    }
    // Write key size and key
    let ks = key.len() as u16;
    page[node_offset + 6..node_offset + 8].copy_from_slice(&ks.to_le_bytes());
    page[node_offset + NODE_HEADER_SIZE..node_offset + NODE_HEADER_SIZE + key.len()]
        .copy_from_slice(key);

    // Write data (leaf only)
    if !is_branch {
        let data_start = node_offset + NODE_HEADER_SIZE + key.len();
        page[data_start..data_start + data.len()].copy_from_slice(data);
    }

    // Update lower and upper
    let new_lower = (lower + 2) as u16;
    let new_upper = node_offset as u16;
    page[12..14].copy_from_slice(&new_lower.to_le_bytes());
    page[14..16].copy_from_slice(&new_upper.to_le_bytes());

    Ok(())
}

/// Remove the node at position `idx` from the page.
///
/// This compacts the node data area so that space is reclaimed. All node
/// pointers that referenced data below the deleted node are adjusted
/// accordingly.
pub fn node_del(page: &mut [u8], _page_size: usize, idx: usize) {
    let lower = u16::from_le_bytes([page[12], page[13]]) as usize;
    let upper = u16::from_le_bytes([page[14], page[15]]) as usize;
    let num_keys = (lower - PAGE_HEADER_SIZE) / 2;

    let is_branch = {
        let pf = u16::from_le_bytes([page[10], page[11]]);
        PageFlags::from_bits_truncate(pf).contains(PageFlags::BRANCH)
    };

    // Get the deleted node's offset and size
    let del_ptr_pos = PAGE_HEADER_SIZE + idx * 2;
    let del_offset = u16::from_le_bytes([page[del_ptr_pos], page[del_ptr_pos + 1]]) as usize;

    // Calculate deleted node's size
    let node_ksize = u16::from_le_bytes([page[del_offset + 6], page[del_offset + 7]]) as usize;
    let node_size = if is_branch {
        NODE_HEADER_SIZE + node_ksize
    } else {
        let lo = u16::from_le_bytes([page[del_offset], page[del_offset + 1]]) as u32;
        let hi = u16::from_le_bytes([page[del_offset + 2], page[del_offset + 3]]) as u32;
        let data_size = (lo | (hi << 16)) as usize;
        NODE_HEADER_SIZE + node_ksize + data_size
    };

    // Remove pointer by shifting left
    if idx + 1 < num_keys {
        let src = PAGE_HEADER_SIZE + (idx + 1) * 2;
        let dst = PAGE_HEADER_SIZE + idx * 2;
        let len = (num_keys - idx - 1) * 2;
        page.copy_within(src..src + len, dst);
    }

    let new_lower = lower - 2;
    let new_num_keys = num_keys - 1;

    // Compact node data: move nodes below del_offset up by node_size
    if del_offset > upper {
        let src = upper;
        let dst = upper + node_size;
        let len = del_offset - upper;
        page.copy_within(src..src + len, dst);

        // Adjust all pointers that pointed to data in the moved range
        for i in 0..new_num_keys {
            let pp = PAGE_HEADER_SIZE + i * 2;
            let ptr_val = u16::from_le_bytes([page[pp], page[pp + 1]]) as usize;
            if ptr_val < del_offset {
                let new_val = (ptr_val + node_size) as u16;
                page[pp..pp + 2].copy_from_slice(&new_val.to_le_bytes());
            }
        }
    }

    let new_upper = upper + node_size;
    page[12..14].copy_from_slice(&(new_lower as u16).to_le_bytes());
    page[14..16].copy_from_slice(&(new_upper as u16).to_le_bytes());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::Page;

    /// Helper: create a fresh page buffer of the given size with the specified
    /// flags.
    fn fresh_page(page_size: usize, flags: PageFlags) -> Vec<u8> {
        let mut buf = vec![0u8; page_size];
        init_page(&mut buf, 0, flags, page_size);
        buf
    }

    /// Read back a node's key from a page at the given index.
    fn read_key(buf: &[u8], idx: usize) -> &[u8] {
        let page = Page::from_raw(buf);
        let node = page.node(idx);
        node.key()
    }

    /// Read back a node's data from a leaf page at the given index.
    fn read_data(buf: &[u8], idx: usize) -> &[u8] {
        let page = Page::from_raw(buf);
        let node = page.node(idx);
        node.node_data()
    }

    /// Read back a node's child pgno from a branch page at the given index.
    fn read_child_pgno(buf: &[u8], idx: usize) -> u64 {
        let page = Page::from_raw(buf);
        let node = page.node(idx);
        node.child_pgno()
    }

    #[test]
    fn test_should_init_page_header() {
        let mut buf = vec![0u8; 4096];
        init_page(&mut buf, 99, PageFlags::LEAF | PageFlags::DIRTY, 4096);
        let page = Page::from_raw(&buf);

        assert_eq!(page.pgno(), 99);
        assert!(page.flags().contains(PageFlags::LEAF));
        assert!(page.flags().contains(PageFlags::DIRTY));
        assert_eq!(page.lower(), PAGE_HEADER_SIZE as u16);
        assert_eq!(page.upper(), 4096);
        assert_eq!(page.num_keys(), 0);
        assert_eq!(page.pad(), 0);
    }

    #[test]
    fn test_should_calculate_leaf_size() {
        // NODE_HEADER_SIZE(8) + 3 + 4 = 15, even(15) = 16, + 2 = 18
        assert_eq!(leaf_size(b"abc", b"defg"), 18);
        // NODE_HEADER_SIZE(8) + 0 + 0 = 8, even(8) = 8, + 2 = 10
        assert_eq!(leaf_size(b"", b""), 10);
        // NODE_HEADER_SIZE(8) + 1 + 0 = 9, even(9) = 10, + 2 = 12
        assert_eq!(leaf_size(b"x", b""), 12);
    }

    #[test]
    fn test_should_calculate_branch_size() {
        // NODE_HEADER_SIZE(8) + 3 = 11, even(11) = 12, + 2 = 14
        assert_eq!(branch_size(b"abc"), 14);
        // NODE_HEADER_SIZE(8) + 0 = 8, even(8) = 8, + 2 = 10
        assert_eq!(branch_size(b""), 10);
    }

    #[test]
    fn test_should_add_and_read_leaf_nodes() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        // Add three nodes
        node_add(
            &mut buf,
            page_size,
            0,
            b"key1",
            b"val1",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add node 0: {e}"))
        .ok();
        node_add(
            &mut buf,
            page_size,
            1,
            b"key2",
            b"val2",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add node 1: {e}"))
        .ok();
        node_add(
            &mut buf,
            page_size,
            2,
            b"key3",
            b"val3",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add node 2: {e}"))
        .ok();

        let page = Page::from_raw(&buf);
        assert_eq!(page.num_keys(), 3);

        assert_eq!(read_key(&buf, 0), b"key1");
        assert_eq!(read_key(&buf, 1), b"key2");
        assert_eq!(read_key(&buf, 2), b"key3");
        assert_eq!(read_data(&buf, 0), b"val1");
        assert_eq!(read_data(&buf, 1), b"val2");
        assert_eq!(read_data(&buf, 2), b"val3");
    }

    #[test]
    fn test_should_roundtrip_branch_child_pgno() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::BRANCH);

        let child_pgno: u64 = 0x0000_1234_5678_9ABC;
        node_add(
            &mut buf,
            page_size,
            0,
            b"brk",
            &[],
            child_pgno,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add branch node: {e}"))
        .ok();

        let page = Page::from_raw(&buf);
        assert_eq!(page.num_keys(), 1);
        assert_eq!(read_key(&buf, 0), b"brk");
        assert_eq!(read_child_pgno(&buf, 0), child_pgno);
    }

    #[test]
    fn test_should_roundtrip_large_branch_pgno() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::BRANCH);

        // Use a pgno that exercises all 48 bits
        let child_pgno: u64 = 0x0000_ABCD_1234_5678;
        node_add(
            &mut buf,
            page_size,
            0,
            b"",
            &[],
            child_pgno,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add branch node: {e}"))
        .ok();

        assert_eq!(read_child_pgno(&buf, 0), child_pgno);
    }

    #[test]
    fn test_should_delete_middle_node() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        for (i, (k, v)) in [
            (b"aaa" as &[u8], b"111" as &[u8]),
            (b"bbb", b"222"),
            (b"ccc", b"333"),
        ]
        .iter()
        .enumerate()
        {
            node_add(&mut buf, page_size, i, k, v, 0, NodeFlags::empty())
                .map_err(|e| format!("add node {i}: {e}"))
                .ok();
        }

        // Delete the middle node (index 1 = "bbb")
        node_del(&mut buf, page_size, 1);

        let page = Page::from_raw(&buf);
        assert_eq!(page.num_keys(), 2);
        assert_eq!(read_key(&buf, 0), b"aaa");
        assert_eq!(read_data(&buf, 0), b"111");
        assert_eq!(read_key(&buf, 1), b"ccc");
        assert_eq!(read_data(&buf, 1), b"333");
    }

    #[test]
    fn test_should_delete_first_node() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        node_add(
            &mut buf,
            page_size,
            0,
            b"first",
            b"1st",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add: {e}"))
        .ok();
        node_add(
            &mut buf,
            page_size,
            1,
            b"second",
            b"2nd",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add: {e}"))
        .ok();

        node_del(&mut buf, page_size, 0);

        let page = Page::from_raw(&buf);
        assert_eq!(page.num_keys(), 1);
        assert_eq!(read_key(&buf, 0), b"second");
        assert_eq!(read_data(&buf, 0), b"2nd");
    }

    #[test]
    fn test_should_delete_last_node() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        node_add(
            &mut buf,
            page_size,
            0,
            b"first",
            b"1st",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add: {e}"))
        .ok();
        node_add(
            &mut buf,
            page_size,
            1,
            b"second",
            b"2nd",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("add: {e}"))
        .ok();

        node_del(&mut buf, page_size, 1);

        let page = Page::from_raw(&buf);
        assert_eq!(page.num_keys(), 1);
        assert_eq!(read_key(&buf, 0), b"first");
        assert_eq!(read_data(&buf, 0), b"1st");
    }

    #[test]
    fn test_should_return_page_full_when_no_space() {
        // Use a tiny page (48 bytes) so it fills up quickly
        let page_size = 48;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        // First node: need ptr(2) + node(8+3+3=14) = 16.
        // Free = 48 - 16 = 32. Fits.
        let r1 = node_add(
            &mut buf,
            page_size,
            0,
            b"abc",
            b"xyz",
            0,
            NodeFlags::empty(),
        );
        assert!(r1.is_ok(), "first node should fit");

        // After: lower=18, upper=34, free=16. Need 16 for second node. Fits.
        let r2 = node_add(
            &mut buf,
            page_size,
            1,
            b"def",
            b"uvw",
            0,
            NodeFlags::empty(),
        );
        assert!(r2.is_ok(), "second node should fit");

        // After: lower=20, upper=20, free=0. Third node needs 16. PageFull.
        let r3 = node_add(
            &mut buf,
            page_size,
            2,
            b"ghi",
            b"rst",
            0,
            NodeFlags::empty(),
        );
        assert!(
            matches!(r3, Err(Error::PageFull)),
            "expected PageFull, got {r3:?}",
        );
    }

    #[test]
    fn test_should_insert_at_beginning() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        // Add two nodes, then insert one at index 0
        node_add(&mut buf, page_size, 0, b"bbb", b"2", 0, NodeFlags::empty())
            .map_err(|e| format!("{e}"))
            .ok();
        node_add(&mut buf, page_size, 1, b"ccc", b"3", 0, NodeFlags::empty())
            .map_err(|e| format!("{e}"))
            .ok();
        node_add(&mut buf, page_size, 0, b"aaa", b"1", 0, NodeFlags::empty())
            .map_err(|e| format!("{e}"))
            .ok();

        let page = Page::from_raw(&buf);
        assert_eq!(page.num_keys(), 3);
        assert_eq!(read_key(&buf, 0), b"aaa");
        assert_eq!(read_key(&buf, 1), b"bbb");
        assert_eq!(read_key(&buf, 2), b"ccc");
    }

    #[test]
    fn test_should_insert_in_middle() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        node_add(&mut buf, page_size, 0, b"aaa", b"1", 0, NodeFlags::empty())
            .map_err(|e| format!("{e}"))
            .ok();
        node_add(&mut buf, page_size, 1, b"ccc", b"3", 0, NodeFlags::empty())
            .map_err(|e| format!("{e}"))
            .ok();
        // Insert "bbb" at index 1 (between aaa and ccc)
        node_add(&mut buf, page_size, 1, b"bbb", b"2", 0, NodeFlags::empty())
            .map_err(|e| format!("{e}"))
            .ok();

        assert_eq!(read_key(&buf, 0), b"aaa");
        assert_eq!(read_key(&buf, 1), b"bbb");
        assert_eq!(read_key(&buf, 2), b"ccc");
    }

    #[test]
    fn test_should_add_and_delete_all_nodes() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        node_add(
            &mut buf,
            page_size,
            0,
            b"only",
            b"one",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("{e}"))
        .ok();

        node_del(&mut buf, page_size, 0);

        let page = Page::from_raw(&buf);
        assert_eq!(page.num_keys(), 0);
        assert_eq!(page.lower(), PAGE_HEADER_SIZE as u16);
        assert_eq!(page.upper(), page_size as u16);
    }

    #[test]
    fn test_should_preserve_node_flags_on_leaf() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        node_add(
            &mut buf,
            page_size,
            0,
            b"sub",
            b"dbdata",
            0,
            NodeFlags::SUBDATA,
        )
        .map_err(|e| format!("{e}"))
        .ok();

        let page = Page::from_raw(&buf);
        let node = page.node(0);
        assert!(node.flags().contains(NodeFlags::SUBDATA));
    }

    #[test]
    fn test_should_handle_empty_key_branch_node() {
        let page_size = 4096;
        let mut buf = fresh_page(page_size, PageFlags::BRANCH);

        // Index 0 in a branch page typically has an empty key
        let child: u64 = 42;
        node_add(&mut buf, page_size, 0, b"", &[], child, NodeFlags::empty())
            .map_err(|e| format!("{e}"))
            .ok();

        assert_eq!(read_key(&buf, 0), b"");
        assert_eq!(read_child_pgno(&buf, 0), child);
    }

    #[test]
    fn test_should_delete_and_reuse_space() {
        let page_size = 256;
        let mut buf = fresh_page(page_size, PageFlags::LEAF);

        // Add some nodes
        node_add(
            &mut buf,
            page_size,
            0,
            b"aaaa",
            b"1111",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("{e}"))
        .ok();
        node_add(
            &mut buf,
            page_size,
            1,
            b"bbbb",
            b"2222",
            0,
            NodeFlags::empty(),
        )
        .map_err(|e| format!("{e}"))
        .ok();

        // Record upper before delete
        let upper_before = u16::from_le_bytes([buf[14], buf[15]]) as usize;

        // Delete first, upper should increase (space reclaimed)
        node_del(&mut buf, page_size, 0);
        let upper_after = u16::from_le_bytes([buf[14], buf[15]]) as usize;
        assert!(
            upper_after > upper_before,
            "upper should increase after delete"
        );

        // We should be able to add a new node in the reclaimed space
        let result = node_add(
            &mut buf,
            page_size,
            1,
            b"cccc",
            b"3333",
            0,
            NodeFlags::empty(),
        );
        assert!(result.is_ok(), "should have space after delete");
    }
}
