//! B+ tree page split, merge, and rebalance operations.
//!
//! This module implements the core B+ tree mutation algorithms: inserting keys
//! with automatic page splitting, deleting keys with rebalancing, and the
//! supporting tree-walk logic. These are free functions that operate on an
//! [`RwTransaction`] to keep the write-path logic separated from the
//! transaction lifecycle code in [`crate::write`].

use std::cmp::Ordering;

use crate::{
    cmp::CmpFn,
    error::{Error, Result},
    node::{init_page, leaf_size, node_add, node_del},
    page::Page,
    types::*,
    write::RwTransaction,
};

// ---------------------------------------------------------------------------
// Tree path
// ---------------------------------------------------------------------------

/// One level of the path through the B+ tree from root to leaf.
///
/// `pgno` is the page number at this level, and `idx` is the child index
/// (or insertion index at the leaf level).
#[derive(Debug, Clone, Copy)]
struct PathLevel {
    pgno: u64,
    idx: usize,
}

/// A path through the B+ tree from root (index 0) to leaf (last index).
type TreePath = Vec<PathLevel>;

// ---------------------------------------------------------------------------
// Collected entry types
// ---------------------------------------------------------------------------

/// A leaf entry collected during a split.
struct LeafEntry {
    key: Vec<u8>,
    data: Vec<u8>,
    flags: NodeFlags,
}

/// A branch entry collected during a split.
struct BranchEntry {
    key: Vec<u8>,
    child_pgno: u64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Insert a key/value pair, splitting pages as needed.
///
/// This is the primary write-path entry point. It walks the tree from root
/// to the target leaf (COW-ing each page along the way), attempts to insert,
/// and splits pages upward when they overflow.
///
/// # Errors
///
/// - [`Error::BadValSize`] if key is empty or exceeds the maximum key size
/// - [`Error::KeyExist`] if `NO_OVERWRITE` is set and the key already exists
/// - [`Error::MapFull`] if no more pages can be allocated
pub fn cursor_put(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    key: &[u8],
    data: &[u8],
    flags: WriteFlags,
) -> Result<()> {
    if key.is_empty() || key.len() > txn.env.max_key_size {
        return Err(Error::BadValSize);
    }

    let db = *txn.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
    let page_size = txn.env.page_size;

    if db.root == P_INVALID {
        // Empty database -- create a new root leaf page.
        let (root_pgno, mut root_buf) = txn.page_alloc()?;
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
        txn.dirty.insert(root_pgno, root_buf);

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = root_pgno;
        db_mut.depth = 1;
        db_mut.leaf_pages = 1;
        db_mut.entries = 1;
        txn.db_dirty[dbi as usize] = true;
        return Ok(());
    }

    // Walk the tree from root to leaf, COW-ing each page.
    let cmp = txn.env.get_cmp(dbi)?;
    let (path, leaf_pgno) = walk_and_touch(txn, db.root, key, &**cmp)?;

    // Try to insert on the leaf page.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let nkeys = leaf_page.num_keys();

    // Find insertion point and check for duplicate.
    let (insert_idx, exact) = page_node_search(&leaf_page, key, &**cmp);
    let mut overwrite = false;

    if exact && insert_idx < nkeys {
        if flags.contains(WriteFlags::NO_OVERWRITE) {
            return Err(Error::KeyExist);
        }
        overwrite = true;
    }

    // If overwriting, delete the old node first.
    if overwrite {
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, insert_idx);
    }

    // Attempt insertion.
    let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
    let add_result = node_add(
        buf.as_mut_slice(),
        page_size,
        insert_idx,
        key,
        data,
        0,
        NodeFlags::empty(),
    );

    match add_result {
        Ok(()) => {
            // Success -- update database metadata.
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = path[0].pgno;
            if !overwrite {
                db_mut.entries += 1;
            }
            txn.db_dirty[dbi as usize] = true;
            Ok(())
        }
        Err(Error::PageFull) => {
            // The leaf is full -- split and insert.
            split_and_insert(
                txn,
                dbi,
                &path,
                key,
                data,
                NodeFlags::empty(),
                insert_idx,
                &**cmp,
            )?;
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            if !overwrite {
                db_mut.entries += 1;
            }
            txn.db_dirty[dbi as usize] = true;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Delete a key from the specified database, rebalancing as needed.
///
/// # Errors
///
/// - [`Error::NotFound`] if the key does not exist
/// - [`Error::BadDbi`] if the database handle is invalid
pub fn cursor_del(txn: &mut RwTransaction<'_>, dbi: u32, key: &[u8]) -> Result<()> {
    let db = *txn.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
    if db.root == P_INVALID {
        return Err(Error::NotFound);
    }

    let cmp = txn.env.get_cmp(dbi)?;
    let page_size = txn.env.page_size;

    // Walk the tree, COW-ing each page.
    let (path, leaf_pgno) = walk_and_touch(txn, db.root, key, &**cmp)?;

    // Verify exact match.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let (idx, exact) = page_node_search(&leaf_page, key, &**cmp);

    if !exact || idx >= leaf_page.num_keys() {
        return Err(Error::NotFound);
    }

    // Verify key really matches (page_node_search uses the comparator).
    {
        let node = leaf_page.node(idx);
        if cmp(key, node.key()) != Ordering::Equal {
            return Err(Error::NotFound);
        }
    }

    // Delete the node.
    let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
    node_del(buf.as_mut_slice(), page_size, idx);

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.entries = db_mut.entries.saturating_sub(1);
    txn.db_dirty[dbi as usize] = true;

    // Check if the leaf is now empty.
    let leaf_page = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?.as_page();

    if leaf_page.num_keys() == 0 {
        if path.len() == 1 {
            // Root leaf is empty -- database is now empty.
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = P_INVALID;
            db_mut.depth = 0;
            db_mut.leaf_pages = 0;
            return Ok(());
        }
        // Non-root empty leaf -- remove from parent and rebalance.
        remove_from_parent(txn, dbi, &path)?;
        return Ok(());
    }

    // Page is not empty -- check if rebalancing is needed.
    let fill = leaf_page.used_space() * 10 / (page_size - PAGE_HEADER_SIZE);
    if path.len() > 1 && fill < FILL_THRESHOLD / 100 {
        rebalance(txn, dbi, &path)?;
    } else {
        // Update root in db metadata (it may have changed due to COW).
        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = path[0].pgno;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal: tree walk
// ---------------------------------------------------------------------------

/// Walk from root to the leaf containing `key`, COW-ing each page.
///
/// Returns the path from root to leaf and the leaf page number. Each page
/// along the path is guaranteed to be in the dirty list after this call.
fn walk_and_touch(
    txn: &mut RwTransaction<'_>,
    root_pgno: u64,
    key: &[u8],
    cmp: &CmpFn,
) -> Result<(TreePath, u64)> {
    let page_size = txn.env.page_size;
    let mut path = TreePath::new();

    // Touch the root page.
    let mut current_pgno = txn.page_touch(root_pgno)?;

    loop {
        let page = read_dirty_page(txn, current_pgno, page_size)?;

        if page.is_leaf() {
            path.push(PathLevel {
                pgno: current_pgno,
                idx: 0,
            });
            return Ok((path, current_pgno));
        }

        if !page.is_branch() {
            return Err(Error::Corrupted);
        }

        // Search for the child to descend into.
        let (idx, exact) = page_node_search(&page, key, cmp);
        let child_idx = if exact {
            idx
        } else if idx > 0 {
            idx - 1
        } else {
            0
        };

        path.push(PathLevel {
            pgno: current_pgno,
            idx: child_idx,
        });

        // Read child pgno from the branch node.
        let child_pgno = page.node(child_idx).child_pgno();

        // Touch (COW) the child page.
        let new_child_pgno = txn.page_touch(child_pgno)?;

        // If the child page was COW'd to a new location, update the parent's
        // branch node pointer.
        if new_child_pgno != child_pgno {
            update_branch_child(txn, current_pgno, child_idx, new_child_pgno, page_size)?;
        }

        current_pgno = new_child_pgno;
    }
}

/// Read a page from the dirty list. The page is guaranteed to be there because
/// `walk_and_touch` COWs every page it visits.
fn read_dirty_page<'a>(
    txn: &'a RwTransaction<'_>,
    pgno: u64,
    page_size: usize,
) -> Result<Page<'a>> {
    // First check the dirty list.
    if let Some(buf) = txn.dirty.find(pgno) {
        return Ok(buf.as_page());
    }
    // Fall back to mmap.
    let ptr = txn.env.get_page(pgno)?;
    let slice = unsafe { std::slice::from_raw_parts(ptr, page_size) };
    Ok(Page::from_raw(slice))
}

/// Update a branch node's child page number in-place.
///
/// This is needed when a child page is COW'd to a new page number.
fn update_branch_child(
    txn: &mut RwTransaction<'_>,
    branch_pgno: u64,
    child_idx: usize,
    new_child_pgno: u64,
    page_size: usize,
) -> Result<()> {
    let buf = txn.dirty.find_mut(branch_pgno).ok_or(Error::Corrupted)?;
    let page = Page::from_raw(buf.as_slice());
    let node_offset = page.ptr_at(child_idx) as usize;

    // Read the existing key from the node.
    let key_size = u16::from_le_bytes([
        buf.as_slice()[node_offset + 6],
        buf.as_slice()[node_offset + 7],
    ]) as usize;
    let key: Vec<u8> = buf.as_slice()
        [node_offset + NODE_HEADER_SIZE..node_offset + NODE_HEADER_SIZE + key_size]
        .to_vec();

    // Delete the old node and re-add with the new child pgno.
    node_del(buf.as_mut_slice(), page_size, child_idx);
    node_add(
        buf.as_mut_slice(),
        page_size,
        child_idx,
        &key,
        &[],
        new_child_pgno,
        NodeFlags::empty(),
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal: page search (standalone, no cursor needed)
// ---------------------------------------------------------------------------

/// Binary search for a key within a single page.
///
/// Returns `(index, exact_match)` similar to `Cursor::node_search`.
fn page_node_search(page: &Page<'_>, key: &[u8], cmp: &CmpFn) -> (usize, bool) {
    let nkeys = page.num_keys();
    if nkeys == 0 {
        return (0, false);
    }

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

// ---------------------------------------------------------------------------
// Internal: split
// ---------------------------------------------------------------------------

/// Split a full leaf page and insert the new entry, propagating the
/// separator key up through the tree. Updates `txn.dbs[dbi].root` and
/// `txn.dbs[dbi].depth` as needed.
#[allow(clippy::too_many_arguments)]
fn split_and_insert(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    key: &[u8],
    data: &[u8],
    flags: NodeFlags,
    insert_idx: usize,
    _cmp: &CmpFn,
) -> Result<()> {
    let page_size = txn.env.page_size;
    let leaf_level = path.len() - 1;
    let leaf_pgno = path[leaf_level].pgno;

    // Collect all existing entries from the full leaf page.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let nkeys = leaf_page.num_keys();

    let mut entries: Vec<LeafEntry> = Vec::with_capacity(nkeys + 1);
    for i in 0..nkeys {
        let node = leaf_page.node(i);
        entries.push(LeafEntry {
            key: node.key().to_vec(),
            data: node.node_data().to_vec(),
            flags: node.flags(),
        });
    }

    // Insert the new entry at the correct position.
    entries.insert(
        insert_idx,
        LeafEntry {
            key: key.to_vec(),
            data: data.to_vec(),
            flags,
        },
    );

    let total = entries.len();
    let split_idx = total / 2;

    // Allocate right sibling leaf.
    let (right_pgno, mut right_buf) = txn.page_alloc()?;
    init_page(
        right_buf.as_mut_slice(),
        right_pgno,
        PageFlags::LEAF | PageFlags::DIRTY,
        page_size,
    );

    // Reinitialize the left leaf page (reuse existing pgno).
    let left_pgno = leaf_pgno;
    let left_buf = txn.dirty.find_mut(left_pgno).ok_or(Error::Corrupted)?;
    init_page(
        left_buf.as_mut_slice(),
        left_pgno,
        PageFlags::LEAF | PageFlags::DIRTY,
        page_size,
    );

    // Populate the left page.
    for (i, entry) in entries[..split_idx].iter().enumerate() {
        node_add(
            left_buf.as_mut_slice(),
            page_size,
            i,
            &entry.key,
            &entry.data,
            0,
            entry.flags,
        )?;
    }

    // Populate the right page.
    for (i, entry) in entries[split_idx..].iter().enumerate() {
        node_add(
            right_buf.as_mut_slice(),
            page_size,
            i,
            &entry.key,
            &entry.data,
            0,
            entry.flags,
        )?;
    }

    // The separator key is the first key on the right page.
    let sep_key = entries[split_idx].key.clone();

    txn.dirty.insert(right_pgno, right_buf);

    // Update leaf page count.
    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.leaf_pages += 1;

    // Propagate separator up to the parent.
    insert_separator(txn, dbi, path, leaf_level, &sep_key, left_pgno, right_pgno)?;

    Ok(())
}

/// Insert a separator key into the parent branch page, splitting the
/// branch if necessary. If the node being split is the root, a new root
/// is created.
fn insert_separator(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    split_level: usize,
    sep_key: &[u8],
    left_pgno: u64,
    right_pgno: u64,
) -> Result<()> {
    let page_size = txn.env.page_size;

    if split_level == 0 {
        // The split page was the root -- create a new root branch.
        let (new_root_pgno, mut new_root_buf) = txn.page_alloc()?;
        init_page(
            new_root_buf.as_mut_slice(),
            new_root_pgno,
            PageFlags::BRANCH | PageFlags::DIRTY,
            page_size,
        );

        // First child (empty key, leftmost pointer).
        node_add(
            new_root_buf.as_mut_slice(),
            page_size,
            0,
            &[],
            &[],
            left_pgno,
            NodeFlags::empty(),
        )?;

        // Second child with separator key.
        node_add(
            new_root_buf.as_mut_slice(),
            page_size,
            1,
            sep_key,
            &[],
            right_pgno,
            NodeFlags::empty(),
        )?;

        txn.dirty.insert(new_root_pgno, new_root_buf);

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = new_root_pgno;
        db_mut.depth += 1;
        db_mut.branch_pages += 1;

        return Ok(());
    }

    // Insert into existing parent branch.
    let parent_level = split_level - 1;
    let parent_pgno = path[parent_level].pgno;
    let parent_child_idx = path[parent_level].idx;

    // The separator goes after the current child index.
    let insert_idx = parent_child_idx + 1;

    let buf = txn.dirty.find_mut(parent_pgno).ok_or(Error::Corrupted)?;
    let add_result = node_add(
        buf.as_mut_slice(),
        page_size,
        insert_idx,
        sep_key,
        &[],
        right_pgno,
        NodeFlags::empty(),
    );

    match add_result {
        Ok(()) => {
            // Update root in db metadata.
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = path[0].pgno;
            Ok(())
        }
        Err(Error::PageFull) => {
            // Parent branch is full -- split it.
            split_branch(
                txn,
                dbi,
                path,
                parent_level,
                sep_key,
                right_pgno,
                insert_idx,
            )?;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Split a full branch page and insert a new separator + child pointer.
fn split_branch(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    branch_level: usize,
    sep_key: &[u8],
    child_pgno: u64,
    insert_idx: usize,
) -> Result<()> {
    let page_size = txn.env.page_size;
    let branch_pgno = path[branch_level].pgno;

    // Collect all existing branch entries.
    let branch_buf = txn.dirty.find(branch_pgno).ok_or(Error::Corrupted)?;
    let branch_page = branch_buf.as_page();
    let nkeys = branch_page.num_keys();

    let mut entries: Vec<BranchEntry> = Vec::with_capacity(nkeys + 1);
    for i in 0..nkeys {
        let node = branch_page.node(i);
        entries.push(BranchEntry {
            key: node.key().to_vec(),
            child_pgno: node.child_pgno(),
        });
    }

    // Insert the new entry.
    entries.insert(
        insert_idx,
        BranchEntry {
            key: sep_key.to_vec(),
            child_pgno,
        },
    );

    let total = entries.len();
    let split_idx = total / 2;

    // The separator for the parent is the key at the split point.
    // For branch splits, the key at split_idx is promoted (not duplicated).
    let promoted_key = entries[split_idx].key.clone();

    // Allocate right sibling branch.
    let (right_pgno, mut right_buf) = txn.page_alloc()?;
    init_page(
        right_buf.as_mut_slice(),
        right_pgno,
        PageFlags::BRANCH | PageFlags::DIRTY,
        page_size,
    );

    // Reinitialize the left branch page.
    let left_pgno = branch_pgno;
    let left_buf = txn.dirty.find_mut(left_pgno).ok_or(Error::Corrupted)?;
    init_page(
        left_buf.as_mut_slice(),
        left_pgno,
        PageFlags::BRANCH | PageFlags::DIRTY,
        page_size,
    );

    // Populate left branch: entries [0..split_idx].
    for (i, entry) in entries[..split_idx].iter().enumerate() {
        node_add(
            left_buf.as_mut_slice(),
            page_size,
            i,
            &entry.key,
            &[],
            entry.child_pgno,
            NodeFlags::empty(),
        )?;
    }

    // Populate right branch: entries [split_idx..].
    // The first entry on the right gets an empty key (leftmost child pointer).
    for (i, entry) in entries[split_idx..].iter().enumerate() {
        let key = if i == 0 { &[] as &[u8] } else { &entry.key };
        node_add(
            right_buf.as_mut_slice(),
            page_size,
            i,
            key,
            &[],
            entry.child_pgno,
            NodeFlags::empty(),
        )?;
    }

    txn.dirty.insert(right_pgno, right_buf);

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.branch_pages += 1;

    // Propagate the promoted key up.
    insert_separator(
        txn,
        dbi,
        path,
        branch_level,
        &promoted_key,
        left_pgno,
        right_pgno,
    )
}

// ---------------------------------------------------------------------------
// Internal: rebalance / merge
// ---------------------------------------------------------------------------

/// Remove an empty child from its parent branch. If the parent becomes
/// a single-child branch root, collapse the tree by one level.
fn remove_from_parent(txn: &mut RwTransaction<'_>, dbi: u32, path: &TreePath) -> Result<()> {
    let page_size = txn.env.page_size;

    // The leaf is at path[leaf_level], its parent is at path[leaf_level - 1].
    let leaf_level = path.len() - 1;
    let parent_level = leaf_level - 1;
    let parent_pgno = path[parent_level].pgno;
    let child_idx = path[parent_level].idx;

    let buf = txn.dirty.find_mut(parent_pgno).ok_or(Error::Corrupted)?;
    node_del(buf.as_mut_slice(), page_size, child_idx);

    let parent_page = Page::from_raw(buf.as_slice());
    let parent_nkeys = parent_page.num_keys();

    if parent_nkeys == 0 {
        // Parent is now empty.
        if parent_level == 0 {
            // Parent was the root -- database is now empty.
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = P_INVALID;
            db_mut.depth = 0;
            db_mut.leaf_pages = 0;
            db_mut.branch_pages = 0;
        } else {
            // Recursively remove from grandparent.
            let sub_path = path[..=parent_level].to_vec();
            remove_from_parent(txn, dbi, &sub_path)?;
        }
        return Ok(());
    }

    // If parent has exactly one child and is the root, collapse.
    if parent_level == 0 && parent_nkeys == 1 {
        let remaining_child_pgno = parent_page.node(0).child_pgno();
        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = remaining_child_pgno;
        db_mut.depth -= 1;
        db_mut.branch_pages = db_mut.branch_pages.saturating_sub(1);
        return Ok(());
    }

    // Parent still has children. If the deleted child was at idx 0, we need
    // to fix the first branch entry to have an empty key.
    if child_idx == 0 && parent_nkeys > 0 {
        let buf = txn.dirty.find_mut(parent_pgno).ok_or(Error::Corrupted)?;
        let page = Page::from_raw(buf.as_slice());
        let first_node = page.node(0);
        let first_key = first_node.key();

        // If the first node already has an empty key, nothing to fix.
        if !first_key.is_empty() {
            let first_child = first_node.child_pgno();
            node_del(buf.as_mut_slice(), page_size, 0);
            node_add(
                buf.as_mut_slice(),
                page_size,
                0,
                &[],
                &[],
                first_child,
                NodeFlags::empty(),
            )?;
        }
    }

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.root = path[0].pgno;

    Ok(())
}

/// Rebalance a leaf page by merging with a sibling.
///
/// This is a simplified rebalance that merges the underfilled leaf with
/// its right (or left) sibling when possible.
fn rebalance(txn: &mut RwTransaction<'_>, dbi: u32, path: &TreePath) -> Result<()> {
    let page_size = txn.env.page_size;
    let leaf_level = path.len() - 1;

    if leaf_level == 0 {
        // Root leaf -- nothing to rebalance against.
        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = path[0].pgno;
        return Ok(());
    }

    let parent_pgno = path[leaf_level - 1].pgno;
    let child_idx = path[leaf_level - 1].idx;
    let leaf_pgno = path[leaf_level].pgno;

    // Read parent to find sibling.
    let parent_page = read_dirty_page(txn, parent_pgno, page_size)?;
    let parent_nkeys = parent_page.num_keys();

    // Pick a sibling: prefer right, fall back to left.
    let (sibling_idx, merge_right) = if child_idx + 1 < parent_nkeys {
        (child_idx + 1, true)
    } else if child_idx > 0 {
        (child_idx - 1, false)
    } else {
        // No sibling -- just update root.
        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = path[0].pgno;
        return Ok(());
    };

    let sibling_pgno_raw = parent_page.node(sibling_idx).child_pgno();
    let sibling_pgno = txn.page_touch(sibling_pgno_raw)?;

    // Update parent if sibling was COW'd.
    if sibling_pgno != sibling_pgno_raw {
        update_branch_child(txn, parent_pgno, sibling_idx, sibling_pgno, page_size)?;
    }

    // Collect all entries from both pages.
    let (left_pgno, right_pgno) = if merge_right {
        (leaf_pgno, sibling_pgno)
    } else {
        (sibling_pgno, leaf_pgno)
    };

    let left_entries = collect_leaf_entries(txn, left_pgno, page_size)?;
    let right_entries = collect_leaf_entries(txn, right_pgno, page_size)?;

    // Check if they can fit on one page.
    let total_size: usize = left_entries
        .iter()
        .chain(right_entries.iter())
        .map(|e| leaf_size(&e.key, &e.data))
        .sum();

    if total_size + PAGE_HEADER_SIZE > page_size {
        // Cannot merge -- just update root.
        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = path[0].pgno;
        return Ok(());
    }

    // Merge into the left page.
    let all_entries: Vec<LeafEntry> = left_entries.into_iter().chain(right_entries).collect();

    let buf = txn.dirty.find_mut(left_pgno).ok_or(Error::Corrupted)?;
    init_page(
        buf.as_mut_slice(),
        left_pgno,
        PageFlags::LEAF | PageFlags::DIRTY,
        page_size,
    );

    for (i, entry) in all_entries.iter().enumerate() {
        node_add(
            buf.as_mut_slice(),
            page_size,
            i,
            &entry.key,
            &entry.data,
            0,
            entry.flags,
        )?;
    }

    // Remove the right page's entry from the parent.
    let remove_idx = if merge_right {
        child_idx + 1
    } else {
        child_idx
    };

    let parent_buf = txn.dirty.find_mut(parent_pgno).ok_or(Error::Corrupted)?;
    node_del(parent_buf.as_mut_slice(), page_size, remove_idx);

    // If we merged left into right and removed the left entry, update the
    // remaining entry to point to left_pgno.
    if !merge_right {
        // The sibling (now at remove_idx position, which shifted) points
        // to left_pgno. Actually, since we merged into sibling_pgno (left),
        // and removed child_idx (the current leaf), the parent now correctly
        // points to sibling_pgno = left_pgno.
    }

    // Fix first branch entry if needed.
    let parent_page = Page::from_raw(parent_buf.as_slice());
    let parent_nkeys_after = parent_page.num_keys();

    if parent_nkeys_after > 0 && remove_idx == 0 {
        let first_node = parent_page.node(0);
        if !first_node.key().is_empty() {
            let first_child = first_node.child_pgno();
            node_del(parent_buf.as_mut_slice(), page_size, 0);
            node_add(
                parent_buf.as_mut_slice(),
                page_size,
                0,
                &[],
                &[],
                first_child,
                NodeFlags::empty(),
            )?;
        }
    }

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.leaf_pages = db_mut.leaf_pages.saturating_sub(1);

    // Check if parent is now a single-child root.
    let parent_page = txn
        .dirty
        .find(parent_pgno)
        .ok_or(Error::Corrupted)?
        .as_page();

    if path.len() > 2 {
        // Parent is not root -- could potentially rebalance further.
        db_mut.root = path[0].pgno;
    } else {
        // Parent is root.
        if parent_page.num_keys() == 1 {
            // Collapse: the single child becomes the new root.
            let remaining = parent_page.node(0).child_pgno();
            db_mut.root = remaining;
            db_mut.depth -= 1;
            db_mut.branch_pages = db_mut.branch_pages.saturating_sub(1);
        } else {
            db_mut.root = parent_pgno;
        }
    }

    Ok(())
}

/// Collect all leaf entries from a dirty page.
fn collect_leaf_entries(
    txn: &RwTransaction<'_>,
    pgno: u64,
    page_size: usize,
) -> Result<Vec<LeafEntry>> {
    let page = read_dirty_page(txn, pgno, page_size)?;
    let nkeys = page.num_keys();
    let mut entries = Vec::with_capacity(nkeys);
    for i in 0..nkeys {
        let node = page.node(i);
        entries.push(LeafEntry {
            key: node.key().to_vec(),
            data: node.node_data().to_vec(),
            flags: node.flags(),
        });
    }
    Ok(entries)
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

    /// Helper to create a test environment with a large enough map.
    fn test_env(dir: &std::path::Path) -> Environment {
        Environment::builder()
            .map_size(64 * 1024 * 1024) // 64 MiB
            .open(dir)
            .expect("open environment")
    }

    #[test]
    fn test_should_split_leaf_on_many_inserts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Insert enough keys to trigger at least one leaf split.
        // With 4096 page size and ~30 byte entries, a page holds ~130 entries.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..200u32 {
                let key = format!("key-{i:06}");
                let val = format!("value-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Verify all keys are readable.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..200u32 {
                let key = format!("key-{i:06}");
                let val = format!("value-{i:06}");
                let got = txn
                    .get(MAIN_DBI, key.as_bytes())
                    .unwrap_or_else(|e| panic!("get {key}: {e}"));
                assert_eq!(got, val.as_bytes(), "mismatch for {key}");
            }
        }
    }

    #[test]
    fn test_should_handle_1000_random_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Use a simple LCG to generate pseudo-random order.
        let mut order: Vec<u32> = (0..1000).collect();
        // Fisher-Yates shuffle with a deterministic seed.
        let mut rng_state: u64 = 12345;
        for i in (1..order.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            order.swap(i, j);
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for &i in &order {
                let key = format!("rkey-{i:06}");
                let val = format!("rval-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Verify all 1000 keys.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..1000u32 {
                let key = format!("rkey-{i:06}");
                let val = format!("rval-{i:06}");
                let got = txn
                    .get(MAIN_DBI, key.as_bytes())
                    .unwrap_or_else(|e| panic!("get {key}: {e}"));
                assert_eq!(got, val.as_bytes(), "mismatch for {key}");
            }
        }
    }

    #[test]
    fn test_should_delete_half_of_entries() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        let count = 500u32;

        // Insert entries.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..count {
                let key = format!("dkey-{i:06}");
                let val = format!("dval-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Delete every other entry.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in (0..count).step_by(2) {
                let key = format!("dkey-{i:06}");
                txn.del(MAIN_DBI, key.as_bytes())
                    .unwrap_or_else(|e| panic!("del {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Verify: even keys gone, odd keys remain.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..count {
                let key = format!("dkey-{i:06}");
                let result = txn.get(MAIN_DBI, key.as_bytes());
                if i % 2 == 0 {
                    assert!(result.is_err(), "deleted key {key} should not be found");
                } else {
                    let val = format!("dval-{i:06}");
                    let got = result.unwrap_or_else(|e| panic!("get {key}: {e}"));
                    assert_eq!(got, val.as_bytes(), "mismatch for {key}");
                }
            }
        }
    }

    #[test]
    fn test_should_insert_in_reverse_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in (0..300u32).rev() {
                let key = format!("rev-{i:06}");
                let val = format!("val-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..300u32 {
                let key = format!("rev-{i:06}");
                let val = format!("val-{i:06}");
                let got = txn
                    .get(MAIN_DBI, key.as_bytes())
                    .unwrap_or_else(|e| panic!("get {key}: {e}"));
                assert_eq!(got, val.as_bytes(), "mismatch for {key}");
            }
        }
    }

    #[test]
    fn test_should_insert_in_sorted_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..300u32 {
                let key = format!("sorted-{i:06}");
                let val = format!("val-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..300u32 {
                let key = format!("sorted-{i:06}");
                let val = format!("val-{i:06}");
                let got = txn
                    .get(MAIN_DBI, key.as_bytes())
                    .unwrap_or_else(|e| panic!("get {key}: {e}"));
                assert_eq!(got, val.as_bytes(), "mismatch for {key}");
            }
        }
    }

    #[test]
    fn test_should_iterate_in_order_after_splits() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        let count = 200u32;

        // Insert keys in a scrambled order.
        let mut order: Vec<u32> = (0..count).collect();
        let mut rng_state: u64 = 99999;
        for i in (1..order.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            order.swap(i, j);
        }

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for &i in &order {
                let key = format!("iter-{i:06}");
                let val = format!("val-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Iterate using cursor and verify sorted order.
        {
            use crate::cursor::Cursor;
            let _txn = env.begin_ro_txn().expect("begin_ro_txn");
            let meta = env.inner().meta();
            let cmp = env.inner().get_cmp(MAIN_DBI).expect("get_cmp");
            let page_size = env.inner().page_size();
            let root = meta.dbs[MAIN_DBI as usize].root;

            let mut cursor = Cursor::new(page_size, MAIN_DBI);
            let get_page =
                |pgno: u64| -> crate::error::Result<*const u8> { env.inner().get_page(pgno) };

            cursor.first(root, &*cmp, &get_page).expect("cursor first");

            let mut collected_keys: Vec<String> = Vec::new();
            loop {
                if let Some(key_bytes) = cursor.current_key() {
                    let key_str = std::str::from_utf8(key_bytes).expect("valid utf8");
                    collected_keys.push(key_str.to_string());
                }
                if cursor.next(&get_page).is_err() {
                    break;
                }
            }

            assert_eq!(
                collected_keys.len(),
                count as usize,
                "should have collected all keys"
            );

            // Verify sorted order.
            for i in 1..collected_keys.len() {
                assert!(
                    collected_keys[i - 1] < collected_keys[i],
                    "keys should be in sorted order: {} >= {}",
                    collected_keys[i - 1],
                    collected_keys[i],
                );
            }
        }
    }

    #[test]
    fn test_should_persist_across_transactions_after_split() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Insert in one transaction.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 0..200u32 {
                let key = format!("persist-{i:06}");
                let val = format!("val-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Add more in a second transaction.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for i in 200..400u32 {
                let key = format!("persist-{i:06}");
                let val = format!("val-{i:06}");
                txn.put(
                    MAIN_DBI,
                    key.as_bytes(),
                    val.as_bytes(),
                    WriteFlags::empty(),
                )
                .unwrap_or_else(|e| panic!("put {key}: {e}"));
            }
            txn.commit().expect("commit");
        }

        // Verify all 400 keys.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..400u32 {
                let key = format!("persist-{i:06}");
                let val = format!("val-{i:06}");
                let got = txn
                    .get(MAIN_DBI, key.as_bytes())
                    .unwrap_or_else(|e| panic!("get {key}: {e}"));
                assert_eq!(got, val.as_bytes(), "mismatch for {key}");
            }
        }
    }
}
