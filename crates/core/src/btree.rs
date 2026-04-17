//! B+ tree page split, merge, and rebalance operations.
//!
//! This module implements the core B+ tree mutation algorithms: inserting keys
//! with automatic page splitting, deleting keys with rebalancing, and the
//! supporting tree-walk logic. These are free functions that operate on an
//! [`RwTransaction`] to keep the write-path logic separated from the
//! transaction lifecycle code in [`crate::write`].

use std::{cmp::Ordering, sync::Arc};

use crate::{
    cmp::CmpFn,
    error::{Error, Result},
    node::{init_page, leaf_size, node_add, node_add_bigdata, node_del},
    page::{Page, even},
    types::*,
    write::{PageBuf, PutHint, RwTransaction, db_stat_to_bytes},
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
    /// For BIGDATA nodes, the actual data size stored in the node header.
    actual_data_size: Option<u32>,
}

/// A branch entry collected during a split.
struct BranchEntry {
    key: Vec<u8>,
    child_pgno: u64,
}

// ---------------------------------------------------------------------------
// Size-aware split point calculation
// ---------------------------------------------------------------------------

/// Compute the on-page size of a single leaf entry (node header + key + inline
/// data + the 2-byte pointer slot).
fn leaf_entry_size(entry: &LeafEntry) -> usize {
    let data_len = if entry.flags.contains(NodeFlags::BIGDATA) {
        // Overflow pointer is a pgno stored inline (8 bytes).
        size_of::<u64>()
    } else {
        entry.data.len()
    };
    even(NODE_HEADER_SIZE + entry.key.len() + data_len) + size_of::<u16>()
}

/// Compute the on-page size of a single branch entry.
fn branch_entry_size(entry: &BranchEntry) -> usize {
    even(NODE_HEADER_SIZE + entry.key.len()) + size_of::<u16>()
}

/// Find the optimal split point for a leaf page using size-aware splitting.
///
/// Mirrors the C LMDB algorithm: if the page has few keys, contains a large
/// new item, or the new item is appended at the end, we accumulate node sizes
/// from one side and split where the page fills up instead of splitting by
/// count.
fn find_leaf_split_point(
    entries: &[LeafEntry],
    insert_idx: usize,
    new_entry_size: usize,
    page_size: usize,
) -> usize {
    let total = entries.len();
    let default_split = total.div_ceil(2);
    let pmax = page_size - PAGE_HEADER_SIZE;
    let key_thresh = page_size >> 7;

    // Decide whether size-aware splitting is needed.
    let need_size_split = total < key_thresh || new_entry_size > pmax / 16 || insert_idx >= total;

    if !need_size_split {
        return default_split;
    }

    // Scan from the left, accumulating sizes until we exceed half the page.
    if insert_idx <= default_split || insert_idx >= total {
        // Scan left-to-right up to default_split + 1.
        let k = if insert_idx >= total {
            total
        } else {
            default_split + 1
        };
        let mut psize = 0usize;
        for (i, entry) in entries[..k].iter().enumerate() {
            psize += leaf_entry_size(entry);
            if psize > pmax {
                return i;
            }
        }
        // All entries up to k fit; use default.
        default_split
    } else {
        // Scan right-to-left from the end down to default_split - 1.
        let k = default_split.saturating_sub(1);
        let mut psize = 0usize;
        for (i, entry) in entries[k + 1..total].iter().enumerate().rev() {
            psize += leaf_entry_size(entry);
            if psize > pmax {
                return k + 1 + i + 1;
            }
        }
        default_split
    }
}

/// Find the optimal split point for a branch page using size-aware splitting.
fn find_branch_split_point(
    entries: &[BranchEntry],
    insert_idx: usize,
    new_entry_size: usize,
    page_size: usize,
) -> usize {
    let total = entries.len();
    let default_split = total.div_ceil(2);
    let pmax = page_size - PAGE_HEADER_SIZE;
    let key_thresh = page_size >> 7;

    let need_size_split = total < key_thresh || new_entry_size > pmax / 16 || insert_idx >= total;

    if !need_size_split {
        return default_split;
    }

    if insert_idx <= default_split || insert_idx >= total {
        let k = if insert_idx >= total {
            total
        } else {
            default_split + 1
        };
        let mut psize = 0usize;
        for (i, entry) in entries[..k].iter().enumerate() {
            psize += branch_entry_size(entry);
            if psize > pmax {
                return i;
            }
        }
        default_split
    } else {
        let k = default_split.saturating_sub(1);
        let mut psize = 0usize;
        for (i, entry) in entries[k + 1..total].iter().enumerate().rev() {
            psize += branch_entry_size(entry);
            if psize > pmax {
                return k + 1 + i + 1;
            }
        }
        default_split
    }
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
    cursor_put_with_flags(txn, dbi, key, data, flags, NodeFlags::empty())
}

/// Insert a key/value pair with explicit node flags, splitting pages as needed.
///
/// This variant allows specifying node-level flags (e.g., `SUBDATA` for named
/// database records). See [`cursor_put`] for the general-purpose version.
///
/// # Errors
///
/// - [`Error::BadValSize`] if key is empty or exceeds the maximum key size
/// - [`Error::KeyExist`] if `NO_OVERWRITE` is set and the key already exists
/// - [`Error::MapFull`] if no more pages can be allocated
pub fn cursor_put_with_flags(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    key: &[u8],
    data: &[u8],
    flags: WriteFlags,
    node_flags: NodeFlags,
) -> Result<()> {
    if key.is_empty() || key.len() > txn.env.max_key_size {
        return Err(Error::BadValSize);
    }

    let db = *txn.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
    let page_size = txn.env.page_size;
    let node_max = txn.env.node_max;

    // Check if the value requires overflow pages.
    let needs_overflow = NODE_HEADER_SIZE + key.len() + data.len() > node_max;

    if db.root == P_INVALID {
        // Empty database -- create a new root leaf page.
        let (root_pgno, mut root_buf) = txn.page_alloc()?;
        init_page(
            root_buf.as_mut_slice(),
            root_pgno,
            PageFlags::LEAF | PageFlags::DIRTY,
            page_size,
        );

        if needs_overflow {
            let (overflow_pgno, num_pages) = write_overflow_pages(txn, data)?;
            node_add_bigdata(
                root_buf.as_mut_slice(),
                page_size,
                0,
                key,
                overflow_pgno,
                data.len() as u32,
            )?;
            txn.dirty.insert(root_pgno, root_buf);
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.overflow_pages += num_pages as u64;
        } else {
            node_add(
                root_buf.as_mut_slice(),
                page_size,
                0,
                key,
                data,
                0,
                node_flags,
            )?;
            txn.dirty.insert(root_pgno, root_buf);
        }

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = root_pgno;
        db_mut.depth = 1;
        db_mut.leaf_pages = 1;
        db_mut.entries = 1;
        txn.db_dirty[dbi as usize] = true;

        // Single-leaf tree, trivially rightmost — populate hint so the very
        // next put can fast-path, not just the third-and-beyond one.
        if !needs_overflow {
            let db_is_dupsort = db.flags & DatabaseFlags::DUP_SORT.bits() as u16 != 0;
            if !db_is_dupsort {
                // Refetch the just-inserted PageBuf to grab a stable pointer
                // into its Vec<u8> heap storage.
                if let Some(buf) = txn.dirty.find_mut(root_pgno) {
                    let leaf_ptr = buf.as_mut_slice().as_mut_ptr();
                    txn.put_hint = Some(PutHint {
                        dbi,
                        root_pgno,
                        leaf_pgno: root_pgno,
                        is_rightmost: true,
                        last_key: key.to_vec(),
                        leaf_ptr,
                    });
                }
            }
        }
        return Ok(());
    }

    // Walk the tree from root to leaf, COW-ing each page.
    // `cmp` is a cheap Arc clone from the local per-txn cache — no RwLock.
    let cmp = txn
        .cmp_cache
        .get(dbi as usize)
        .ok_or(Error::BadDbi)?
        .clone();

    // APPEND fast-path: walk to the last leaf, verify key ordering.
    if flags.contains(WriteFlags::APPEND) {
        return append_put(txn, dbi, key, data, node_flags, needs_overflow, &cmp);
    }

    // Leaf-hint fast path — monotonic / append-style workloads repeatedly
    // hit the same rightmost leaf. When the previous put left a valid
    // hint, skip the root-to-leaf walk entirely.
    let db_is_dupsort_check = db.flags & DatabaseFlags::DUP_SORT.bits() as u16 != 0;
    if !db_is_dupsort_check && !needs_overflow && !flags.contains(WriteFlags::NO_DUP_DATA) {
        // Snapshot the raw leaf pointer out of the hint so we can skip
        // the `dirty.find_mut` binary search on the hot path. The pointer
        // was captured when this hint was populated; it's valid as long
        // as no hint-invalidating op has run since (see PutHint docs).
        let hint_leaf = match &txn.put_hint {
            Some(h)
                if h.dbi == dbi
                    && h.root_pgno == db.root
                    && h.is_rightmost
                    && !h.leaf_ptr.is_null()
                    && (**cmp)(key, &h.last_key) == Ordering::Greater =>
            {
                Some(h.leaf_ptr)
            }
            _ => None,
        };

        if let Some(leaf_ptr) = hint_leaf {
            // SAFETY: `leaf_ptr` was captured from a `PageBuf` held in
            // `DirtyPages`. The `PageBuf` owns a `Vec<u8>` of fixed
            // `page_size` capacity; its heap storage address is stable
            // for the Vec's lifetime. The hint is invalidated by every
            // op that could remove the PageBuf (spill, explicit remove,
            // commit/abort, cursor-level mutation, del/reserve/etc).
            // RwTransaction is single-threaded, so no concurrent
            // access is possible.
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(leaf_ptr, page_size) };
            let insert_idx = Page::from_raw(buf_slice).num_keys();
            let add = node_add(buf_slice, page_size, insert_idx, key, data, 0, node_flags);
            match add {
                Ok(()) => {
                    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                    db_mut.entries += 1;
                    txn.db_dirty[dbi as usize] = true;
                    // Appended at the end of a rightmost leaf — refresh
                    // `last_key` in place without dropping the hint.
                    if let Some(h) = txn.put_hint.as_mut() {
                        h.last_key.clear();
                        h.last_key.extend_from_slice(key);
                    }
                    return Ok(());
                }
                Err(Error::PageFull) => {
                    // Leaf is full — we need a split. Invalidate the
                    // hint; the slow path below will re-populate if the
                    // split's target leaf is still rightmost.
                    txn.put_hint = None;
                }
                Err(e) => return Err(e),
            }
        }
    }

    let (path, leaf_pgno, is_rightmost) = walk_and_touch(txn, db.root, key, &**cmp)?;

    // Try to insert on the leaf page.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let nkeys = leaf_page.num_keys();

    // Find insertion point and check for duplicate.
    let (insert_idx, exact) = page_node_search(&leaf_page, key, &**cmp);
    let mut overwrite = false;

    let db_is_dupsort = db.flags & DatabaseFlags::DUP_SORT.bits() as u16 != 0;

    if exact && insert_idx < nkeys {
        if db_is_dupsort {
            // DUPSORT: insert the value as a duplicate, don't overwrite the key
            let dcmp = txn
                .dcmp_cache
                .get(dbi as usize)
                .ok_or(Error::BadDbi)?
                .clone();
            return dupsort_put(
                txn, dbi, &path, leaf_pgno, insert_idx, key, data, flags, &**dcmp,
            );
        }
        if flags.contains(WriteFlags::NO_OVERWRITE) {
            return Err(Error::KeyExist);
        }
        overwrite = true;
    }

    // If overwriting, free old overflow pages before deleting the node.
    if overwrite {
        free_overflow_if_bigdata(txn, dbi, leaf_pgno, insert_idx, page_size)?;
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, insert_idx);
    }

    // If the value needs overflow pages, allocate them now.
    if needs_overflow {
        let (overflow_pgno, num_pages) = write_overflow_pages(txn, data)?;
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        let add_result = node_add_bigdata(
            buf.as_mut_slice(),
            page_size,
            insert_idx,
            key,
            overflow_pgno,
            data.len() as u32,
        );
        match add_result {
            Ok(()) => {
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                db_mut.root = path[0].pgno;
                db_mut.overflow_pages += num_pages as u64;
                if !overwrite {
                    db_mut.entries += 1;
                }
                txn.db_dirty[dbi as usize] = true;
                // Overflow put — don't bother with the hint (fast path
                // requires !needs_overflow anyway).
                txn.put_hint = None;
                Ok(())
            }
            Err(Error::PageFull) => {
                // Split with the overflow reference (8-byte pgno as data).
                let pgno_bytes = overflow_pgno.to_le_bytes();
                split_and_insert_bigdata(
                    txn,
                    dbi,
                    &path,
                    key,
                    &pgno_bytes,
                    data.len() as u32,
                    insert_idx,
                    &**cmp,
                )?;
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                db_mut.overflow_pages += num_pages as u64;
                if !overwrite {
                    db_mut.entries += 1;
                }
                txn.db_dirty[dbi as usize] = true;
                txn.put_hint = None;
                Ok(())
            }
            Err(e) => Err(e),
        }
    } else {
        // Attempt inline insertion.
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        let add_result = node_add(
            buf.as_mut_slice(),
            page_size,
            insert_idx,
            key,
            data,
            0,
            node_flags,
        );

        match add_result {
            Ok(()) => {
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                db_mut.root = path[0].pgno;
                if !overwrite {
                    db_mut.entries += 1;
                }
                txn.db_dirty[dbi as usize] = true;
                // Populate the leaf hint if (a) we appended at the end of a
                // rightmost leaf, (b) no overflow, (c) not DUPSORT, (d) not
                // overwriting. These are the conditions the fast-path check
                // later relies on.
                if is_rightmost && !overwrite && !db_is_dupsort && insert_idx == nkeys {
                    // Grab a stable raw pointer into the leaf PageBuf so the
                    // next fast-path put can skip the `dirty.find_mut`.
                    let leaf_ptr = txn
                        .dirty
                        .find_mut(leaf_pgno)
                        .map(|b| b.as_mut_slice().as_mut_ptr())
                        .unwrap_or(std::ptr::null_mut());
                    if !leaf_ptr.is_null() {
                        txn.put_hint = Some(PutHint {
                            dbi,
                            root_pgno: path[0].pgno,
                            leaf_pgno,
                            is_rightmost: true,
                            last_key: key.to_vec(),
                            leaf_ptr,
                        });
                    } else {
                        txn.put_hint = None;
                    }
                } else {
                    txn.put_hint = None;
                }
                Ok(())
            }
            Err(Error::PageFull) => {
                split_and_insert(txn, dbi, &path, key, data, node_flags, insert_idx, &**cmp)?;
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                if !overwrite {
                    db_mut.entries += 1;
                }
                txn.db_dirty[dbi as usize] = true;
                // Splits change the tree shape — invalidate the hint. The
                // next put will re-walk and, if it lands on a rightmost
                // leaf, re-populate.
                txn.put_hint = None;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

/// Delete a key from the specified database, rebalancing as needed.
///
/// # Errors
///
/// - [`Error::NotFound`] if the key does not exist
/// - [`Error::BadDbi`] if the database handle is invalid
///
/// For DUPSORT databases:
/// - If `dup_data` is `None`, deletes the key and ALL its duplicates.
/// - If `dup_data` is `Some`, deletes only the matching duplicate value.
///
/// For non-DUPSORT databases, `dup_data` is ignored.
pub fn cursor_del(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    key: &[u8],
    dup_data: Option<&[u8]>,
) -> Result<()> {
    let db = *txn.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
    if db.root == P_INVALID {
        return Err(Error::NotFound);
    }

    let cmp = txn
        .cmp_cache
        .get(dbi as usize)
        .ok_or(Error::BadDbi)?
        .clone();
    let page_size = txn.env.page_size;

    // Walk the tree, COW-ing each page.
    let (path, leaf_pgno, _is_rightmost) = walk_and_touch(txn, db.root, key, &**cmp)?;

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

    // Check if this is a DUPSORT database and we need to delete a specific dup.
    let db_is_dupsort = db.flags & DatabaseFlags::DUP_SORT.bits() as u16 != 0;
    if db_is_dupsort {
        if let Some(del_data) = dup_data {
            let dcmp = txn.env.get_dcmp(dbi)?;
            return dupsort_del_single(txn, dbi, &path, leaf_pgno, idx, del_data, &**dcmp);
        }
        // dup_data is None: delete all dups (the whole key).
        // For DUPSORT nodes with F_DUPDATA, count how many entries to subtract.
        let node = leaf_page.node(idx);
        if node.is_dupdata() {
            let dup_count = count_dups_in_node(&node);
            free_overflow_if_bigdata(txn, dbi, leaf_pgno, idx, page_size)?;
            let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
            node_del(buf.as_mut_slice(), page_size, idx);

            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.entries = db_mut.entries.saturating_sub(dup_count);
            txn.db_dirty[dbi as usize] = true;
            return finish_del(txn, dbi, &path, leaf_pgno, page_size);
        }
        // Single value (no F_DUPDATA), fall through to normal delete.
    }

    // Free overflow pages if this is a BIGDATA node.
    free_overflow_if_bigdata(txn, dbi, leaf_pgno, idx, page_size)?;

    // Delete the node.
    let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
    node_del(buf.as_mut_slice(), page_size, idx);

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.entries = db_mut.entries.saturating_sub(1);
    txn.db_dirty[dbi as usize] = true;

    finish_del(txn, dbi, &path, leaf_pgno, page_size)
}

/// Post-deletion handling: check for empty leaf and rebalance.
fn finish_del(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    leaf_pgno: u64,
    page_size: usize,
) -> Result<()> {
    let leaf_page = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?.as_page();

    if leaf_page.num_keys() == 0 {
        if path.len() == 1 {
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = P_INVALID;
            db_mut.depth = 0;
            db_mut.leaf_pages = 0;
            return Ok(());
        }
        remove_from_parent(txn, dbi, path)?;
        return Ok(());
    }

    let fill = leaf_page.used_space() * 10 / (page_size - PAGE_HEADER_SIZE);
    if path.len() > 1 && fill < FILL_THRESHOLD / 100 {
        rebalance(txn, dbi, path)?;
    } else {
        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = path[0].pgno;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal: APPEND fast-path
// ---------------------------------------------------------------------------

/// APPEND mode insert: walk to the rightmost leaf, verify that the new key
/// is strictly greater than the last key, and insert at the end position.
///
/// This skips binary search entirely and is optimized for sequential
/// (monotonically increasing) key inserts.
///
/// # Errors
///
/// Returns [`Error::KeyExist`] if the new key is not strictly greater
/// than the last existing key.
fn append_put(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    key: &[u8],
    data: &[u8],
    node_flags: NodeFlags,
    needs_overflow: bool,
    cmp: &Arc<Box<CmpFn>>,
) -> Result<()> {
    let db = *txn.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
    let page_size = txn.env.page_size;

    let (path, leaf_pgno) = walk_to_last(txn, db.root)?;

    // Check that the new key is strictly greater than the last key on the page.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let nkeys = leaf_page.num_keys();

    if nkeys > 0 {
        let last_node = leaf_page.node(nkeys - 1);
        if cmp(key, last_node.key()) != Ordering::Greater {
            return Err(Error::KeyExist);
        }
    }

    let insert_idx = nkeys;

    // Insert at the end position — same logic as the normal put path.
    if needs_overflow {
        let (overflow_pgno, num_pages) = write_overflow_pages(txn, data)?;
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        let add_result = node_add_bigdata(
            buf.as_mut_slice(),
            page_size,
            insert_idx,
            key,
            overflow_pgno,
            data.len() as u32,
        );
        match add_result {
            Ok(()) => {
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                db_mut.root = path[0].pgno;
                db_mut.overflow_pages += num_pages as u64;
                db_mut.entries += 1;
                txn.db_dirty[dbi as usize] = true;
                Ok(())
            }
            Err(Error::PageFull) => {
                let pgno_bytes = overflow_pgno.to_le_bytes();
                split_and_insert_bigdata(
                    txn,
                    dbi,
                    &path,
                    key,
                    &pgno_bytes,
                    data.len() as u32,
                    insert_idx,
                    &**cmp,
                )?;
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                db_mut.overflow_pages += num_pages as u64;
                db_mut.entries += 1;
                txn.db_dirty[dbi as usize] = true;
                Ok(())
            }
            Err(e) => Err(e),
        }
    } else {
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        let add_result = node_add(
            buf.as_mut_slice(),
            page_size,
            insert_idx,
            key,
            data,
            0,
            node_flags,
        );
        match add_result {
            Ok(()) => {
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                db_mut.root = path[0].pgno;
                db_mut.entries += 1;
                txn.db_dirty[dbi as usize] = true;
                Ok(())
            }
            Err(Error::PageFull) => {
                split_and_insert(txn, dbi, &path, key, data, node_flags, insert_idx, &**cmp)?;
                let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
                db_mut.entries += 1;
                txn.db_dirty[dbi as usize] = true;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

/// Walk from root to the rightmost leaf, COW-ing each page along the path.
///
/// Returns the path from root to leaf and the leaf page number. Always
/// descends to the rightmost child at each branch level.
fn walk_to_last(txn: &mut RwTransaction<'_>, root_pgno: u64) -> Result<(TreePath, u64)> {
    let page_size = txn.env.page_size;
    let mut path = TreePath::new();

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

        let nkeys = page.num_keys();
        if nkeys == 0 {
            return Err(Error::Corrupted);
        }

        // Always go to the rightmost child.
        let child_idx = nkeys - 1;

        path.push(PathLevel {
            pgno: current_pgno,
            idx: child_idx,
        });

        let child_pgno = page.node(child_idx).child_pgno();
        let new_child_pgno = txn.page_touch(child_pgno)?;

        if new_child_pgno != child_pgno {
            update_branch_child(txn, current_pgno, child_idx, new_child_pgno, page_size)?;
        }

        current_pgno = new_child_pgno;
    }
}

// ---------------------------------------------------------------------------
// Internal: tree walk
// ---------------------------------------------------------------------------

/// Walk from root to the leaf containing `key`, COW-ing each page.
///
/// Returns `(path, leaf_pgno, is_rightmost)`. `is_rightmost` is true when
/// every branch-level descent took the rightmost child — the caller uses
/// this to decide whether a subsequent put can fast-path to this leaf
/// (there is no right-sibling leaf that could own a still-larger key).
///
/// Each page along the path is guaranteed to be in the dirty list after
/// this call.
fn walk_and_touch(
    txn: &mut RwTransaction<'_>,
    root_pgno: u64,
    key: &[u8],
    cmp: &CmpFn,
) -> Result<(TreePath, u64, bool)> {
    let page_size = txn.env.page_size;
    let mut path = TreePath::new();
    let mut is_rightmost = true;

    // Touch the root page.
    let mut current_pgno = txn.page_touch(root_pgno)?;

    loop {
        let page = read_dirty_page(txn, current_pgno, page_size)?;

        if page.is_leaf() {
            path.push(PathLevel {
                pgno: current_pgno,
                idx: 0,
            });
            return Ok((path, current_pgno, is_rightmost));
        }

        if !page.is_branch() {
            return Err(Error::Corrupted);
        }

        // Search for the child to descend into.
        let nkeys = page.num_keys();
        let (idx, exact) = page_node_search(&page, key, cmp);
        let child_idx = if exact {
            idx
        } else if idx > 0 {
            idx - 1
        } else {
            0
        };

        // We stay on the rightmost spine only if every branch descent picks
        // the rightmost child.
        if child_idx + 1 != nkeys {
            is_rightmost = false;
        }

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
/// Called on every COW descent when the child gets a new page number — so
/// this is on the write hot path. The previous implementation did
/// `node_del` + `node_add`, each an O(n) shift of the page's pointer array
/// and an even-aligned memmove of the node data area. Since only the 6
/// bytes encoding the 48-bit child pgno change (lo/hi/flags fields of the
/// node header), we overwrite them directly; the key, pointer array, and
/// all sibling nodes stay put.
#[inline]
fn update_branch_child(
    txn: &mut RwTransaction<'_>,
    branch_pgno: u64,
    child_idx: usize,
    new_child_pgno: u64,
    _page_size: usize,
) -> Result<()> {
    let buf = txn.dirty.find_mut(branch_pgno).ok_or(Error::Corrupted)?;
    let slice = buf.as_mut_slice();

    // Locate the target node header via the pointer array.
    let ptr_offset = PAGE_HEADER_SIZE + child_idx * 2;
    let node_offset = u16::from_le_bytes([slice[ptr_offset], slice[ptr_offset + 1]]) as usize;

    // Encode the 48-bit pgno into the 6 bytes at [node_offset, node_offset+6).
    //   lo    = pgno & 0xFFFF           (offset 0..2)
    //   hi    = (pgno >> 16) & 0xFFFF   (offset 2..4)
    //   flags = (pgno >> 32) & 0xFFFF   (offset 4..6)   -- branch nodes reuse
    //                                                      the flags slot as
    //                                                      the high pgno bits.
    let lo = (new_child_pgno & 0xFFFF) as u16;
    let hi = ((new_child_pgno >> 16) & 0xFFFF) as u16;
    let flags_raw = ((new_child_pgno >> 32) & 0xFFFF) as u16;
    slice[node_offset..node_offset + 2].copy_from_slice(&lo.to_le_bytes());
    slice[node_offset + 2..node_offset + 4].copy_from_slice(&hi.to_le_bytes());
    slice[node_offset + 4..node_offset + 6].copy_from_slice(&flags_raw.to_le_bytes());

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
        let ads = if node.is_bigdata() {
            Some(node.data_size())
        } else {
            None
        };
        entries.push(LeafEntry {
            key: node.key().to_vec(),
            data: node.node_data().to_vec(),
            flags: node.flags(),
            actual_data_size: ads,
        });
    }

    // Insert the new entry at the correct position.
    entries.insert(
        insert_idx,
        LeafEntry {
            key: key.to_vec(),
            data: data.to_vec(),
            flags,
            actual_data_size: None,
        },
    );

    let new_entry_size = leaf_entry_size(&entries[insert_idx]);
    let split_idx = find_leaf_split_point(&entries, insert_idx, new_entry_size, page_size);

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
        add_leaf_entry(left_buf.as_mut_slice(), page_size, i, entry)?;
    }

    // Populate the right page.
    for (i, entry) in entries[split_idx..].iter().enumerate() {
        add_leaf_entry(right_buf.as_mut_slice(), page_size, i, entry)?;
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

    let new_entry_size = branch_entry_size(&entries[insert_idx]);
    let split_idx = find_branch_split_point(&entries, insert_idx, new_entry_size, page_size);

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
        // Cannot merge and pages are too full to steal keys.
        // This is acceptable — the page is underfilled but not empty.
        // Key-stealing optimization would move one key from the sibling
        // to balance fill, but it's not required for correctness.
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
        add_leaf_entry(buf.as_mut_slice(), page_size, i, entry)?;
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

/// Steal one entry from a sibling to rebalance an underfilled leaf page.
///
/// When the sibling and the underfilled page cannot be merged (combined
/// size exceeds page capacity), we move one entry from the sibling to
/// the underfilled page. This keeps both pages above the fill threshold.
///
/// If merging from the right sibling, the first entry of the right page
/// is moved to the end of the left page. If from the left sibling, the
/// last entry of the left page is moved to the beginning of the right page.
/// The parent separator key is updated accordingly.
#[allow(clippy::too_many_arguments, dead_code)]
fn steal_key(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    leaf_pgno: u64,
    sibling_pgno: u64,
    merge_right: bool,
    left_entries: Vec<LeafEntry>,
    right_entries: Vec<LeafEntry>,
) -> Result<()> {
    let page_size = txn.env.page_size;
    let leaf_level = path.len() - 1;
    let parent_pgno = path[leaf_level - 1].pgno;
    let child_idx = path[leaf_level - 1].idx;

    if merge_right {
        // Steal first entry from right sibling (sibling_pgno) to left (leaf_pgno).
        if right_entries.is_empty() {
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = path[0].pgno;
            return Ok(());
        }

        let stolen = &right_entries[0];
        let new_left: Vec<&LeafEntry> =
            left_entries.iter().chain(std::iter::once(stolen)).collect();
        let new_right: Vec<&LeafEntry> = right_entries[1..].iter().collect();

        // Rebuild left page.
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        init_page(
            buf.as_mut_slice(),
            leaf_pgno,
            PageFlags::LEAF | PageFlags::DIRTY,
            page_size,
        );
        for (i, entry) in new_left.iter().enumerate() {
            add_leaf_entry(buf.as_mut_slice(), page_size, i, entry)?;
        }

        // Rebuild right page.
        let buf = txn.dirty.find_mut(sibling_pgno).ok_or(Error::Corrupted)?;
        init_page(
            buf.as_mut_slice(),
            sibling_pgno,
            PageFlags::LEAF | PageFlags::DIRTY,
            page_size,
        );
        for (i, entry) in new_right.iter().enumerate() {
            add_leaf_entry(buf.as_mut_slice(), page_size, i, entry)?;
        }

        // Update the parent separator: it should be the new first key of the right page.
        if !new_right.is_empty() {
            let sibling_parent_idx = child_idx + 1;
            let new_sep = new_right[0].key.clone();
            let parent_buf = txn.dirty.find_mut(parent_pgno).ok_or(Error::Corrupted)?;
            let parent_page = Page::from_raw(parent_buf.as_slice());
            let old_child_pgno = parent_page.node(sibling_parent_idx).child_pgno();
            node_del(parent_buf.as_mut_slice(), page_size, sibling_parent_idx);
            node_add(
                parent_buf.as_mut_slice(),
                page_size,
                sibling_parent_idx,
                &new_sep,
                &[],
                old_child_pgno,
                NodeFlags::empty(),
            )?;
        }
    } else {
        // Steal last entry from left sibling (sibling_pgno) to right (leaf_pgno).
        if left_entries.is_empty() {
            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = path[0].pgno;
            return Ok(());
        }

        let stolen = &left_entries[left_entries.len() - 1];
        let new_left: Vec<&LeafEntry> = left_entries[..left_entries.len() - 1].iter().collect();
        let new_right: Vec<&LeafEntry> = std::iter::once(stolen)
            .chain(right_entries.iter())
            .collect();

        // Rebuild left page (sibling).
        let buf = txn.dirty.find_mut(sibling_pgno).ok_or(Error::Corrupted)?;
        init_page(
            buf.as_mut_slice(),
            sibling_pgno,
            PageFlags::LEAF | PageFlags::DIRTY,
            page_size,
        );
        for (i, entry) in new_left.iter().enumerate() {
            add_leaf_entry(buf.as_mut_slice(), page_size, i, entry)?;
        }

        // Rebuild right page (leaf).
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        init_page(
            buf.as_mut_slice(),
            leaf_pgno,
            PageFlags::LEAF | PageFlags::DIRTY,
            page_size,
        );
        for (i, entry) in new_right.iter().enumerate() {
            add_leaf_entry(buf.as_mut_slice(), page_size, i, entry)?;
        }

        // Update the parent separator: it should be the new first key of the right page
        // (leaf_pgno).
        let new_sep = new_right[0].key.clone();
        let parent_buf = txn.dirty.find_mut(parent_pgno).ok_or(Error::Corrupted)?;
        let parent_page = Page::from_raw(parent_buf.as_slice());
        let old_child_pgno = parent_page.node(child_idx).child_pgno();
        node_del(parent_buf.as_mut_slice(), page_size, child_idx);
        node_add(
            parent_buf.as_mut_slice(),
            page_size,
            child_idx,
            &new_sep,
            &[],
            old_child_pgno,
            NodeFlags::empty(),
        )?;
    }

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.root = path[0].pgno;
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
        let ads = if node.is_bigdata() {
            Some(node.data_size())
        } else {
            None
        };
        entries.push(LeafEntry {
            key: node.key().to_vec(),
            data: node.node_data().to_vec(),
            flags: node.flags(),
            actual_data_size: ads,
        });
    }
    Ok(entries)
}

/// Add a leaf entry to a page, handling BIGDATA nodes correctly.
///
/// For BIGDATA entries, uses `node_add_bigdata` to encode the actual data
/// size in the header. For normal entries, uses `node_add`.
fn add_leaf_entry(page: &mut [u8], page_size: usize, idx: usize, entry: &LeafEntry) -> Result<()> {
    if let Some(ads) = entry.actual_data_size {
        let mut pgno_bytes = [0u8; 8];
        pgno_bytes.copy_from_slice(&entry.data[..8]);
        let overflow_pgno = u64::from_le_bytes(pgno_bytes);
        node_add_bigdata(page, page_size, idx, &entry.key, overflow_pgno, ads)
    } else {
        node_add(
            page,
            page_size,
            idx,
            &entry.key,
            &entry.data,
            0,
            entry.flags,
        )
    }
}

// ---------------------------------------------------------------------------
// Internal: overflow page helpers
// ---------------------------------------------------------------------------

/// Allocate overflow pages and write data into them.
///
/// Returns `(overflow_pgno, num_pages)` where `overflow_pgno` is the first
/// page number and `num_pages` is the count of contiguous pages allocated.
///
/// The overflow data is stored as a single large contiguous `PageBuf`
/// at the starting pgno. This allows within-transaction reads to access the
/// data via a single pointer without reassembly.
fn write_overflow_pages(txn: &mut RwTransaction<'_>, data: &[u8]) -> Result<(u64, usize)> {
    let page_size = txn.env.page_size;
    let num_pages = (PAGE_HEADER_SIZE + data.len()).div_ceil(page_size);
    let total_size = num_pages * page_size;

    let (start_pgno, _bufs) = txn.page_alloc_multi(num_pages)?;

    // Create a single large buffer covering all overflow pages.
    let mut big_buf = PageBuf::new(total_size);
    let buf = big_buf.as_mut_slice();

    // Initialize the first overflow page header.
    // pgno (offset 0..8)
    buf[0..8].copy_from_slice(&start_pgno.to_le_bytes());
    // pad (offset 8..10) = 0
    // flags (offset 10..12) = P_OVERFLOW | P_DIRTY
    let flags = PageFlags::OVERFLOW | PageFlags::DIRTY;
    buf[10..12].copy_from_slice(&flags.bits().to_le_bytes());
    // overflow_pages count (offset 12..16) as u32
    buf[12..16].copy_from_slice(&(num_pages as u32).to_le_bytes());

    // Write data starting at PAGE_HEADER_SIZE.
    buf[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + data.len()].copy_from_slice(data);

    // Store the entire overflow chain as a single buffer at start_pgno.
    txn.dirty.insert(start_pgno, big_buf);

    Ok((start_pgno, num_pages))
}

/// If the node at `idx` on the leaf page `leaf_pgno` has the `BIGDATA` flag,
/// free its overflow pages and update the database overflow page counter.
fn free_overflow_if_bigdata(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    leaf_pgno: u64,
    idx: usize,
    page_size: usize,
) -> Result<()> {
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let node = leaf_page.node(idx);

    if !node.is_bigdata() {
        return Ok(());
    }

    let overflow_pgno = node.overflow_pgno();

    // Read the overflow page to get the page count.
    let ovf_ptr = txn.get_page(overflow_pgno)?;
    let ovf_slice = unsafe { std::slice::from_raw_parts(ovf_ptr, page_size) };
    let ovf_page = Page::from_raw(ovf_slice);
    let num_pages = ovf_page.overflow_pages() as u64;

    // Free all overflow pages.
    for pg in overflow_pgno..overflow_pgno + num_pages {
        txn.free_pgs.push(pg);
    }
    // Remove from dirty list (stored as single buffer at start_pgno).
    txn.dirty.remove(overflow_pgno);

    // Update overflow page counter.
    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.overflow_pages = db_mut.overflow_pages.saturating_sub(num_pages);

    Ok(())
}

/// Split a full leaf page and insert a new bigdata entry, propagating the
/// separator key up through the tree.
#[allow(clippy::too_many_arguments)]
fn split_and_insert_bigdata(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    key: &[u8],
    pgno_data: &[u8],
    actual_data_size: u32,
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
        let ads = if node.is_bigdata() {
            Some(node.data_size())
        } else {
            None
        };
        entries.push(LeafEntry {
            key: node.key().to_vec(),
            data: node.node_data().to_vec(),
            flags: node.flags(),
            actual_data_size: ads,
        });
    }

    // Insert the new bigdata entry at the correct position.
    entries.insert(
        insert_idx,
        LeafEntry {
            key: key.to_vec(),
            data: pgno_data.to_vec(),
            flags: NodeFlags::BIGDATA,
            actual_data_size: Some(actual_data_size),
        },
    );

    let new_entry_size = leaf_entry_size(&entries[insert_idx]);
    let split_idx = find_leaf_split_point(&entries, insert_idx, new_entry_size, page_size);

    // Allocate right sibling leaf.
    let (right_pgno, mut right_buf) = txn.page_alloc()?;
    init_page(
        right_buf.as_mut_slice(),
        right_pgno,
        PageFlags::LEAF | PageFlags::DIRTY,
        page_size,
    );

    // Reinitialize the left leaf page.
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
        add_leaf_entry(left_buf.as_mut_slice(), page_size, i, entry)?;
    }

    // Populate the right page.
    for (i, entry) in entries[split_idx..].iter().enumerate() {
        add_leaf_entry(right_buf.as_mut_slice(), page_size, i, entry)?;
    }

    let sep_key = entries[split_idx].key.clone();

    txn.dirty.insert(right_pgno, right_buf);

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.leaf_pages += 1;

    insert_separator(txn, dbi, path, leaf_level, &sep_key, left_pgno, right_pgno)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// DUPSORT: sub-page helpers
// ---------------------------------------------------------------------------

/// Read all values from a sub-page (inline dup storage).
///
/// In a sub-page, the "keys" of nodes are the duplicate values. Nodes have no
/// data field.
fn read_sub_page_values(sub_page_data: &[u8]) -> Vec<Vec<u8>> {
    let sp = Page::from_raw(sub_page_data);
    let nkeys = sp.num_keys();
    let mut vals = Vec::with_capacity(nkeys);

    if sp.is_leaf2() {
        let val_size = sp.pad() as usize;
        for i in 0..nkeys {
            vals.push(sp.leaf2_key(i, val_size).to_vec());
        }
    } else {
        for i in 0..nkeys {
            let node = sp.node(i);
            vals.push(node.key().to_vec());
        }
    }
    vals
}

/// Build a sub-page from a sorted list of values.
///
/// The sub-page uses `P_LEAF | P_SUBPAGE | P_DIRTY` flags. If `dupfixed` is
/// true and all values have the same size, uses `P_LEAF2` compact format.
fn build_sub_page(sorted_vals: &[Vec<u8>], dupfixed: bool) -> Vec<u8> {
    if dupfixed && !sorted_vals.is_empty() {
        let val_size = sorted_vals[0].len();
        let all_same_size = sorted_vals.iter().all(|v| v.len() == val_size);
        if all_same_size {
            return build_leaf2_sub_page(sorted_vals, val_size);
        }
    }

    // Calculate total space needed.
    let mut total_node_space = 0usize;
    for v in sorted_vals {
        total_node_space += even(NODE_HEADER_SIZE + v.len());
    }
    let ptrs_space = sorted_vals.len() * 2;
    let buf_size = PAGE_HEADER_SIZE + ptrs_space + total_node_space;

    let mut buf = vec![0u8; buf_size];
    let flags = PageFlags::LEAF | PageFlags::SUBPAGE | PageFlags::DIRTY;
    // pgno = 0 for sub-pages (not meaningful)
    buf[0..8].copy_from_slice(&0u64.to_le_bytes());
    buf[8..10].copy_from_slice(&0u16.to_le_bytes());
    buf[10..12].copy_from_slice(&flags.bits().to_le_bytes());
    buf[12..14].copy_from_slice(&(PAGE_HEADER_SIZE as u16).to_le_bytes());
    buf[14..16].copy_from_slice(&(buf_size as u16).to_le_bytes());

    // Add each value as a node.
    for (i, v) in sorted_vals.iter().enumerate() {
        // node_add treats 'key' as the key and 'data' as data for leaf nodes.
        // For sub-page nodes, the "key" is the dup value and data is empty.
        // Index is simply `i` since we're appending in sorted order.
        node_add(&mut buf, buf_size, i, v, &[], 0, NodeFlags::empty()).unwrap_or_else(|_| {
            // Sub-page should never be full since we sized it correctly.
            unreachable!("sub-page buffer should have enough space")
        });
    }

    buf
}

/// Build a compact LEAF2 sub-page for DUPFIXED databases.
fn build_leaf2_sub_page(sorted_vals: &[Vec<u8>], val_size: usize) -> Vec<u8> {
    let data_size = sorted_vals.len() * val_size;
    let buf_size = PAGE_HEADER_SIZE + data_size;
    let mut buf = vec![0u8; buf_size];

    let flags = PageFlags::LEAF | PageFlags::LEAF2 | PageFlags::SUBPAGE | PageFlags::DIRTY;
    buf[0..8].copy_from_slice(&0u64.to_le_bytes());
    buf[8..10].copy_from_slice(&(val_size as u16).to_le_bytes());
    buf[10..12].copy_from_slice(&flags.bits().to_le_bytes());
    // For LEAF2 pages, lower tracks num_keys via the formula:
    // lower = PAGE_HEADER_SIZE (no pointer array for LEAF2)
    // Instead, num_keys is derived from data length / val_size.
    // But our Page::num_keys() uses (lower - PAGE_HEADER_SIZE) / 2.
    // For LEAF2, we need lower = PAGE_HEADER_SIZE + num_keys * 2 to
    // make num_keys() work correctly.
    let lower = PAGE_HEADER_SIZE + sorted_vals.len() * 2;
    buf[12..14].copy_from_slice(&(lower as u16).to_le_bytes());
    buf[14..16].copy_from_slice(&(buf_size as u16).to_le_bytes());

    // Pack values contiguously after header.
    for (i, v) in sorted_vals.iter().enumerate() {
        let start = PAGE_HEADER_SIZE + i * val_size;
        buf[start..start + val_size].copy_from_slice(v);
    }

    buf
}

/// Handle DUPSORT insertion when the key already exists.
#[allow(clippy::too_many_arguments)]
fn dupsort_put(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    leaf_pgno: u64,
    insert_idx: usize,
    _key: &[u8],
    data: &[u8],
    flags: WriteFlags,
    dcmp: &CmpFn,
) -> Result<()> {
    let page_size = txn.env.page_size;
    let node_max = txn.env.node_max;
    let db = *txn.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
    let dupfixed = db.flags & DatabaseFlags::DUP_FIXED.bits() as u16 != 0;

    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let node = leaf_page.node(insert_idx);

    if !node.is_dupdata() {
        // First duplicate: currently a single value, convert to sub-page.
        let old_data = node.node_data().to_vec();
        let node_key = node.key().to_vec();

        let cmp_result = dcmp(data, &old_data);
        if cmp_result == Ordering::Equal {
            if flags.contains(WriteFlags::NO_DUP_DATA) {
                return Err(Error::KeyExist);
            }
            // Same data already exists.
            return Ok(());
        }

        // Sort the two values.
        let sorted = if cmp_result == Ordering::Less {
            vec![data.to_vec(), old_data]
        } else {
            vec![old_data, data.to_vec()]
        };

        let sub_page = build_sub_page(&sorted, dupfixed);

        // Check if sub-page fits in the node.
        if NODE_HEADER_SIZE + node_key.len() + sub_page.len() > node_max {
            // Sub-page too large even for 2 values, promote directly to sub-DB.
            let sub_db = promote_to_sub_db(txn, &sorted, dupfixed)?;
            let db_bytes = db_stat_to_bytes(&sub_db);

            let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
            node_del(buf.as_mut_slice(), page_size, insert_idx);

            let nf = NodeFlags::DUPDATA | NodeFlags::SUBDATA;
            let add_result = node_add(
                buf.as_mut_slice(),
                page_size,
                insert_idx,
                &node_key,
                &db_bytes,
                0,
                nf,
            );

            match add_result {
                Ok(()) => {}
                Err(Error::PageFull) => {
                    split_and_insert(
                        txn,
                        dbi,
                        path,
                        &node_key,
                        &db_bytes,
                        nf,
                        insert_idx,
                        &|a: &[u8], b: &[u8]| a.cmp(b),
                    )?;
                }
                Err(e) => return Err(e),
            }

            let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
            db_mut.root = path[0].pgno;
            db_mut.entries += 1;
            txn.db_dirty[dbi as usize] = true;
            return Ok(());
        }

        // Delete old node and insert new one with F_DUPDATA.
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, insert_idx);

        let add_result = node_add(
            buf.as_mut_slice(),
            page_size,
            insert_idx,
            &node_key,
            &sub_page,
            0,
            NodeFlags::DUPDATA,
        );

        match add_result {
            Ok(()) => {}
            Err(Error::PageFull) => {
                // Need to split the leaf page.
                split_and_insert(
                    txn,
                    dbi,
                    path,
                    &node_key,
                    &sub_page,
                    NodeFlags::DUPDATA,
                    insert_idx,
                    &|a: &[u8], b: &[u8]| a.cmp(b),
                )?;
            }
            Err(e) => return Err(e),
        }

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = path[0].pgno;
        db_mut.entries += 1; // We added one new dup value.
        txn.db_dirty[dbi as usize] = true;
        return Ok(());
    }

    // Node already has F_DUPDATA: check if it's a sub-database or sub-page.
    if node.flags().contains(NodeFlags::SUBDATA) {
        // Insert into existing sub-database.
        return dupsort_put_into_sub_db(txn, dbi, path, leaf_pgno, insert_idx, data, flags, dcmp);
    }

    // Node has sub-page: insert into existing sub-page.
    let sub_page_data = node.node_data().to_vec();
    let node_key = node.key().to_vec();

    let mut vals = read_sub_page_values(&sub_page_data);

    // Binary search for the insertion point.
    let mut found = false;
    let mut ins_pos = vals.len();
    {
        let mut lo = 0;
        let mut hi = vals.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match dcmp(data, &vals[mid]) {
                Ordering::Equal => {
                    found = true;
                    ins_pos = mid;
                    break;
                }
                Ordering::Greater => lo = mid + 1,
                Ordering::Less => hi = mid,
            }
        }
        if !found {
            ins_pos = lo;
        }
    }

    if found {
        if flags.contains(WriteFlags::NO_DUP_DATA) {
            return Err(Error::KeyExist);
        }
        // Value already exists, nothing to do.
        return Ok(());
    }

    // Insert the new value.
    vals.insert(ins_pos, data.to_vec());

    let new_sub_page = build_sub_page(&vals, dupfixed);

    // Check if the new sub-page fits in a node.
    if NODE_HEADER_SIZE + node_key.len() + new_sub_page.len() > node_max {
        // Sub-page too large, promote to sub-database.
        let sub_db = promote_to_sub_db(txn, &vals, dupfixed)?;
        let db_bytes = db_stat_to_bytes(&sub_db);

        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, insert_idx);

        let nf = NodeFlags::DUPDATA | NodeFlags::SUBDATA;
        let add_result = node_add(
            buf.as_mut_slice(),
            page_size,
            insert_idx,
            &node_key,
            &db_bytes,
            0,
            nf,
        );

        match add_result {
            Ok(()) => {}
            Err(Error::PageFull) => {
                split_and_insert(
                    txn,
                    dbi,
                    path,
                    &node_key,
                    &db_bytes,
                    nf,
                    insert_idx,
                    &|a: &[u8], b: &[u8]| a.cmp(b),
                )?;
            }
            Err(e) => return Err(e),
        }

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.root = path[0].pgno;
        db_mut.entries += 1;
        txn.db_dirty[dbi as usize] = true;
        return Ok(());
    }

    // Delete old node and insert updated one.
    let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
    node_del(buf.as_mut_slice(), page_size, insert_idx);

    let add_result = node_add(
        buf.as_mut_slice(),
        page_size,
        insert_idx,
        &node_key,
        &new_sub_page,
        0,
        NodeFlags::DUPDATA,
    );

    match add_result {
        Ok(()) => {}
        Err(Error::PageFull) => {
            split_and_insert(
                txn,
                dbi,
                path,
                &node_key,
                &new_sub_page,
                NodeFlags::DUPDATA,
                insert_idx,
                &|a: &[u8], b: &[u8]| a.cmp(b),
            )?;
        }
        Err(e) => return Err(e),
    }

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.root = path[0].pgno;
    db_mut.entries += 1;
    txn.db_dirty[dbi as usize] = true;
    Ok(())
}

/// Promote a set of sorted duplicate values into a sub-database B+ tree.
///
/// Creates a new root leaf page for the sub-DB and inserts all values
/// into it. Each value becomes a "key" in the sub-DB with empty data.
/// Returns the `DbStat` describing the new sub-database.
///
/// # Errors
///
/// Returns [`Error::MapFull`] if pages cannot be allocated.
fn promote_to_sub_db(
    txn: &mut RwTransaction<'_>,
    sorted_vals: &[Vec<u8>],
    dupfixed: bool,
) -> Result<DbStat> {
    let page_size = txn.env.page_size;

    // Create root page for sub-DB.
    let (root_pgno, mut root_buf) = txn.page_alloc()?;
    init_page(
        root_buf.as_mut_slice(),
        root_pgno,
        PageFlags::LEAF | PageFlags::DIRTY,
        page_size,
    );

    let mut sub_db = DbStat {
        pad: 0,
        flags: if dupfixed {
            DatabaseFlags::DUP_FIXED.bits() as u16
        } else {
            0
        },
        depth: 1,
        branch_pages: 0,
        leaf_pages: 1,
        overflow_pages: 0,
        entries: 0,
        root: root_pgno,
    };

    // Try to fit all values into the root page. If it overflows,
    // we use cursor_put to handle splitting.
    let mut need_cursor_insert: Vec<Vec<u8>> = Vec::new();
    for val in sorted_vals {
        let result = node_add(
            root_buf.as_mut_slice(),
            page_size,
            sub_db.entries as usize,
            val,
            &[],
            0,
            NodeFlags::empty(),
        );
        match result {
            Ok(()) => sub_db.entries += 1,
            Err(Error::PageFull) => {
                // Remaining values need cursor-based insertion with splitting.
                need_cursor_insert.push(val.clone());
            }
            Err(e) => return Err(e),
        }
    }

    txn.dirty.insert(root_pgno, root_buf);

    // If some values didn't fit, insert them via the full B+ tree machinery.
    if !need_cursor_insert.is_empty() {
        for val in &need_cursor_insert {
            insert_into_sub_db_tree(txn, &mut sub_db, val)?;
        }
    }

    Ok(sub_db)
}

/// Insert a single value into a sub-database B+ tree.
///
/// The sub-DB stores each duplicate value as a "key" with empty data.
/// This function uses the existing B+ tree walk/insert/split machinery
/// to insert into the sub-DB.
fn insert_into_sub_db_tree(
    txn: &mut RwTransaction<'_>,
    sub_db: &mut DbStat,
    val: &[u8],
) -> Result<()> {
    let page_size = txn.env.page_size;
    let cmp: Box<CmpFn> = Box::new(|a: &[u8], b: &[u8]| a.cmp(b));

    if sub_db.root == P_INVALID {
        // Empty sub-DB: create a new root leaf.
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
            val,
            &[],
            0,
            NodeFlags::empty(),
        )?;
        txn.dirty.insert(root_pgno, root_buf);
        sub_db.root = root_pgno;
        sub_db.depth = 1;
        sub_db.leaf_pages = 1;
        sub_db.entries = 1;
        return Ok(());
    }

    // Walk the sub-DB tree from root to leaf, COW-ing each page.
    let (path, leaf_pgno, _is_rightmost) = walk_and_touch(txn, sub_db.root, val, &*cmp)?;

    // Find insertion point.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let (insert_idx, exact) = page_node_search(&leaf_page, val, &*cmp);

    if exact && insert_idx < leaf_page.num_keys() {
        // Value already exists, nothing to do.
        return Ok(());
    }

    // Attempt inline insertion.
    let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
    let add_result = node_add(
        buf.as_mut_slice(),
        page_size,
        insert_idx,
        val,
        &[],
        0,
        NodeFlags::empty(),
    );

    match add_result {
        Ok(()) => {
            sub_db.root = path[0].pgno;
            sub_db.entries += 1;
        }
        Err(Error::PageFull) => {
            split_and_insert_sub_db(txn, sub_db, &path, val, insert_idx, &*cmp)?;
            sub_db.entries += 1;
        }
        Err(e) => return Err(e),
    }

    Ok(())
}

/// Split a full leaf page in a sub-database and insert the new entry.
///
/// This is similar to `split_and_insert` but updates the sub-DB stats
/// instead of `txn.dbs[dbi]`.
fn split_and_insert_sub_db(
    txn: &mut RwTransaction<'_>,
    sub_db: &mut DbStat,
    path: &TreePath,
    val: &[u8],
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
            actual_data_size: None,
        });
    }

    // Insert the new entry.
    entries.insert(
        insert_idx,
        LeafEntry {
            key: val.to_vec(),
            data: Vec::new(),
            flags: NodeFlags::empty(),
            actual_data_size: None,
        },
    );

    let new_entry_size = leaf_entry_size(&entries[insert_idx]);
    let split_idx = find_leaf_split_point(&entries, insert_idx, new_entry_size, page_size);

    // Allocate right sibling leaf.
    let (right_pgno, mut right_buf) = txn.page_alloc()?;
    init_page(
        right_buf.as_mut_slice(),
        right_pgno,
        PageFlags::LEAF | PageFlags::DIRTY,
        page_size,
    );

    // Reinitialize the left leaf page.
    let left_pgno = leaf_pgno;
    let left_buf = txn.dirty.find_mut(left_pgno).ok_or(Error::Corrupted)?;
    init_page(
        left_buf.as_mut_slice(),
        left_pgno,
        PageFlags::LEAF | PageFlags::DIRTY,
        page_size,
    );

    for (i, entry) in entries[..split_idx].iter().enumerate() {
        add_leaf_entry(left_buf.as_mut_slice(), page_size, i, entry)?;
    }

    for (i, entry) in entries[split_idx..].iter().enumerate() {
        add_leaf_entry(right_buf.as_mut_slice(), page_size, i, entry)?;
    }

    let sep_key = entries[split_idx].key.clone();
    txn.dirty.insert(right_pgno, right_buf);

    sub_db.leaf_pages += 1;

    // Propagate separator up to the parent.
    insert_separator_sub_db(
        txn, sub_db, path, leaf_level, &sep_key, left_pgno, right_pgno,
    )?;

    Ok(())
}

/// Insert a separator key into the parent branch for a sub-database split.
///
/// Creates a new root if the split page was the root.
fn insert_separator_sub_db(
    txn: &mut RwTransaction<'_>,
    sub_db: &mut DbStat,
    path: &TreePath,
    split_level: usize,
    sep_key: &[u8],
    left_pgno: u64,
    right_pgno: u64,
) -> Result<()> {
    let page_size = txn.env.page_size;

    if split_level == 0 {
        // The split page was the root: create a new root branch.
        let (new_root_pgno, mut new_root_buf) = txn.page_alloc()?;
        init_page(
            new_root_buf.as_mut_slice(),
            new_root_pgno,
            PageFlags::BRANCH | PageFlags::DIRTY,
            page_size,
        );

        node_add(
            new_root_buf.as_mut_slice(),
            page_size,
            0,
            &[],
            &[],
            left_pgno,
            NodeFlags::empty(),
        )?;

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

        sub_db.root = new_root_pgno;
        sub_db.depth += 1;
        sub_db.branch_pages += 1;

        return Ok(());
    }

    // Insert into existing parent branch.
    let parent_level = split_level - 1;
    let parent_pgno = path[parent_level].pgno;
    let parent_child_idx = path[parent_level].idx;
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
            sub_db.root = path[0].pgno;
            Ok(())
        }
        Err(Error::PageFull) => {
            // Parent branch is full: split it too.
            split_branch_sub_db(
                txn,
                sub_db,
                path,
                parent_level,
                sep_key,
                right_pgno,
                insert_idx,
            )
        }
        Err(e) => Err(e),
    }
}

/// Split a full branch page in a sub-database.
fn split_branch_sub_db(
    txn: &mut RwTransaction<'_>,
    sub_db: &mut DbStat,
    path: &TreePath,
    branch_level: usize,
    sep_key: &[u8],
    child_pgno: u64,
    insert_idx: usize,
) -> Result<()> {
    let page_size = txn.env.page_size;
    let branch_pgno = path[branch_level].pgno;

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

    entries.insert(
        insert_idx,
        BranchEntry {
            key: sep_key.to_vec(),
            child_pgno,
        },
    );

    let new_entry_size = branch_entry_size(&entries[insert_idx]);
    let split_idx = find_branch_split_point(&entries, insert_idx, new_entry_size, page_size);
    let promoted_key = entries[split_idx].key.clone();

    let (right_pgno, mut right_buf) = txn.page_alloc()?;
    init_page(
        right_buf.as_mut_slice(),
        right_pgno,
        PageFlags::BRANCH | PageFlags::DIRTY,
        page_size,
    );

    let left_pgno = branch_pgno;
    let left_buf = txn.dirty.find_mut(left_pgno).ok_or(Error::Corrupted)?;
    init_page(
        left_buf.as_mut_slice(),
        left_pgno,
        PageFlags::BRANCH | PageFlags::DIRTY,
        page_size,
    );

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
    sub_db.branch_pages += 1;

    insert_separator_sub_db(
        txn,
        sub_db,
        path,
        branch_level,
        &promoted_key,
        left_pgno,
        right_pgno,
    )
}

/// Insert a value into an existing sub-database stored in a DUPSORT node.
///
/// Reads the `DbStat` from the node, inserts the value into the sub-DB's
/// B+ tree, then updates the node with the new `DbStat`.
#[allow(clippy::too_many_arguments)]
fn dupsort_put_into_sub_db(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    leaf_pgno: u64,
    insert_idx: usize,
    data: &[u8],
    flags: WriteFlags,
    dcmp: &CmpFn,
) -> Result<()> {
    let page_size = txn.env.page_size;

    // Read the sub-DB record from the node.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let node = leaf_page.node(insert_idx);
    let node_key = node.key().to_vec();
    let mut sub_db = node.sub_db();

    // Check if value already exists by searching the sub-DB.
    if sub_db.root != P_INVALID {
        let found = sub_db_contains(txn, &sub_db, data, dcmp)?;
        if found {
            if flags.contains(WriteFlags::NO_DUP_DATA) {
                return Err(Error::KeyExist);
            }
            return Ok(());
        }
    }

    // Insert the value into the sub-DB tree.
    insert_into_sub_db_tree(txn, &mut sub_db, data)?;

    // Update the node with the new DbStat.
    let db_bytes = db_stat_to_bytes(&sub_db);
    let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
    node_del(buf.as_mut_slice(), page_size, insert_idx);

    let nf = NodeFlags::DUPDATA | NodeFlags::SUBDATA;
    let add_result = node_add(
        buf.as_mut_slice(),
        page_size,
        insert_idx,
        &node_key,
        &db_bytes,
        0,
        nf,
    );

    match add_result {
        Ok(()) => {}
        Err(Error::PageFull) => {
            split_and_insert(
                txn,
                dbi,
                path,
                &node_key,
                &db_bytes,
                nf,
                insert_idx,
                &|a: &[u8], b: &[u8]| a.cmp(b),
            )?;
        }
        Err(e) => return Err(e),
    }

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.root = path[0].pgno;
    db_mut.entries += 1;
    txn.db_dirty[dbi as usize] = true;
    Ok(())
}

/// Check whether a value exists in a sub-database.
fn sub_db_contains(
    txn: &RwTransaction<'_>,
    sub_db: &DbStat,
    val: &[u8],
    _dcmp: &CmpFn,
) -> Result<bool> {
    if sub_db.root == P_INVALID {
        return Ok(false);
    }

    let page_size = txn.env.page_size;
    let cmp: Box<CmpFn> = Box::new(|a: &[u8], b: &[u8]| a.cmp(b));

    // Walk from root to leaf.
    let mut current_pgno = sub_db.root;
    loop {
        let page = read_page(txn, current_pgno, page_size)?;

        if page.is_leaf() {
            let (idx, exact) = page_node_search(&page, val, &*cmp);
            if exact && idx < page.num_keys() {
                return Ok(true);
            }
            return Ok(false);
        }

        if !page.is_branch() {
            return Err(Error::Corrupted);
        }

        let (idx, exact) = page_node_search(&page, val, &*cmp);
        let child_idx = if exact {
            idx
        } else if idx > 0 {
            idx - 1
        } else {
            0
        };

        current_pgno = page.node(child_idx).child_pgno();
    }
}

/// Read a page from the dirty list or mmap (read-only, no COW).
fn read_page<'a>(txn: &'a RwTransaction<'_>, pgno: u64, page_size: usize) -> Result<Page<'a>> {
    if let Some(buf) = txn.dirty.find(pgno) {
        return Ok(buf.as_page());
    }
    let ptr = txn.env.get_page(pgno)?;
    // SAFETY: ptr points into the mmap which is valid for page_size bytes.
    let slice = unsafe { std::slice::from_raw_parts(ptr, page_size) };
    Ok(Page::from_raw(slice))
}

/// Read all values from a sub-database B+ tree.
///
/// Walks the sub-DB tree in order and collects all keys (which are the
/// duplicate values).
fn read_sub_db_values(txn: &RwTransaction<'_>, sub_db: &DbStat) -> Result<Vec<Vec<u8>>> {
    if sub_db.root == P_INVALID {
        return Ok(Vec::new());
    }

    let page_size = txn.env.page_size;
    let mut vals = Vec::with_capacity(sub_db.entries as usize);
    read_sub_db_leaf_values(txn, sub_db.root, page_size, &mut vals)?;
    Ok(vals)
}

/// Recursively collect all values from a sub-database page.
fn read_sub_db_leaf_values(
    txn: &RwTransaction<'_>,
    pgno: u64,
    page_size: usize,
    vals: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let page = read_page(txn, pgno, page_size)?;

    if page.is_leaf() {
        let nkeys = page.num_keys();
        for i in 0..nkeys {
            let node = page.node(i);
            vals.push(node.key().to_vec());
        }
        return Ok(());
    }

    if page.is_branch() {
        let nkeys = page.num_keys();
        for i in 0..nkeys {
            let child_pgno = page.node(i).child_pgno();
            read_sub_db_leaf_values(txn, child_pgno, page_size, vals)?;
        }
    }

    Ok(())
}

/// Count the number of entries in a sub-database.
fn sub_db_count(sub_db: &DbStat) -> u64 {
    sub_db.entries
}

/// Get the first value from a sub-database B+ tree.
///
/// Walks to the leftmost leaf and returns the first key.
#[allow(dead_code)]
fn sub_db_first_value<'a>(
    txn: &'a RwTransaction<'_>,
    sub_db: &DbStat,
    page_size: usize,
) -> Result<Option<&'a [u8]>> {
    if sub_db.root == P_INVALID {
        return Ok(None);
    }

    let mut current_pgno = sub_db.root;
    loop {
        let page = read_page(txn, current_pgno, page_size)?;
        if page.is_leaf() {
            if page.num_keys() == 0 {
                return Ok(None);
            }
            return Ok(Some(page.node(0).key()));
        }
        if !page.is_branch() {
            return Err(Error::Corrupted);
        }
        current_pgno = page.node(0).child_pgno();
    }
}

/// Delete a single dup value from a DUPSORT node.
fn dupsort_del_single(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    leaf_pgno: u64,
    idx: usize,
    del_data: &[u8],
    dcmp: &CmpFn,
) -> Result<()> {
    let page_size = txn.env.page_size;

    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let node = leaf_page.node(idx);
    let node_key = node.key().to_vec();
    let db = *txn.dbs.get(dbi as usize).ok_or(Error::BadDbi)?;
    let dupfixed = db.flags & DatabaseFlags::DUP_FIXED.bits() as u16 != 0;

    if !node.is_dupdata() {
        // Single value -- check if it matches.
        let existing = node.node_data();
        if dcmp(del_data, existing) != Ordering::Equal {
            return Err(Error::NotFound);
        }
        // Delete the entire node (removing last dup = removing key).
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, idx);

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.entries = db_mut.entries.saturating_sub(1);
        txn.db_dirty[dbi as usize] = true;
        return finish_del(txn, dbi, path, leaf_pgno, page_size);
    }

    // Check if this is a sub-database node.
    if node.flags().contains(NodeFlags::SUBDATA) {
        return dupsort_del_single_sub_db(
            txn, dbi, path, leaf_pgno, idx, del_data, &node_key, dupfixed,
        );
    }

    // Has sub-page: find and remove the value.
    let sub_page_data = node.node_data().to_vec();
    let mut vals = read_sub_page_values(&sub_page_data);

    // Find the value to delete.
    let mut found_idx = None;
    for (i, v) in vals.iter().enumerate() {
        if dcmp(del_data, v) == Ordering::Equal {
            found_idx = Some(i);
            break;
        }
    }

    let fi = found_idx.ok_or(Error::NotFound)?;
    vals.remove(fi);

    if vals.is_empty() {
        // No more dups -- remove the entire node.
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, idx);

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.entries = db_mut.entries.saturating_sub(1);
        txn.db_dirty[dbi as usize] = true;
        return finish_del(txn, dbi, path, leaf_pgno, page_size);
    }

    if vals.len() == 1 {
        // Only one value left -- convert back from sub-page to single value.
        let remaining = vals.into_iter().next().ok_or(Error::Corrupted)?;
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, idx);
        node_add(
            buf.as_mut_slice(),
            page_size,
            idx,
            &node_key,
            &remaining,
            0,
            NodeFlags::empty(),
        )?;
    } else {
        // Rebuild sub-page with remaining values.
        let new_sub_page = build_sub_page(&vals, dupfixed);
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, idx);
        node_add(
            buf.as_mut_slice(),
            page_size,
            idx,
            &node_key,
            &new_sub_page,
            0,
            NodeFlags::DUPDATA,
        )?;
    }

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.root = path[0].pgno;
    db_mut.entries = db_mut.entries.saturating_sub(1);
    txn.db_dirty[dbi as usize] = true;
    Ok(())
}

/// Delete a single value from a sub-database.
///
/// Reads all values from the sub-DB, removes the target, and either:
/// - Removes the entire node if no values remain
/// - Converts back to a single value or sub-page if few values remain
/// - Updates the sub-DB if many values remain
#[allow(clippy::too_many_arguments)]
fn dupsort_del_single_sub_db(
    txn: &mut RwTransaction<'_>,
    dbi: u32,
    path: &TreePath,
    leaf_pgno: u64,
    idx: usize,
    del_data: &[u8],
    node_key: &[u8],
    dupfixed: bool,
) -> Result<()> {
    let page_size = txn.env.page_size;
    let node_max = txn.env.node_max;

    // Read the sub-DB and collect all values.
    let leaf_buf = txn.dirty.find(leaf_pgno).ok_or(Error::Corrupted)?;
    let leaf_page = leaf_buf.as_page();
    let node = leaf_page.node(idx);
    let sub_db = node.sub_db();

    let mut vals = read_sub_db_values(txn, &sub_db)?;

    // Find the value to delete.
    let mut found_idx = None;
    for (i, v) in vals.iter().enumerate() {
        if del_data.cmp(v) == Ordering::Equal {
            found_idx = Some(i);
            break;
        }
    }

    let fi = found_idx.ok_or(Error::NotFound)?;
    vals.remove(fi);

    if vals.is_empty() {
        // No more dups: remove the entire node.
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, idx);

        let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
        db_mut.entries = db_mut.entries.saturating_sub(1);
        txn.db_dirty[dbi as usize] = true;
        return finish_del(txn, dbi, path, leaf_pgno, page_size);
    }

    if vals.len() == 1 {
        // Only one value left: convert back to single value.
        let remaining = vals.into_iter().next().ok_or(Error::Corrupted)?;
        let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
        node_del(buf.as_mut_slice(), page_size, idx);
        node_add(
            buf.as_mut_slice(),
            page_size,
            idx,
            node_key,
            &remaining,
            0,
            NodeFlags::empty(),
        )?;
    } else {
        // Try to convert back to sub-page if it fits.
        let sub_page = build_sub_page(&vals, dupfixed);
        if NODE_HEADER_SIZE + node_key.len() + sub_page.len() <= node_max {
            // Fits as sub-page: demote from sub-DB.
            let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
            node_del(buf.as_mut_slice(), page_size, idx);
            node_add(
                buf.as_mut_slice(),
                page_size,
                idx,
                node_key,
                &sub_page,
                0,
                NodeFlags::DUPDATA,
            )?;
        } else {
            // Still too large for sub-page: rebuild as sub-DB.
            let new_sub_db = promote_to_sub_db(txn, &vals, dupfixed)?;
            let db_bytes = db_stat_to_bytes(&new_sub_db);
            let buf = txn.dirty.find_mut(leaf_pgno).ok_or(Error::Corrupted)?;
            node_del(buf.as_mut_slice(), page_size, idx);
            let nf = NodeFlags::DUPDATA | NodeFlags::SUBDATA;
            node_add(
                buf.as_mut_slice(),
                page_size,
                idx,
                node_key,
                &db_bytes,
                0,
                nf,
            )?;
        }
    }

    let db_mut = txn.dbs.get_mut(dbi as usize).ok_or(Error::BadDbi)?;
    db_mut.root = path[0].pgno;
    db_mut.entries = db_mut.entries.saturating_sub(1);
    txn.db_dirty[dbi as usize] = true;
    Ok(())
}

/// Count the number of duplicate values stored in a DUPSORT node.
fn count_dups_in_node(node: &crate::page::Node<'_>) -> u64 {
    if !node.is_dupdata() {
        return 1;
    }
    if node.flags().contains(NodeFlags::SUBDATA) {
        return sub_db_count(&node.sub_db());
    }
    let sp = node.sub_page();
    sp.num_keys() as u64
}

/// Read all duplicate values for a DUPSORT node.
///
/// This is the public API for reading all dups from a node. Used by `get`
/// and cursor operations.
///
/// For sub-database nodes (`DUPDATA | SUBDATA`), this returns an empty vec
/// because walking the sub-DB tree requires transaction access. Use
/// [`read_dup_values_with_txn`] instead when a transaction is available.
pub fn read_dup_values(node: &crate::page::Node<'_>) -> Vec<Vec<u8>> {
    if !node.is_dupdata() {
        // Single value node.
        return vec![node.node_data().to_vec()];
    }
    if node.flags().contains(NodeFlags::SUBDATA) {
        // Sub-database: cannot read without transaction access.
        return Vec::new();
    }
    read_sub_page_values(node.node_data())
}

/// Read all duplicate values for a DUPSORT node, with transaction access
/// for sub-database nodes.
///
/// # Errors
///
/// Returns an error if sub-database pages cannot be read.
pub fn read_dup_values_with_txn(
    node: &crate::page::Node<'_>,
    txn: &RwTransaction<'_>,
) -> Result<Vec<Vec<u8>>> {
    if !node.is_dupdata() {
        return Ok(vec![node.node_data().to_vec()]);
    }
    if node.flags().contains(NodeFlags::SUBDATA) {
        let sub_db = node.sub_db();
        return read_sub_db_values(txn, &sub_db);
    }
    Ok(read_sub_page_values(node.node_data()))
}

/// Search for a specific dup value in a DUPSORT node.
///
/// Returns the dup value if found, or `None` if not found.
///
/// For sub-database nodes, this cannot search without transaction access
/// and returns `None`.
pub fn find_dup_value<'a>(
    node: &crate::page::Node<'a>,
    data: &[u8],
    dcmp: &CmpFn,
) -> Option<&'a [u8]> {
    if !node.is_dupdata() {
        let existing = node.node_data();
        if dcmp(data, existing) == Ordering::Equal {
            return Some(existing);
        }
        return None;
    }

    if node.flags().contains(NodeFlags::SUBDATA) {
        // Sub-database: cannot search without transaction access.
        return None;
    }

    let sp = node.sub_page();
    let nkeys = sp.num_keys();

    if sp.is_leaf2() {
        let val_size = sp.pad() as usize;
        for i in 0..nkeys {
            let val = sp.leaf2_key(i, val_size);
            if dcmp(data, val) == Ordering::Equal {
                return Some(val);
            }
        }
    } else {
        for i in 0..nkeys {
            let sub_node = sp.node(i);
            let val = sub_node.key();
            if dcmp(data, val) == Ordering::Equal {
                return Some(val);
            }
        }
    }

    None
}

/// Get the value at a specific dup index in a DUPSORT node.
///
/// Returns `None` if the index is out of bounds.
///
/// For sub-database nodes, this cannot access the tree without a
/// transaction and returns `None`.
pub fn get_dup_at_index<'a>(node: &crate::page::Node<'a>, dup_idx: usize) -> Option<&'a [u8]> {
    if !node.is_dupdata() {
        if dup_idx == 0 {
            return Some(node.node_data());
        }
        return None;
    }

    if node.flags().contains(NodeFlags::SUBDATA) {
        // Sub-database: cannot access without transaction.
        return None;
    }

    let sp = node.sub_page();
    let nkeys = sp.num_keys();

    if dup_idx >= nkeys {
        return None;
    }

    if sp.is_leaf2() {
        let val_size = sp.pad() as usize;
        Some(sp.leaf2_key(dup_idx, val_size))
    } else {
        Some(sp.node(dup_idx).key())
    }
}

/// Get the number of duplicate values in a DUPSORT node.
pub fn dup_count(node: &crate::page::Node<'_>) -> usize {
    if !node.is_dupdata() {
        return 1;
    }
    if node.flags().contains(NodeFlags::SUBDATA) {
        return sub_db_count(&node.sub_db()) as usize;
    }
    node.sub_page().num_keys()
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
                txn.del(MAIN_DBI, key.as_bytes(), None)
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

    // -------------------------------------------------------------------
    // Overflow page tests
    // -------------------------------------------------------------------

    #[test]
    fn test_should_insert_and_read_overflow_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Insert a value larger than page size (8 KB on 4096-byte pages).
        let big_val = vec![0xAB_u8; 32768];
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI, b"big-key", &big_val, WriteFlags::empty())
                .expect("put overflow value");
            txn.commit().expect("commit");
        }

        // Read it back via read-only transaction.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let got = txn.get(MAIN_DBI, b"big-key").expect("get overflow value");
            assert_eq!(got.len(), 32768);
            assert!(got.iter().all(|&b| b == 0xAB));
        }
    }

    #[test]
    fn test_should_read_overflow_within_write_txn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        let big_val = vec![0xCD_u8; 32768];
        let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
        txn.put(MAIN_DBI, b"within-txn", &big_val, WriteFlags::empty())
            .expect("put");

        // Read within the same write transaction.
        let got = txn.get(MAIN_DBI, b"within-txn").expect("get within txn");
        assert_eq!(got.len(), 32768);
        assert!(got.iter().all(|&b| b == 0xCD));

        txn.commit().expect("commit");
    }

    #[test]
    fn test_should_update_overflow_with_larger_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Insert initial overflow value.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let val = vec![0x11_u8; 32768];
            txn.put(MAIN_DBI, b"upd-key", &val, WriteFlags::empty())
                .expect("put initial");
            txn.commit().expect("commit");
        }

        // Update with a larger overflow value.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let val = vec![0x22_u8; 65536];
            txn.put(MAIN_DBI, b"upd-key", &val, WriteFlags::empty())
                .expect("put larger");
            txn.commit().expect("commit");
        }

        // Verify the updated value.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let got = txn.get(MAIN_DBI, b"upd-key").expect("get updated");
            assert_eq!(got.len(), 65536);
            assert!(got.iter().all(|&b| b == 0x22));
        }
    }

    #[test]
    fn test_should_update_overflow_to_inline() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Insert overflow value.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let val = vec![0x33_u8; 32768];
            txn.put(MAIN_DBI, b"shrink-key", &val, WriteFlags::empty())
                .expect("put overflow");
            txn.commit().expect("commit");
        }

        // Update with a small inline value.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.put(MAIN_DBI, b"shrink-key", b"small", WriteFlags::empty())
                .expect("put inline");
            txn.commit().expect("commit");
        }

        // Verify the inline value.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let got = txn.get(MAIN_DBI, b"shrink-key").expect("get inline");
            assert_eq!(got, b"small");
        }
    }

    #[test]
    fn test_should_delete_overflow_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Insert overflow value.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let val = vec![0x44_u8; 32768];
            txn.put(MAIN_DBI, b"del-key", &val, WriteFlags::empty())
                .expect("put overflow");
            txn.commit().expect("commit");
        }

        // Delete it.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            txn.del(MAIN_DBI, b"del-key", None).expect("del overflow");
            txn.commit().expect("commit");
        }

        // Verify deleted.
        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            let result = txn.get(MAIN_DBI, b"del-key");
            assert!(
                matches!(result, Err(crate::error::Error::NotFound)),
                "expected NotFound, got {result:?}",
            );
        }
    }

    #[test]
    fn test_should_insert_multiple_overflow_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        let sizes = [32768usize, 49152, 65536, 131072];

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            for &size in &sizes {
                let key = format!("multi-{size}");
                let val = vec![(size & 0xFF) as u8; size];
                txn.put(MAIN_DBI, key.as_bytes(), &val, WriteFlags::empty())
                    .expect("put");
            }
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for &size in &sizes {
                let key = format!("multi-{size}");
                let got = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
                assert_eq!(got.len(), size, "size mismatch for {key}");
                assert!(
                    got.iter().all(|&b| b == (size & 0xFF) as u8),
                    "content mismatch for {key}",
                );
            }
        }
    }

    #[test]
    fn test_should_mix_inline_and_overflow_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            // Insert mix of inline and overflow values.
            for i in 0..100u32 {
                let key = format!("mix-{i:04}");
                let size = if i % 5 == 0 { 32768 } else { 100 };
                let val = vec![(i & 0xFF) as u8; size];
                txn.put(MAIN_DBI, key.as_bytes(), &val, WriteFlags::empty())
                    .expect("put");
            }
            txn.commit().expect("commit");
        }

        {
            let txn = env.begin_ro_txn().expect("begin_ro_txn");
            for i in 0..100u32 {
                let key = format!("mix-{i:04}");
                let expected_size = if i % 5 == 0 { 32768 } else { 100 };
                let got = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
                assert_eq!(got.len(), expected_size, "size mismatch for {key}");
                assert!(
                    got.iter().all(|&b| b == (i & 0xFF) as u8),
                    "content mismatch for {key}",
                );
            }
        }
    }

    #[test]
    fn test_should_track_overflow_pages_in_stats() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env = test_env(dir.path());

        // Insert a value guaranteed to exceed node_max (which depends on
        // the OS page size). Use a large enough value to trigger overflow
        // on all platforms.
        {
            let mut txn = env.begin_rw_txn().expect("begin_rw_txn");
            let node_max = txn.env.node_max;
            // Ensure the value exceeds node_max.
            let val_size = node_max + 100;
            let val = vec![0xEE_u8; val_size];

            txn.put(MAIN_DBI, b"stat-key", &val, WriteFlags::empty())
                .expect("put");

            // Check the overflow_pages counter within the transaction.
            let db = txn.dbs[MAIN_DBI as usize];
            assert!(
                db.overflow_pages > 0,
                "overflow_pages should be > 0 within txn, got {}",
                db.overflow_pages,
            );
            txn.commit().expect("commit");
        }
    }
}
