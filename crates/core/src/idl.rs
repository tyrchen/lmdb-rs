//! Sorted ID list for free page tracking.
//!
//! This module implements the IDL (ID List) data structures from LMDB's `midl.c`.
//! IDs are page numbers represented as `u64`.
//!
//! - [`IdList`] maintains page IDs in **descending** order (largest first), used for free page
//!   tracking.
//! - [`Id2List`] maintains `(ID, index)` pairs in **ascending** order, used for dirty page
//!   tracking.

/// A sorted list of page IDs in descending order.
///
/// Used for free page tracking. The descending order matches LMDB's convention
/// where the largest page number is at index 0.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::idl::IdList;
///
/// let mut idl = IdList::new();
/// idl.append(5);
/// idl.append(10);
/// idl.append(3);
/// idl.sort();
/// assert_eq!(idl.as_slice(), &[10, 5, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct IdList {
    ids: Vec<u64>,
}

impl IdList {
    /// Creates an empty ID list.
    pub fn new() -> Self {
        Self { ids: Vec::new() }
    }

    /// Creates an empty ID list with the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            ids: Vec::with_capacity(cap),
        }
    }

    /// Returns the number of IDs in the list.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns `true` if the list contains no IDs.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Removes all IDs from the list.
    pub fn clear(&mut self) {
        self.ids.clear();
    }

    /// Returns the ID at the given index (0-indexed into the sorted list).
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    pub fn get(&self, idx: usize) -> u64 {
        self.ids[idx]
    }

    /// Returns the list as a slice.
    pub fn as_slice(&self) -> &[u64] {
        &self.ids
    }

    /// Binary search in the descending-sorted list.
    ///
    /// Returns the index of the first element `>= id` (in descending order,
    /// meaning the first element whose value is `<= id`). If all elements are
    /// greater than `id`, returns `len`.
    ///
    /// This matches LMDB's `mdb_midl_search` behavior: it returns the position
    /// where `id` would be inserted to maintain descending order.
    pub fn search(&self, id: u64) -> usize {
        // Binary search for descending order.
        // We want the first index where ids[index] <= id.
        let mut lo = 0;
        let mut hi = self.ids.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.ids[mid] > id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Appends an ID without maintaining sort order.
    ///
    /// Call [`sort`](Self::sort) after all appends to restore descending order.
    pub fn append(&mut self, id: u64) {
        self.ids.push(id);
    }

    /// Appends a range of IDs `[start_id, start_id + count)` without sorting.
    ///
    /// Call [`sort`](Self::sort) after all appends to restore descending order.
    pub fn append_range(&mut self, start_id: u64, count: u32) {
        self.ids.reserve(count as usize);
        for i in 0..u64::from(count) {
            self.ids.push(start_id + i);
        }
    }

    /// Sorts the list in descending order (largest first).
    pub fn sort(&mut self) {
        self.ids.sort_unstable_by(|a, b| b.cmp(a));
    }

    /// Merges another sorted (descending) ID list into this one.
    ///
    /// Both lists must already be sorted in descending order. Uses an O(n+m)
    /// merge algorithm that works from back to front, matching LMDB's
    /// `mdb_midl_xmerge`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmdb_rs_core::idl::IdList;
    ///
    /// let mut a = IdList::new();
    /// a.append(10);
    /// a.append(5);
    /// a.append(2);
    /// // a is already descending: [10, 5, 2]
    ///
    /// let mut b = IdList::new();
    /// b.append(8);
    /// b.append(3);
    /// // b is already descending: [8, 3]
    ///
    /// a.merge(&b);
    /// assert_eq!(a.as_slice(), &[10, 8, 5, 3, 2]);
    /// ```
    pub fn merge(&mut self, other: &IdList) {
        if other.is_empty() {
            return;
        }

        let a_len = self.ids.len();
        let b_len = other.ids.len();
        let total = a_len + b_len;

        // Extend self to make room for merged result.
        self.ids.resize(total, 0);

        // Merge from back to front (smallest values first).
        let mut i = a_len; // pointer into original self (from end)
        let mut j = b_len; // pointer into other (from end)
        let mut k = total; // write pointer (from end)

        while i > 0 && j > 0 {
            k -= 1;
            if self.ids[i - 1] < other.ids[j - 1] {
                self.ids[k] = self.ids[i - 1];
                i -= 1;
            } else {
                self.ids[k] = other.ids[j - 1];
                j -= 1;
            }
        }

        // Copy remaining elements from other (self elements are already in place).
        while j > 0 {
            k -= 1;
            self.ids[k] = other.ids[j - 1];
            j -= 1;
        }
        // If i > 0, those elements are already in the correct position (indices 0..i == 0..k).
    }

    /// Returns `true` if the list contains the given ID.
    ///
    /// Uses binary search; the list must be sorted in descending order.
    pub fn contains(&self, id: u64) -> bool {
        let idx = self.search(id);
        idx < self.ids.len() && self.ids[idx] == id
    }

    /// Extends the list with IDs from a slice without sorting.
    ///
    /// Call [`sort`](Self::sort) after all extends to restore descending order.
    pub fn extend_from_slice(&mut self, ids: &[u64]) {
        self.ids.extend_from_slice(ids);
    }
}

impl Default for IdList {
    fn default() -> Self {
        Self::new()
    }
}

/// An entry in an [`Id2List`], pairing a page ID with an index.
#[derive(Debug, Clone, Copy)]
pub struct Id2Entry {
    /// The page ID.
    pub mid: u64,
    /// An associated index or pointer-sized value.
    pub idx: usize,
}

/// A sorted list of `(ID, index)` pairs in ascending order.
///
/// Used for dirty page tracking. Entries are maintained in ascending order
/// by their `mid` (page ID) field.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::idl::{Id2List, Id2Entry};
///
/// let mut list = Id2List::new();
/// assert!(list.insert(Id2Entry { mid: 5, idx: 0 }));
/// assert!(list.insert(Id2Entry { mid: 3, idx: 1 }));
/// assert_eq!(list.get(0).mid, 3);
/// assert_eq!(list.get(1).mid, 5);
/// ```
#[derive(Debug, Clone)]
pub struct Id2List {
    entries: Vec<Id2Entry>,
}

impl Id2List {
    /// Creates an empty ID2 list.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Creates an empty ID2 list with the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
        }
    }

    /// Returns the number of entries in the list.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the list contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes all entries from the list.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Binary search for `id` in the ascending-sorted list.
    ///
    /// Returns the index where `id` is found or where it would be inserted
    /// to maintain ascending order.
    pub fn search(&self, id: u64) -> usize {
        let mut lo = 0;
        let mut hi = self.entries.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.entries[mid].mid < id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Inserts an entry in sorted (ascending) order.
    ///
    /// Returns `true` if the entry was inserted, or `false` if an entry
    /// with the same `mid` already exists (no insertion performed).
    pub fn insert(&mut self, entry: Id2Entry) -> bool {
        let pos = self.search(entry.mid);
        if pos < self.entries.len() && self.entries[pos].mid == entry.mid {
            return false;
        }
        self.entries.insert(pos, entry);
        true
    }

    /// Returns a reference to the entry at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    pub fn get(&self, idx: usize) -> &Id2Entry {
        &self.entries[idx]
    }

    /// Searches for an entry with the given `id` and returns it if found.
    pub fn get_by_id(&self, id: u64) -> Option<&Id2Entry> {
        let pos = self.search(id);
        if pos < self.entries.len() && self.entries[pos].mid == id {
            Some(&self.entries[pos])
        } else {
            None
        }
    }

    /// Returns the entries as a slice.
    pub fn as_slice(&self) -> &[Id2Entry] {
        &self.entries
    }
}

impl Default for Id2List {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== IdList tests ====================

    #[test]
    fn test_should_create_empty_idlist() {
        let idl = IdList::new();
        assert!(idl.is_empty());
        assert_eq!(idl.len(), 0);
    }

    #[test]
    fn test_should_create_idlist_with_capacity() {
        let idl = IdList::with_capacity(100);
        assert!(idl.is_empty());
        assert_eq!(idl.len(), 0);
    }

    #[test]
    fn test_should_append_single_element() {
        let mut idl = IdList::new();
        idl.append(42);
        assert_eq!(idl.len(), 1);
        assert_eq!(idl.get(0), 42);
    }

    #[test]
    fn test_should_sort_descending() {
        let mut idl = IdList::new();
        idl.append(3);
        idl.append(10);
        idl.append(1);
        idl.append(7);
        idl.sort();
        assert_eq!(idl.as_slice(), &[10, 7, 3, 1]);
    }

    #[test]
    fn test_should_sort_already_sorted() {
        let mut idl = IdList::new();
        idl.append(10);
        idl.append(5);
        idl.append(1);
        idl.sort();
        assert_eq!(idl.as_slice(), &[10, 5, 1]);
    }

    #[test]
    fn test_should_sort_empty_list() {
        let mut idl = IdList::new();
        idl.sort();
        assert!(idl.is_empty());
    }

    #[test]
    fn test_should_search_in_descending_list() {
        let mut idl = IdList::new();
        idl.append(10);
        idl.append(7);
        idl.append(5);
        idl.append(3);
        idl.append(1);
        // Already in descending order

        // Exact matches
        assert_eq!(idl.search(10), 0);
        assert_eq!(idl.search(7), 1);
        assert_eq!(idl.search(5), 2);
        assert_eq!(idl.search(3), 3);
        assert_eq!(idl.search(1), 4);

        // Value between existing elements
        assert_eq!(idl.search(8), 1); // between 10 and 7
        assert_eq!(idl.search(4), 3); // between 5 and 3

        // Value larger than all
        assert_eq!(idl.search(100), 0);

        // Value smaller than all
        assert_eq!(idl.search(0), 5);
    }

    #[test]
    fn test_should_search_empty_list() {
        let idl = IdList::new();
        assert_eq!(idl.search(42), 0);
    }

    #[test]
    fn test_should_search_single_element() {
        let mut idl = IdList::new();
        idl.append(5);

        assert_eq!(idl.search(5), 0); // exact match
        assert_eq!(idl.search(10), 0); // larger
        assert_eq!(idl.search(1), 1); // smaller
    }

    #[test]
    fn test_should_contain_existing_ids() {
        let mut idl = IdList::new();
        idl.append(10);
        idl.append(5);
        idl.append(3);
        // Already descending

        assert!(idl.contains(10));
        assert!(idl.contains(5));
        assert!(idl.contains(3));
        assert!(!idl.contains(7));
        assert!(!idl.contains(0));
        assert!(!idl.contains(11));
    }

    #[test]
    fn test_should_not_contain_in_empty_list() {
        let idl = IdList::new();
        assert!(!idl.contains(1));
    }

    #[test]
    fn test_should_append_range() {
        let mut idl = IdList::new();
        idl.append_range(5, 4);
        assert_eq!(idl.len(), 4);
        // Unsorted: [5, 6, 7, 8]
        idl.sort();
        assert_eq!(idl.as_slice(), &[8, 7, 6, 5]);
    }

    #[test]
    fn test_should_append_range_zero_count() {
        let mut idl = IdList::new();
        idl.append_range(5, 0);
        assert!(idl.is_empty());
    }

    #[test]
    fn test_should_append_range_single() {
        let mut idl = IdList::new();
        idl.append_range(42, 1);
        assert_eq!(idl.len(), 1);
        assert_eq!(idl.get(0), 42);
    }

    #[test]
    fn test_should_merge_two_sorted_lists() {
        let mut a = IdList::new();
        a.append(10);
        a.append(5);
        a.append(2);

        let mut b = IdList::new();
        b.append(8);
        b.append(3);

        a.merge(&b);
        assert_eq!(a.as_slice(), &[10, 8, 5, 3, 2]);
    }

    #[test]
    fn test_should_merge_with_empty_other() {
        let mut a = IdList::new();
        a.append(10);
        a.append(5);

        let b = IdList::new();
        a.merge(&b);
        assert_eq!(a.as_slice(), &[10, 5]);
    }

    #[test]
    fn test_should_merge_empty_self_with_nonempty() {
        let mut a = IdList::new();

        let mut b = IdList::new();
        b.append(8);
        b.append(3);

        a.merge(&b);
        assert_eq!(a.as_slice(), &[8, 3]);
    }

    #[test]
    fn test_should_merge_both_empty() {
        let mut a = IdList::new();
        let b = IdList::new();
        a.merge(&b);
        assert!(a.is_empty());
    }

    #[test]
    fn test_should_merge_interleaved() {
        let mut a = IdList::new();
        a.append(9);
        a.append(7);
        a.append(5);
        a.append(3);
        a.append(1);

        let mut b = IdList::new();
        b.append(10);
        b.append(8);
        b.append(6);
        b.append(4);
        b.append(2);

        a.merge(&b);
        assert_eq!(a.as_slice(), &[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_should_merge_when_other_all_larger() {
        let mut a = IdList::new();
        a.append(3);
        a.append(2);
        a.append(1);

        let mut b = IdList::new();
        b.append(6);
        b.append(5);
        b.append(4);

        a.merge(&b);
        assert_eq!(a.as_slice(), &[6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_should_merge_when_other_all_smaller() {
        let mut a = IdList::new();
        a.append(6);
        a.append(5);
        a.append(4);

        let mut b = IdList::new();
        b.append(3);
        b.append(2);
        b.append(1);

        a.merge(&b);
        assert_eq!(a.as_slice(), &[6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_should_clear_idlist() {
        let mut idl = IdList::new();
        idl.append(1);
        idl.append(2);
        idl.clear();
        assert!(idl.is_empty());
    }

    #[test]
    fn test_should_extend_from_slice() {
        let mut idl = IdList::new();
        idl.extend_from_slice(&[3, 1, 4, 1, 5]);
        assert_eq!(idl.len(), 5);
        idl.sort();
        assert_eq!(idl.as_slice(), &[5, 4, 3, 1, 1]);
    }

    #[test]
    fn test_should_default_to_empty() {
        let idl = IdList::default();
        assert!(idl.is_empty());
    }

    // ==================== Id2List tests ====================

    #[test]
    fn test_should_create_empty_id2list() {
        let list = Id2List::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_should_create_id2list_with_capacity() {
        let list = Id2List::with_capacity(100);
        assert!(list.is_empty());
    }

    #[test]
    fn test_should_insert_maintaining_ascending_order() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 5, idx: 0 }));
        assert!(list.insert(Id2Entry { mid: 3, idx: 1 }));
        assert!(list.insert(Id2Entry { mid: 8, idx: 2 }));
        assert!(list.insert(Id2Entry { mid: 1, idx: 3 }));

        assert_eq!(list.len(), 4);
        assert_eq!(list.get(0).mid, 1);
        assert_eq!(list.get(1).mid, 3);
        assert_eq!(list.get(2).mid, 5);
        assert_eq!(list.get(3).mid, 8);
    }

    #[test]
    fn test_should_reject_duplicate_id() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 5, idx: 0 }));
        assert!(!list.insert(Id2Entry { mid: 5, idx: 1 }));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_should_search_ascending() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 2, idx: 0 }));
        assert!(list.insert(Id2Entry { mid: 5, idx: 1 }));
        assert!(list.insert(Id2Entry { mid: 8, idx: 2 }));

        // Exact matches
        assert_eq!(list.search(2), 0);
        assert_eq!(list.search(5), 1);
        assert_eq!(list.search(8), 2);

        // Insertion points
        assert_eq!(list.search(1), 0); // before all
        assert_eq!(list.search(3), 1); // between 2 and 5
        assert_eq!(list.search(10), 3); // after all
    }

    #[test]
    fn test_should_search_empty_id2list() {
        let list = Id2List::new();
        assert_eq!(list.search(42), 0);
    }

    #[test]
    fn test_should_get_by_id_exact_match() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 3, idx: 10 }));
        assert!(list.insert(Id2Entry { mid: 7, idx: 20 }));

        let entry = list.get_by_id(3);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.mid, 3);
        assert_eq!(entry.idx, 10);
    }

    #[test]
    fn test_should_return_none_for_missing_id() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 3, idx: 10 }));

        assert!(list.get_by_id(5).is_none());
        assert!(list.get_by_id(0).is_none());
    }

    #[test]
    fn test_should_return_none_for_empty_id2list() {
        let list = Id2List::new();
        assert!(list.get_by_id(1).is_none());
    }

    #[test]
    fn test_should_clear_id2list() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 1, idx: 0 }));
        assert!(list.insert(Id2Entry { mid: 2, idx: 1 }));
        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn test_should_return_as_slice() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 1, idx: 10 }));
        assert!(list.insert(Id2Entry { mid: 5, idx: 20 }));

        let slice = list.as_slice();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].mid, 1);
        assert_eq!(slice[1].mid, 5);
    }

    #[test]
    fn test_should_preserve_idx_values() {
        let mut list = Id2List::new();
        assert!(list.insert(Id2Entry { mid: 100, idx: 42 }));
        assert!(list.insert(Id2Entry { mid: 50, idx: 99 }));

        // Sorted ascending by mid, idx preserved
        assert_eq!(list.get(0).mid, 50);
        assert_eq!(list.get(0).idx, 99);
        assert_eq!(list.get(1).mid, 100);
        assert_eq!(list.get(1).idx, 42);
    }

    #[test]
    fn test_should_default_to_empty_id2list() {
        let list = Id2List::default();
        assert!(list.is_empty());
    }
}
