//! Key comparison functions for B+ tree operations.

use std::cmp::Ordering;

/// Comparison function type alias.
pub type CmpFn = dyn Fn(&[u8], &[u8]) -> Ordering + Send + Sync;

/// Select the default comparison function for a database based on its flags.
///
/// Database flags are `u16` from `DbStat.flags`:
/// - `REVERSE_KEY` = 0x02 — reverse lexicographic
/// - `INTEGER_KEY` = 0x08 — native integer
/// - Default — lexicographic
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::cmp::default_cmp;
/// let cmp = default_cmp(0x00);
/// assert_eq!(cmp(b"abc", b"abd"), std::cmp::Ordering::Less);
/// ```
pub fn default_cmp(db_flags: u16) -> Box<CmpFn> {
    if db_flags & 0x08 != 0 {
        Box::new(cmp_int)
    } else if db_flags & 0x02 != 0 {
        Box::new(cmp_reverse)
    } else {
        Box::new(cmp_lexicographic)
    }
}

/// Select the default data comparison for `DUPSORT` databases.
///
/// - `REVERSE_DUP` = 0x40 — reverse
/// - `INTEGER_DUP` = 0x20 — integer
/// - Default — lexicographic
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::cmp::default_dcmp;
/// let cmp = default_dcmp(0x00);
/// assert_eq!(cmp(b"abc", b"abd"), std::cmp::Ordering::Less);
/// ```
pub fn default_dcmp(db_flags: u16) -> Box<CmpFn> {
    if db_flags & 0x20 != 0 {
        Box::new(cmp_int)
    } else if db_flags & 0x40 != 0 {
        Box::new(cmp_reverse)
    } else {
        Box::new(cmp_lexicographic)
    }
}

/// Standard lexicographic comparison (`memcmp` with length tiebreaker).
///
/// This is the default comparison for LMDB keys. Rust's `&[u8]` `Ord`
/// implementation is exactly `memcmp` followed by a length comparison.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::cmp::cmp_lexicographic;
/// use std::cmp::Ordering;
/// assert_eq!(cmp_lexicographic(b"abc", b"abd"), Ordering::Less);
/// assert_eq!(cmp_lexicographic(b"abc", b"abc"), Ordering::Equal);
/// assert_eq!(cmp_lexicographic(b"ab", b"abc"), Ordering::Less);
/// ```
pub fn cmp_lexicographic(a: &[u8], b: &[u8]) -> Ordering {
    a.cmp(b)
}

/// Reverse byte-order comparison (compares from end to beginning).
///
/// Compares bytes starting from the last byte of each slice, working
/// toward the first byte. When all overlapping bytes are equal, the
/// longer slice is considered greater.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::cmp::cmp_reverse;
/// use std::cmp::Ordering;
/// assert_eq!(cmp_reverse(b"\x00\x01", b"\x00\x02"), Ordering::Less);
/// ```
pub fn cmp_reverse(a: &[u8], b: &[u8]) -> Ordering {
    let len = a.len().min(b.len());
    for i in 0..len {
        match a[a.len() - 1 - i].cmp(&b[b.len() - 1 - i]) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}

/// Native unsigned integer comparison.
///
/// Supports 4-byte (`u32`) and 8-byte (`u64`) integers in native byte
/// order. For other sizes, falls back to lexicographic comparison.
///
/// # Examples
///
/// ```
/// use lmdb_rs_core::cmp::cmp_int;
/// use std::cmp::Ordering;
/// let a = 42u32.to_ne_bytes();
/// let b = 100u32.to_ne_bytes();
/// assert_eq!(cmp_int(&a, &b), Ordering::Less);
/// ```
pub fn cmp_int(a: &[u8], b: &[u8]) -> Ordering {
    match a.len() {
        4 => {
            let va = u32::from_ne_bytes([a[0], a[1], a[2], a[3]]);
            let vb = u32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
            va.cmp(&vb)
        }
        8 => {
            let va = u64::from_ne_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]);
            let vb = u64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            va.cmp(&vb)
        }
        _ => cmp_lexicographic(a, b),
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;

    // --- cmp_lexicographic ---

    #[test]
    fn test_should_compare_empty_slices_as_equal() {
        assert_eq!(cmp_lexicographic(b"", b""), Ordering::Equal);
    }

    #[test]
    fn test_should_compare_empty_less_than_nonempty() {
        assert_eq!(cmp_lexicographic(b"", b"\x00"), Ordering::Less);
        assert_eq!(cmp_lexicographic(b"\x00", b""), Ordering::Greater);
    }

    #[test]
    fn test_should_compare_single_bytes() {
        assert_eq!(cmp_lexicographic(b"\x00", b"\x01"), Ordering::Less);
        assert_eq!(cmp_lexicographic(b"\xff", b"\x00"), Ordering::Greater);
        assert_eq!(cmp_lexicographic(b"\x42", b"\x42"), Ordering::Equal);
    }

    #[test]
    fn test_should_compare_equal_length_slices() {
        assert_eq!(cmp_lexicographic(b"abc", b"abd"), Ordering::Less);
        assert_eq!(cmp_lexicographic(b"xyz", b"xyz"), Ordering::Equal);
        assert_eq!(cmp_lexicographic(b"abd", b"abc"), Ordering::Greater);
    }

    #[test]
    fn test_should_compare_different_length_slices() {
        assert_eq!(cmp_lexicographic(b"ab", b"abc"), Ordering::Less);
        assert_eq!(cmp_lexicographic(b"abc", b"ab"), Ordering::Greater);
        assert_eq!(cmp_lexicographic(b"a", b"aa"), Ordering::Less);
    }

    // --- cmp_reverse ---

    #[test]
    fn test_should_reverse_compare_empty_slices() {
        assert_eq!(cmp_reverse(b"", b""), Ordering::Equal);
    }

    #[test]
    fn test_should_reverse_compare_empty_vs_nonempty() {
        assert_eq!(cmp_reverse(b"", b"\x00"), Ordering::Less);
        assert_eq!(cmp_reverse(b"\x00", b""), Ordering::Greater);
    }

    #[test]
    fn test_should_reverse_compare_from_end() {
        // Last bytes differ: 0x02 > 0x01
        assert_eq!(cmp_reverse(b"\x00\x02", b"\x00\x01"), Ordering::Greater);
        // Last bytes equal, second-to-last differs
        assert_eq!(cmp_reverse(b"\x01\x00", b"\x02\x00"), Ordering::Less);
    }

    #[test]
    fn test_should_reverse_compare_different_lengths() {
        // Overlapping bytes from end are equal, longer is greater
        assert_eq!(cmp_reverse(b"\x01\x02", b"\x02"), Ordering::Greater);
    }

    #[test]
    fn test_should_reverse_compare_single_bytes() {
        assert_eq!(cmp_reverse(b"\x05", b"\x03"), Ordering::Greater);
        assert_eq!(cmp_reverse(b"\x03", b"\x05"), Ordering::Less);
        assert_eq!(cmp_reverse(b"\x07", b"\x07"), Ordering::Equal);
    }

    // --- cmp_int ---

    #[test]
    fn test_should_compare_u32_integers() {
        let a = 42u32.to_ne_bytes();
        let b = 100u32.to_ne_bytes();
        assert_eq!(cmp_int(&a, &b), Ordering::Less);
        assert_eq!(cmp_int(&b, &a), Ordering::Greater);
        assert_eq!(cmp_int(&a, &a), Ordering::Equal);
    }

    #[test]
    fn test_should_compare_u32_boundaries() {
        let zero = 0u32.to_ne_bytes();
        let max = u32::MAX.to_ne_bytes();
        assert_eq!(cmp_int(&zero, &max), Ordering::Less);
        assert_eq!(cmp_int(&max, &zero), Ordering::Greater);
    }

    #[test]
    fn test_should_compare_u64_integers() {
        let a = 1000u64.to_ne_bytes();
        let b = 2000u64.to_ne_bytes();
        assert_eq!(cmp_int(&a, &b), Ordering::Less);
        assert_eq!(cmp_int(&b, &a), Ordering::Greater);
        assert_eq!(cmp_int(&a, &a), Ordering::Equal);
    }

    #[test]
    fn test_should_compare_u64_boundaries() {
        let zero = 0u64.to_ne_bytes();
        let max = u64::MAX.to_ne_bytes();
        assert_eq!(cmp_int(&zero, &max), Ordering::Less);
        assert_eq!(cmp_int(&max, &zero), Ordering::Greater);
    }

    #[test]
    fn test_should_fallback_to_lexicographic_for_other_sizes() {
        // 3-byte values fall back to lexicographic
        assert_eq!(cmp_int(b"\x01\x02\x03", b"\x01\x02\x04"), Ordering::Less);
        // 1-byte
        assert_eq!(cmp_int(b"\xff", b"\x00"), Ordering::Greater);
        // Empty
        assert_eq!(cmp_int(b"", b""), Ordering::Equal);
    }

    // --- default_cmp ---

    #[test]
    fn test_should_select_lexicographic_by_default() {
        let cmp = default_cmp(0x00);
        assert_eq!(cmp(b"abc", b"abd"), Ordering::Less);
    }

    #[test]
    fn test_should_select_reverse_for_reverse_key_flag() {
        let cmp = default_cmp(0x02);
        // Reverse comparison: compares from end
        assert_eq!(cmp(b"\x00\x02", b"\x00\x01"), Ordering::Greater);
    }

    #[test]
    fn test_should_select_integer_for_integer_key_flag() {
        let cmp = default_cmp(0x08);
        let a = 10u32.to_ne_bytes();
        let b = 20u32.to_ne_bytes();
        assert_eq!(cmp(&a, &b), Ordering::Less);
    }

    #[test]
    fn test_should_prefer_integer_over_reverse_when_both_set() {
        // INTEGER_KEY (0x08) is checked first
        let cmp = default_cmp(0x08 | 0x02);
        let a = 10u32.to_ne_bytes();
        let b = 20u32.to_ne_bytes();
        assert_eq!(cmp(&a, &b), Ordering::Less);
    }

    // --- default_dcmp ---

    #[test]
    fn test_should_select_lexicographic_dcmp_by_default() {
        let cmp = default_dcmp(0x00);
        assert_eq!(cmp(b"abc", b"abd"), Ordering::Less);
    }

    #[test]
    fn test_should_select_integer_dcmp_for_integer_dup_flag() {
        let cmp = default_dcmp(0x20);
        let a = 5u32.to_ne_bytes();
        let b = 15u32.to_ne_bytes();
        assert_eq!(cmp(&a, &b), Ordering::Less);
    }

    #[test]
    fn test_should_select_reverse_dcmp_for_reverse_dup_flag() {
        let cmp = default_dcmp(0x40);
        assert_eq!(cmp(b"\x00\x02", b"\x00\x01"), Ordering::Greater);
    }

    #[test]
    fn test_should_prefer_integer_dup_over_reverse_dup() {
        let cmp = default_dcmp(0x20 | 0x40);
        let a = 5u32.to_ne_bytes();
        let b = 15u32.to_ne_bytes();
        assert_eq!(cmp(&a, &b), Ordering::Less);
    }
}
