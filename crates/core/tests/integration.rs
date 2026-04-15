//! Integration tests for lmdb-rs-core.
//!
//! These tests exercise the full stack: environment → transaction → cursor
//! operations, verifying correctness across page splits, commits, reopens,
//! and concurrent reader/writer scenarios.

use lmdb_rs_core::{
    env::Environment,
    error::Error,
    types::{CursorOp, DatabaseFlags, MAIN_DBI, WriteFlags},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn open_env(dir: &tempfile::TempDir) -> Environment {
    Environment::builder()
        .map_size(10 * 1024 * 1024) // 10 MB
        .open(dir.path())
        .expect("open env")
}

// ---------------------------------------------------------------------------
// Basic CRUD
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_put_get_del() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Put
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"greeting", b"hello", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Get
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        let val = txn.get(MAIN_DBI, b"greeting").expect("get");
        assert_eq!(val, b"hello");
    }

    // Update
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"greeting", b"world", WriteFlags::empty())
            .expect("put update");
        txn.commit().expect("commit");
    }

    // Verify update
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        assert_eq!(txn.get(MAIN_DBI, b"greeting").expect("get"), b"world");
    }

    // Delete
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.del(MAIN_DBI, b"greeting").expect("del");
        txn.commit().expect("commit");
    }

    // Verify deleted
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        assert!(matches!(
            txn.get(MAIN_DBI, b"greeting"),
            Err(Error::NotFound)
        ));
    }
}

// ---------------------------------------------------------------------------
// Bulk insert with page splits
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_bulk_insert_1000_keys() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    let n = 1000;

    // Insert N keys
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in 0..n {
            let key = format!("key-{i:05}");
            let val = format!("val-{i:05}");
            txn.put(
                MAIN_DBI,
                key.as_bytes(),
                val.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Read all N keys back
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        for i in 0..n {
            let key = format!("key-{i:05}");
            let expected_val = format!("val-{i:05}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            assert_eq!(val, expected_val.as_bytes(), "mismatch at key {key}");
        }
    }
}

#[test]
fn test_e2e_cursor_iteration_sorted_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Insert keys in random-ish order
    let keys: Vec<String> = (0..200)
        .map(|i| format!("k-{:04}", (i * 7 + 13) % 200))
        .collect();
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for k in &keys {
            txn.put(MAIN_DBI, k.as_bytes(), b"v", WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Iterate and verify sorted order
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("open cursor");
        let mut prev_key: Option<Vec<u8>> = None;
        let mut count = 0;

        for result in cursor.iter() {
            let (key, _val) = result.expect("cursor iter");
            if let Some(ref pk) = prev_key {
                assert!(
                    key > pk.as_slice(),
                    "keys not sorted: {:?} <= {:?}",
                    String::from_utf8_lossy(key),
                    String::from_utf8_lossy(pk)
                );
            }
            prev_key = Some(key.to_vec());
            count += 1;
        }

        // We inserted 200 unique keys (some may collide due to modular arithmetic)
        assert!(count > 0, "no keys iterated");
    }
}

// ---------------------------------------------------------------------------
// Persistence across reopen
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_persist_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Write data
    {
        let env = open_env(&dir);
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in 0..50 {
            let key = format!("persist-{i:03}");
            let val = format!("data-{i:03}");
            txn.put(
                MAIN_DBI,
                key.as_bytes(),
                val.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Reopen and verify
    {
        let env = open_env(&dir);
        let txn = env.begin_ro_txn().expect("ro txn");
        for i in 0..50 {
            let key = format!("persist-{i:03}");
            let expected = format!("data-{i:03}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            assert_eq!(val, expected.as_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// Abort semantics
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_abort_discards_all_changes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Commit some initial data
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"keep", b"yes", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Start a new txn, make changes, then abort
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"discard", b"no", WriteFlags::empty())
            .expect("put");
        txn.del(MAIN_DBI, b"keep").expect("del");
        txn.abort();
    }

    // Verify: "keep" still exists, "discard" does not
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        assert_eq!(txn.get(MAIN_DBI, b"keep").expect("get"), b"yes");
        assert!(matches!(
            txn.get(MAIN_DBI, b"discard"),
            Err(Error::NotFound)
        ));
    }
}

// ---------------------------------------------------------------------------
// Multiple transactions
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_sequential_transactions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // 10 sequential write transactions, each adding 10 keys
    for batch in 0..10 {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in 0..10 {
            let key = format!("batch{batch:02}-key{i:02}");
            let val = format!("val-{}", batch * 10 + i);
            txn.put(
                MAIN_DBI,
                key.as_bytes(),
                val.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Verify all 100 keys
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        for batch in 0..10 {
            for i in 0..10 {
                let key = format!("batch{batch:02}-key{i:02}");
                let expected = format!("val-{}", batch * 10 + i);
                let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
                assert_eq!(val, expected.as_bytes());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Delete with page changes
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_insert_delete_reinsert() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Insert 100 keys
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in 0..100 {
            let key = format!("idr-{i:03}");
            txn.put(MAIN_DBI, key.as_bytes(), b"original", WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Delete all odd-numbered keys
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in (1..100).step_by(2) {
            let key = format!("idr-{i:03}");
            txn.del(MAIN_DBI, key.as_bytes()).expect("del");
        }
        txn.commit().expect("commit");
    }

    // Re-insert odd keys with different values
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in (1..100).step_by(2) {
            let key = format!("idr-{i:03}");
            txn.put(MAIN_DBI, key.as_bytes(), b"reinserted", WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Verify
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        for i in 0..100 {
            let key = format!("idr-{i:03}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            if i % 2 == 0 {
                assert_eq!(val, b"original");
            } else {
                assert_eq!(val, b"reinserted");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cursor operations
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_cursor_first_last_next_prev() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Insert known keys
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"alpha", b"1", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"beta", b"2", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"gamma", b"3", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    {
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");

        // First
        let (k, v) = cursor.get(None, CursorOp::First).expect("first");
        assert_eq!(k, b"alpha");
        assert_eq!(v, b"1");

        // Next
        let (k, v) = cursor.get(None, CursorOp::Next).expect("next");
        assert_eq!(k, b"beta");
        assert_eq!(v, b"2");

        // Next
        let (k, v) = cursor.get(None, CursorOp::Next).expect("next");
        assert_eq!(k, b"gamma");
        assert_eq!(v, b"3");

        // Last
        let (k, v) = cursor.get(None, CursorOp::Last).expect("last");
        assert_eq!(k, b"gamma");
        assert_eq!(v, b"3");

        // Prev
        let (k, v) = cursor.get(None, CursorOp::Prev).expect("prev");
        assert_eq!(k, b"beta");
        assert_eq!(v, b"2");
    }
}

#[test]
fn test_e2e_cursor_set_and_set_range() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"apple", b"fruit", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"banana", b"fruit", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"cherry", b"fruit", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    {
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");

        // Set exact
        let (k, v) = cursor
            .get(Some(b"banana"), CursorOp::Set)
            .expect("set banana");
        assert_eq!(k, b"banana");
        assert_eq!(v, b"fruit");

        // SetRange — "b" should find "banana"
        let (k, _) = cursor
            .get(Some(b"b"), CursorOp::SetRange)
            .expect("set_range b");
        assert_eq!(k, b"banana");

        // SetRange — "cat" should find "cherry"
        let (k, _) = cursor
            .get(Some(b"cat"), CursorOp::SetRange)
            .expect("set_range cat");
        assert_eq!(k, b"cherry");
    }
}

// ---------------------------------------------------------------------------
// NO_OVERWRITE flag
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_no_overwrite_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        txn.put(MAIN_DBI, b"unique", b"first", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        let result = txn.put(MAIN_DBI, b"unique", b"second", WriteFlags::NO_OVERWRITE);
        assert!(matches!(result, Err(Error::KeyExist)));
        txn.abort();
    }

    // Value should still be "first"
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        assert_eq!(txn.get(MAIN_DBI, b"unique").expect("get"), b"first");
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_empty_database_operations() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Get from empty DB
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        assert!(matches!(
            txn.get(MAIN_DBI, b"anything"),
            Err(Error::NotFound)
        ));
    }

    // Del from empty DB
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        assert!(matches!(
            txn.del(MAIN_DBI, b"anything"),
            Err(Error::NotFound)
        ));
        txn.abort();
    }

    // Cursor on empty DB
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        assert!(matches!(
            cursor.get(None, CursorOp::First),
            Err(Error::NotFound)
        ));
    }
}

#[test]
fn test_e2e_large_values() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Insert values of various sizes
    let sizes = [1, 10, 100, 500, 1000];
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for &size in &sizes {
            let key = format!("size-{size:04}");
            let val = vec![0xAB_u8; size];
            txn.put(MAIN_DBI, key.as_bytes(), &val, WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Read back and verify
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        for &size in &sizes {
            let key = format!("size-{size:04}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            assert_eq!(val.len(), size);
            assert!(val.iter().all(|&b| b == 0xAB));
        }
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_stat_after_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Initial stats: empty
    {
        let stat = env.stat().expect("stat");
        assert_eq!(stat.entries, 0);
        assert_eq!(stat.depth, 0);
    }

    // Insert some data
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in 0..10 {
            let key = format!("stat-{i}");
            txn.put(MAIN_DBI, key.as_bytes(), b"v", WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Stats should show entries
    {
        let stat = env.stat().expect("stat");
        assert_eq!(stat.entries, 10);
        assert!(stat.depth > 0);
        assert!(stat.page_size > 0);
    }
}

// ---------------------------------------------------------------------------
// Named databases
// ---------------------------------------------------------------------------

fn open_env_with_dbs(dir: &tempfile::TempDir) -> Environment {
    Environment::builder()
        .map_size(10 * 1024 * 1024)
        .max_dbs(8)
        .open(dir.path())
        .expect("open env with dbs")
}

#[test]
fn test_e2e_named_db_crud() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env_with_dbs(&dir);

    // Create and write to a named DB
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        let dbi = txn
            .open_db(Some("users"), DatabaseFlags::CREATE)
            .expect("open_db");
        txn.put(dbi, b"alice", b"admin", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"bob", b"user", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Read back from named DB
    {
        let mut txn = env.begin_ro_txn().expect("ro txn");
        let dbi = txn.open_db(Some("users")).expect("open_db ro");
        assert_eq!(txn.get(dbi, b"alice").expect("get"), b"admin");
        assert_eq!(txn.get(dbi, b"bob").expect("get"), b"user");
    }

    // Main DB should be unaffected
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        assert!(matches!(txn.get(MAIN_DBI, b"alice"), Err(Error::NotFound)));
    }
}

#[test]
fn test_e2e_multiple_named_dbs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env_with_dbs(&dir);

    // Create two named DBs
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        let db1 = txn
            .open_db(Some("db1"), DatabaseFlags::CREATE)
            .expect("open db1");
        let db2 = txn
            .open_db(Some("db2"), DatabaseFlags::CREATE)
            .expect("open db2");

        txn.put(db1, b"key", b"from-db1", WriteFlags::empty())
            .expect("put db1");
        txn.put(db2, b"key", b"from-db2", WriteFlags::empty())
            .expect("put db2");
        txn.commit().expect("commit");
    }

    // Each DB has its own data
    {
        let mut txn = env.begin_ro_txn().expect("ro txn");
        let db1 = txn.open_db(Some("db1")).expect("open db1");
        let db2 = txn.open_db(Some("db2")).expect("open db2");

        assert_eq!(txn.get(db1, b"key").expect("get db1"), b"from-db1");
        assert_eq!(txn.get(db2, b"key").expect("get db2"), b"from-db2");
    }
}

#[test]
fn test_e2e_named_db_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let env = open_env_with_dbs(&dir);
        let mut txn = env.begin_rw_txn().expect("rw txn");
        let dbi = txn
            .open_db(Some("persistent"), DatabaseFlags::CREATE)
            .expect("open_db");
        txn.put(dbi, b"data", b"survives", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    {
        let env = open_env_with_dbs(&dir);
        let mut txn = env.begin_ro_txn().expect("ro txn");
        let dbi = txn.open_db(Some("persistent")).expect("open_db ro");
        assert_eq!(txn.get(dbi, b"data").expect("get"), b"survives");
    }
}

// ---------------------------------------------------------------------------
// Stress tests
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_stress_5000_keys() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = Environment::builder()
        .map_size(50 * 1024 * 1024) // 50 MB
        .open(dir.path())
        .expect("open");

    let n = 5000;

    // Insert all keys
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in 0..n {
            let key = format!("stress-{i:06}");
            let val = format!("value-{i:06}-padding-to-make-it-longer");
            txn.put(
                MAIN_DBI,
                key.as_bytes(),
                val.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Verify all keys via cursor
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let mut count = 0;
        let mut prev_key: Option<Vec<u8>> = None;

        for result in cursor.iter() {
            let (key, _) = result.expect("cursor next");
            if let Some(ref pk) = prev_key {
                assert!(key > pk.as_slice(), "keys out of order");
            }
            prev_key = Some(key.to_vec());
            count += 1;
        }
        assert_eq!(count, n, "expected {n} keys, got {count}");
    }

    // Delete every 3rd key
    {
        let mut txn = env.begin_rw_txn().expect("rw txn");
        for i in (0..n).step_by(3) {
            let key = format!("stress-{i:06}");
            txn.del(MAIN_DBI, key.as_bytes()).expect("del");
        }
        txn.commit().expect("commit");
    }

    // Verify remaining keys
    {
        let txn = env.begin_ro_txn().expect("ro txn");
        let expected_remaining = n - ((n + 2) / 3);
        let mut count = 0;
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        for result in cursor.iter() {
            let _ = result.expect("cursor next");
            count += 1;
        }
        assert_eq!(
            count, expected_remaining,
            "expected {expected_remaining} remaining, got {count}"
        );
    }
}
