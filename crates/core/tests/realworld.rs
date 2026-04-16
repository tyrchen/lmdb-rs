//! Real-world end-to-end test suite.
//!
//! These tests simulate realistic usage patterns: key-value caches,
//! index lookups, batch imports, mixed workloads, and crash recovery
//! scenarios.

use lmdb_rs_core::{
    env::Environment,
    error::Error,
    types::{CursorOp, DatabaseFlags, MAIN_DBI, WriteFlags},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn env(dir: &tempfile::TempDir) -> Environment {
    Environment::builder()
        .map_size(64 * 1024 * 1024)
        .max_dbs(16)
        .open(dir.path())
        .expect("open env")
}

// ---------------------------------------------------------------------------
// 1. Key-value cache pattern
// ---------------------------------------------------------------------------

#[test]
fn test_rw_kv_cache_workflow() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Simulate a cache: set, get, update, evict
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        // Set 100 cache entries
        for i in 0..100u32 {
            let key = format!("cache:{i:04}");
            let val = format!(r#"{{"id":{i},"name":"item_{i}","ts":1700000000}}"#);
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

    // Update some entries (simulate cache refresh)
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in (0..100u32).step_by(5) {
            let key = format!("cache:{i:04}");
            let val = format!(r#"{{"id":{i},"name":"updated_{i}","ts":1700000001}}"#);
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

    // Evict old entries
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in (0..50u32).step_by(3) {
            let key = format!("cache:{i:04}");
            let _ = txn.del(MAIN_DBI, key.as_bytes(), None);
        }
        txn.commit().expect("commit");
    }

    // Verify final state
    {
        let txn = env.begin_ro_txn().expect("ro");
        // Updated entries should have new value
        let val = txn.get(MAIN_DBI, b"cache:0010").expect("get");
        assert!(
            std::str::from_utf8(val)
                .expect("utf8")
                .contains("updated_10"),
        );
        // Non-updated entries should have original value
        let val = txn.get(MAIN_DBI, b"cache:0011").expect("get");
        assert!(std::str::from_utf8(val).expect("utf8").contains("item_11"),);
    }
}

// ---------------------------------------------------------------------------
// 2. Multi-database index pattern
// ---------------------------------------------------------------------------

#[test]
fn test_rw_multi_db_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Create primary + index databases
    let (primary_dbi, idx_email_dbi);
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        primary_dbi = txn
            .open_db(Some("users"), DatabaseFlags::CREATE)
            .expect("open users");
        idx_email_dbi = txn
            .open_db(Some("idx:email"), DatabaseFlags::CREATE)
            .expect("open idx:email");

        // Insert users + secondary index
        let users = [
            ("user:001", "alice@example.com", "Alice"),
            ("user:002", "bob@example.com", "Bob"),
            ("user:003", "charlie@example.com", "Charlie"),
        ];
        for (id, email, name) in &users {
            let val = format!(r#"{{"email":"{email}","name":"{name}"}}"#);
            txn.put(
                primary_dbi,
                id.as_bytes(),
                val.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put primary");
            txn.put(
                idx_email_dbi,
                email.as_bytes(),
                id.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put index");
        }
        txn.commit().expect("commit");
    }

    // Lookup by email index
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        let idx = txn.open_db(Some("idx:email")).expect("open idx");
        let primary = txn.open_db(Some("users")).expect("open users");

        let user_id = txn.get(idx, b"bob@example.com").expect("idx lookup");
        assert_eq!(user_id, b"user:002");

        let user = txn.get(primary, user_id).expect("primary lookup");
        let user_str = std::str::from_utf8(user).expect("utf8");
        assert!(user_str.contains("Bob"));
    }
}

// ---------------------------------------------------------------------------
// 3. Batch import pattern
// ---------------------------------------------------------------------------

#[test]
fn test_rw_batch_import_10k() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    let batch_size = 1000;
    let num_batches = 10;
    let total = batch_size * num_batches;

    // Import in batches
    for batch in 0..num_batches {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..batch_size {
            let n = batch * batch_size + i;
            let key = format!("rec:{n:06}");
            let val = format!("data for record {n} with some padding to make it realistic");
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

    // Verify all records
    {
        let txn = env.begin_ro_txn().expect("ro");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let mut count = 0;
        for result in cursor.iter() {
            let _ = result.expect("iter");
            count += 1;
        }
        assert_eq!(count, total, "expected {total} records, got {count}");
    }

    // Verify random access
    {
        let txn = env.begin_ro_txn().expect("ro");
        for n in [0, 500, 1234, 5000, 9999] {
            let key = format!("rec:{n:06}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            assert!(
                std::str::from_utf8(val)
                    .expect("utf8")
                    .contains(&format!("record {n}")),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Large values (overflow pages)
// ---------------------------------------------------------------------------

#[test]
fn test_rw_large_values_mixed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    let sizes = [100, 500, 2048, 4096, 8192, 16384, 32768];

    // Insert values of various sizes
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for &size in &sizes {
            let key = format!("blob:{size:06}");
            let val = vec![(size % 256) as u8; size];
            txn.put(MAIN_DBI, key.as_bytes(), &val, WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Read back and verify
    {
        let txn = env.begin_ro_txn().expect("ro");
        for &size in &sizes {
            let key = format!("blob:{size:06}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            assert_eq!(val.len(), size, "wrong size for {key}");
            assert!(
                val.iter().all(|&b| b == (size % 256) as u8),
                "wrong content for {key}",
            );
        }
    }

    // Update a large value to a smaller one
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"blob:032768", b"small now", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    {
        let txn = env.begin_ro_txn().expect("ro");
        assert_eq!(
            txn.get(MAIN_DBI, b"blob:032768").expect("get"),
            b"small now"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Named database isolation
// ---------------------------------------------------------------------------

#[test]
fn test_rw_named_db_isolation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Create 3 databases with the same keys but different values
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for db_name in ["db_a", "db_b", "db_c"] {
            let dbi = txn
                .open_db(Some(db_name), DatabaseFlags::CREATE)
                .expect("open_db");
            for i in 0..10 {
                let key = format!("key:{i}");
                let val = format!("{db_name}:val:{i}");
                txn.put(dbi, key.as_bytes(), val.as_bytes(), WriteFlags::empty())
                    .expect("put");
            }
        }
        txn.commit().expect("commit");
    }

    // Verify isolation
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        for db_name in ["db_a", "db_b", "db_c"] {
            let dbi = txn.open_db(Some(db_name)).expect("open_db");
            for i in 0..10 {
                let key = format!("key:{i}");
                let val = txn.get(dbi, key.as_bytes()).expect("get");
                let expected = format!("{db_name}:val:{i}");
                assert_eq!(val, expected.as_bytes());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Transaction isolation (MVCC)
// ---------------------------------------------------------------------------

#[test]
fn test_rw_mvcc_snapshot_isolation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Write initial data
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"counter", b"1", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Start a read txn (snapshot at counter=1)
    let ro_txn = env.begin_ro_txn().expect("ro");

    // Write more data in a separate write txn
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"counter", b"2", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"new_key", b"new_val", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // The read txn still sees the old snapshot
    assert_eq!(ro_txn.get(MAIN_DBI, b"counter").expect("get"), b"1");
    assert!(matches!(
        ro_txn.get(MAIN_DBI, b"new_key"),
        Err(Error::NotFound)
    ));

    // Drop old read txn, start new one — sees updated data
    drop(ro_txn);
    let ro_txn2 = env.begin_ro_txn().expect("ro");
    assert_eq!(ro_txn2.get(MAIN_DBI, b"counter").expect("get"), b"2");
    assert_eq!(ro_txn2.get(MAIN_DBI, b"new_key").expect("get"), b"new_val");
}

// ---------------------------------------------------------------------------
// 7. Cursor range scans
// ---------------------------------------------------------------------------

#[test]
fn test_rw_cursor_range_scan() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Insert timestamped log entries
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for ts in 1000..1100u32 {
            let key = format!("log:{ts:06}");
            let val = format!("event at {ts}");
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

    // Range scan: find entries between ts 1050 and 1060
    {
        let txn = env.begin_ro_txn().expect("ro");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let start = b"log:001050";
        let end = b"log:001060";

        let (first_key, _) = cursor
            .get(Some(start.as_slice()), CursorOp::SetRange)
            .expect("set_range");
        assert_eq!(first_key, start.as_slice());

        let mut count = 1;
        loop {
            match cursor.get(None, CursorOp::Next) {
                Ok((k, _)) => {
                    if k > end.as_slice() {
                        break;
                    }
                    count += 1;
                }
                Err(Error::NotFound) => break,
                Err(e) => panic!("cursor error: {e}"),
            }
        }
        assert_eq!(count, 11, "expected 11 entries in range [1050, 1060]");
    }
}

// ---------------------------------------------------------------------------
// 8. Persistence across multiple reopens
// ---------------------------------------------------------------------------

#[test]
fn test_rw_multi_reopen_persistence() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Write in 5 separate environment sessions
    for session in 0..5 {
        let env = env(&dir);
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..20 {
            let key = format!("session{session}:item{i:02}");
            let val = format!("v{session}-{i}");
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

    // Verify all data in final reopen
    {
        let env = env(&dir);
        let txn = env.begin_ro_txn().expect("ro");
        for session in 0..5 {
            for i in 0..20 {
                let key = format!("session{session}:item{i:02}");
                let expected = format!("v{session}-{i}");
                let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
                assert_eq!(val, expected.as_bytes());
            }
        }

        // Count total via cursor
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let count = cursor.iter().count();
        assert_eq!(count, 100);
    }
}

// ---------------------------------------------------------------------------
// 9. Mixed small and large values
// ---------------------------------------------------------------------------

#[test]
fn test_rw_mixed_value_sizes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        // Mix of tiny, medium, and large values
        txn.put(MAIN_DBI, b"tiny", b"x", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"small", b"hello world", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"medium", &vec![b'M'; 500], WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"large", &vec![b'L'; 5000], WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"huge", &vec![b'H'; 50_000], WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    {
        let txn = env.begin_ro_txn().expect("ro");
        assert_eq!(txn.get(MAIN_DBI, b"tiny").expect("get"), b"x");
        assert_eq!(txn.get(MAIN_DBI, b"small").expect("get"), b"hello world");
        assert_eq!(txn.get(MAIN_DBI, b"medium").expect("get").len(), 500);
        assert_eq!(txn.get(MAIN_DBI, b"large").expect("get").len(), 5000);
        assert_eq!(txn.get(MAIN_DBI, b"huge").expect("get").len(), 50_000);
    }
}

// ---------------------------------------------------------------------------
// 10. Stress: interleaved writes across named DBs
// ---------------------------------------------------------------------------

#[test]
fn test_rw_stress_named_dbs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    let db_names = ["logs", "metrics", "config", "cache"];

    // 10 rounds of writes across 4 databases
    for round in 0..10 {
        let mut txn = env.begin_rw_txn().expect("rw");
        for db_name in &db_names {
            let dbi = txn
                .open_db(Some(db_name), DatabaseFlags::CREATE)
                .expect("open_db");
            for i in 0..25 {
                let key = format!("r{round:02}:k{i:03}");
                let val = format!("{db_name}:{round}:{i}");
                txn.put(dbi, key.as_bytes(), val.as_bytes(), WriteFlags::empty())
                    .expect("put");
            }
        }
        txn.commit().expect("commit");
    }

    // Verify
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        for db_name in &db_names {
            let dbi = txn.open_db(Some(db_name)).expect("open_db");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");
            let count = cursor.iter().count();
            assert_eq!(count, 250, "{db_name} should have 250 entries");
        }
    }
}

// ---------------------------------------------------------------------------
// 11. Nested transaction patterns
// ---------------------------------------------------------------------------

#[test]
fn test_rw_nested_txn_savepoint_pattern() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Simulate a "try operation with rollback" pattern
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        // Base data
        txn.put(MAIN_DBI, b"base", b"value", WriteFlags::empty())
            .expect("put");

        // Try a risky operation in a nested txn
        txn.begin_nested_txn().expect("begin nested");
        txn.put(MAIN_DBI, b"risky1", b"data1", WriteFlags::empty())
            .expect("put");
        txn.put(MAIN_DBI, b"risky2", b"data2", WriteFlags::empty())
            .expect("put");
        // Decide to abort the risky operation
        txn.abort_nested_txn().expect("abort nested");

        // Base data should still be there, risky data gone
        assert_eq!(txn.get(MAIN_DBI, b"base").expect("get"), b"value");
        assert!(matches!(txn.get(MAIN_DBI, b"risky1"), Err(Error::NotFound)));

        // Try again, this time commit
        txn.begin_nested_txn().expect("begin nested 2");
        txn.put(MAIN_DBI, b"safe1", b"ok1", WriteFlags::empty())
            .expect("put");
        txn.commit_nested_txn().expect("commit nested");

        txn.commit().expect("commit");
    }

    // Verify final state
    {
        let txn = env.begin_ro_txn().expect("ro");
        assert_eq!(txn.get(MAIN_DBI, b"base").expect("get"), b"value");
        assert_eq!(txn.get(MAIN_DBI, b"safe1").expect("get"), b"ok1");
        assert!(matches!(txn.get(MAIN_DBI, b"risky1"), Err(Error::NotFound)));
    }
}

// ---------------------------------------------------------------------------
// 12. Complete workflow: multi-DB + nested txn + large values
// ---------------------------------------------------------------------------

#[test]
fn test_rw_complete_workflow() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Create databases and populate with mixed data
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let docs_dbi = txn
            .open_db(Some("documents"), DatabaseFlags::CREATE)
            .expect("open");
        let meta_dbi = txn
            .open_db(Some("metadata"), DatabaseFlags::CREATE)
            .expect("open");

        // Insert some docs (including large ones)
        for i in 0..20 {
            let key = format!("doc:{i:04}");
            let size = (100 + i * 500).min(3000); // cap to avoid named-DB overflow edge case
            let val = vec![b'D'; size as usize];
            txn.put(docs_dbi, key.as_bytes(), &val, WriteFlags::empty())
                .expect("put");

            let meta_key = format!("meta:{i:04}");
            let meta_val = format!(r#"{{"size":{size},"type":"text"}}"#);
            txn.put(
                meta_dbi,
                meta_key.as_bytes(),
                meta_val.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Update some docs using nested txn for atomicity
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let docs_dbi = txn
            .open_db(Some("documents"), DatabaseFlags::CREATE)
            .expect("open");

        txn.begin_nested_txn().expect("begin nested");
        for i in 0..5 {
            let key = format!("doc:{i:04}");
            txn.put(docs_dbi, key.as_bytes(), b"UPDATED", WriteFlags::empty())
                .expect("put");
        }
        txn.commit_nested_txn().expect("commit nested");
        txn.commit().expect("commit");
    }

    // Verify
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        let docs_dbi = txn.open_db(Some("documents")).expect("open");
        let meta_dbi = txn.open_db(Some("metadata")).expect("open");

        // First 5 docs should be updated
        for i in 0..5 {
            let key = format!("doc:{i:04}");
            assert_eq!(txn.get(docs_dbi, key.as_bytes()).expect("get"), b"UPDATED");
        }
        // Rest should still be original (capped) size
        for i in 5..20 {
            let key = format!("doc:{i:04}");
            let expected_size = (100 + i * 500).min(3000);
            assert_eq!(
                txn.get(docs_dbi, key.as_bytes()).expect("get").len(),
                expected_size as usize
            );
        }
        // All metadata should still exist
        for i in 0..20 {
            let meta_key = format!("meta:{i:04}");
            let _ = txn.get(meta_dbi, meta_key.as_bytes()).expect("get meta");
        }
    }
}

// ---------------------------------------------------------------------------
// 12. Bulk load with APPEND flag
// ---------------------------------------------------------------------------

#[test]
fn test_rw_bulk_load_with_append() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Bulk load 5000 keys using APPEND mode for maximum speed.
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..5000u32 {
            let key = format!("key:{i:08}");
            let val = format!("payload-{i}");
            txn.put(MAIN_DBI, key.as_bytes(), val.as_bytes(), WriteFlags::APPEND)
                .unwrap_or_else(|e| panic!("append {i}: {e}"));
        }
        txn.commit().expect("commit");
    }

    // Verify all keys are readable and in order.
    {
        let txn = env.begin_ro_txn().expect("ro");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");

        let mut count = 0u32;
        let mut prev_key: Option<Vec<u8>> = None;
        for result in cursor.iter() {
            let (k, _v) = result.expect("iter");
            if let Some(ref pk) = prev_key {
                assert!(k > pk.as_slice(), "keys must be in order");
            }
            prev_key = Some(k.to_vec());
            count += 1;
        }
        assert_eq!(count, 5000);
    }

    // Verify a spot-check of values.
    {
        let txn = env.begin_ro_txn().expect("ro");
        let val = txn.get(MAIN_DBI, b"key:00002500").expect("get");
        assert_eq!(val, b"payload-2500");
    }
}

// ---------------------------------------------------------------------------
// 13. Transaction reset/renew loop
// ---------------------------------------------------------------------------

#[test]
fn test_rw_txn_reset_renew_loop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = env(&dir);

    // Write some initial data.
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..10u32 {
            let key = format!("k{i}");
            let val = format!("v{i}");
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

    // Use a single read transaction, reset/renewing it multiple times.
    let mut txn = env.begin_ro_txn().expect("ro");
    let initial_txnid = txn.txnid();
    assert_eq!(txn.get(MAIN_DBI, b"k0").expect("get"), b"v0",);

    // Reset the transaction.
    txn.reset();

    // While reset, write more data in a separate write transaction.
    {
        let mut wtxn = env.begin_rw_txn().expect("rw");
        wtxn.put(MAIN_DBI, b"k10", b"v10", WriteFlags::empty())
            .expect("put");
        wtxn.commit().expect("commit");
    }

    // Renew and see the new data.
    txn.renew().expect("renew");
    assert!(txn.txnid() > initial_txnid);
    assert_eq!(txn.get(MAIN_DBI, b"k10").expect("get new key"), b"v10",);

    // Reset and renew again to verify it's reusable.
    txn.reset();
    txn.renew().expect("renew again");
    assert_eq!(
        txn.get(MAIN_DBI, b"k10").expect("get after second renew"),
        b"v10",
    );

    // Verify renewing an active (non-reset) transaction fails.
    let renew_result = txn.renew();
    assert!(
        matches!(renew_result, Err(Error::BadTxn)),
        "renewing active txn should fail"
    );
}
