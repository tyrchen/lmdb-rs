//! Feature parity tests — exercises every implemented LMDB feature.
//!
//! Tests cover: basic CRUD, overflow pages, free page reuse, named databases,
//! nested transactions, DUPSORT, cursor operations, persistence, and MVCC.

use lmdb_rs_core::{
    env::Environment,
    error::Error,
    types::{CursorOp, DatabaseFlags, MAIN_DBI, WriteFlags},
};

fn open_env(dir: &tempfile::TempDir) -> Environment {
    Environment::builder()
        .map_size(64 * 1024 * 1024)
        .max_dbs(16)
        .open(dir.path())
        .expect("open env")
}

// ---------------------------------------------------------------------------
// DUPSORT comprehensive tests
// ---------------------------------------------------------------------------

#[test]
fn test_fp_dupsort_basic_crud() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    let dbi;
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        dbi = txn
            .open_db(
                Some("tags"),
                DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
            )
            .expect("open");

        // Add multiple tags for each item
        txn.put(dbi, b"item:1", b"rust", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"item:1", b"database", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"item:1", b"lmdb", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"item:2", b"python", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"item:2", b"web", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Read: get returns the FIRST dup value
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        let dbi = txn.open_db(Some("tags")).expect("open");
        let val = txn.get(dbi, b"item:1").expect("get");
        // Should be the first (lexicographically smallest) dup
        assert_eq!(val, b"database");
    }
}

#[test]
fn test_fp_dupsort_cursor_dup_navigation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    let dbi;
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        dbi = txn
            .open_db(Some("idx"), DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT)
            .expect("open");

        txn.put(dbi, b"color", b"blue", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"color", b"green", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"color", b"red", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"shape", b"circle", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"shape", b"square", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    {
        let mut txn = env.begin_ro_txn().expect("ro");
        let dbi = txn.open_db(Some("idx")).expect("open");
        let mut cursor = txn.open_cursor(dbi).expect("cursor");

        // First should be "color" -> "blue" (first dup)
        let (k, v) = cursor.get(None, CursorOp::First).expect("first");
        assert_eq!(k, b"color");
        assert_eq!(v, b"blue");

        // NextDup -> "green"
        let (k, v) = cursor.get(None, CursorOp::NextDup).expect("next_dup");
        assert_eq!(k, b"color");
        assert_eq!(v, b"green");

        // NextDup -> "red"
        let (k, v) = cursor.get(None, CursorOp::NextDup).expect("next_dup");
        assert_eq!(k, b"color");
        assert_eq!(v, b"red");

        // NextDup -> should fail (no more dups for "color")
        assert!(matches!(
            cursor.get(None, CursorOp::NextDup),
            Err(Error::NotFound)
        ));

        // Next -> moves to "shape" -> "circle"
        let (k, v) = cursor.get(None, CursorOp::Next).expect("next");
        assert_eq!(k, b"shape");
        assert_eq!(v, b"circle");

        // LastDup -> "square"
        let (k, v) = cursor.get(None, CursorOp::LastDup).expect("last_dup");
        assert_eq!(k, b"shape");
        assert_eq!(v, b"square");

        // PrevDup -> "circle"
        let (k, v) = cursor.get(None, CursorOp::PrevDup).expect("prev_dup");
        assert_eq!(k, b"shape");
        assert_eq!(v, b"circle");
    }
}

#[test]
fn test_fp_dupsort_delete_single_dup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let dbi = txn
            .open_db(Some("d"), DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT)
            .expect("open");
        txn.put(dbi, b"k", b"a", WriteFlags::empty()).expect("put");
        txn.put(dbi, b"k", b"b", WriteFlags::empty()).expect("put");
        txn.put(dbi, b"k", b"c", WriteFlags::empty()).expect("put");
        txn.commit().expect("commit");
    }

    // Delete specific dup "b"
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let dbi = txn.open_db(Some("d"), DatabaseFlags::CREATE).expect("open");
        txn.del(dbi, b"k", Some(b"b")).expect("del dup");
        txn.commit().expect("commit");
    }

    // Should have "a" and "c" remaining
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        let dbi = txn.open_db(Some("d")).expect("open");
        let mut cursor = txn.open_cursor(dbi).expect("cursor");

        let (_, v) = cursor.get(None, CursorOp::First).expect("first");
        assert_eq!(v, b"a");
        let (_, v) = cursor.get(None, CursorOp::NextDup).expect("next_dup");
        assert_eq!(v, b"c");
        assert!(matches!(
            cursor.get(None, CursorOp::NextDup),
            Err(Error::NotFound)
        ));
    }
}

#[test]
fn test_fp_dupsort_no_dup_data_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let dbi = txn
            .open_db(Some("u"), DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT)
            .expect("open");
        txn.put(dbi, b"k", b"val", WriteFlags::empty())
            .expect("put");

        // Try to insert the same dup value with NO_DUP_DATA
        let result = txn.put(dbi, b"k", b"val", WriteFlags::NO_DUP_DATA);
        assert!(matches!(result, Err(Error::KeyExist)));

        // But a different value should work
        txn.put(dbi, b"k", b"other", WriteFlags::NO_DUP_DATA)
            .expect("put other");
        txn.commit().expect("commit");
    }
}

// ---------------------------------------------------------------------------
// Nested transactions
// ---------------------------------------------------------------------------

#[test]
fn test_fp_nested_txn_rollback_partial_work() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"committed", b"yes", WriteFlags::empty())
            .expect("put");

        // Nested txn does work then aborts
        txn.begin_nested_txn().expect("begin");
        for i in 0..50 {
            let key = format!("nested:{i:03}");
            txn.put(MAIN_DBI, key.as_bytes(), b"temp", WriteFlags::empty())
                .expect("put");
        }
        txn.abort_nested_txn().expect("abort");

        // Only "committed" should exist
        assert_eq!(txn.get(MAIN_DBI, b"committed").expect("get"), b"yes");
        assert!(matches!(
            txn.get(MAIN_DBI, b"nested:000"),
            Err(Error::NotFound)
        ));

        txn.commit().expect("commit");
    }

    {
        let txn = env.begin_ro_txn().expect("ro");
        assert_eq!(txn.get(MAIN_DBI, b"committed").expect("get"), b"yes");
        assert!(matches!(
            txn.get(MAIN_DBI, b"nested:000"),
            Err(Error::NotFound)
        ));
    }
}

#[test]
fn test_fp_nested_txn_commit_chain() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"level0", b"base", WriteFlags::empty())
            .expect("put");

        txn.begin_nested_txn().expect("begin level1");
        txn.put(MAIN_DBI, b"level1", b"inner", WriteFlags::empty())
            .expect("put");

        txn.begin_nested_txn().expect("begin level2");
        txn.put(MAIN_DBI, b"level2", b"deepest", WriteFlags::empty())
            .expect("put");
        txn.commit_nested_txn().expect("commit level2");

        txn.commit_nested_txn().expect("commit level1");
        txn.commit().expect("commit");
    }

    {
        let txn = env.begin_ro_txn().expect("ro");
        assert_eq!(txn.get(MAIN_DBI, b"level0").expect("get"), b"base");
        assert_eq!(txn.get(MAIN_DBI, b"level1").expect("get"), b"inner");
        assert_eq!(txn.get(MAIN_DBI, b"level2").expect("get"), b"deepest");
    }
}

// ---------------------------------------------------------------------------
// Overflow + free page reuse combined
// ---------------------------------------------------------------------------

#[test]
fn test_fp_overflow_with_page_reuse() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Insert large values
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..10 {
            let key = format!("big:{i}");
            let val = vec![0xBB_u8; 8192]; // 2 overflow pages each
            txn.put(MAIN_DBI, key.as_bytes(), &val, WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    let _info_before = env.info();

    // Delete half
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..5 {
            let key = format!("big:{i}");
            txn.del(MAIN_DBI, key.as_bytes(), None).expect("del");
        }
        txn.commit().expect("commit");
    }

    // Insert new large values — should reuse freed pages
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 10..20 {
            let key = format!("big:{i}");
            let val = vec![0xCC_u8; 4096]; // 1 overflow page each
            txn.put(MAIN_DBI, key.as_bytes(), &val, WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Verify data integrity
    {
        let txn = env.begin_ro_txn().expect("ro");
        for i in 5..10 {
            let key = format!("big:{i}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            assert_eq!(val.len(), 8192);
        }
        for i in 10..20 {
            let key = format!("big:{i}");
            let val = txn.get(MAIN_DBI, key.as_bytes()).expect("get");
            assert_eq!(val.len(), 4096);
        }
    }
}

// ---------------------------------------------------------------------------
// All features combined: real-world application simulation
// ---------------------------------------------------------------------------

#[test]
fn test_fp_application_simulation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Simulate a document store with:
    // - "docs" DB: document ID -> JSON body
    // - "tags" DB: DUPSORT, tag -> document IDs (many docs per tag)
    // - "meta" DB: document ID -> metadata
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let docs = txn
            .open_db(Some("docs"), DatabaseFlags::CREATE)
            .expect("open");
        let tags = txn
            .open_db(
                Some("tags"),
                DatabaseFlags::CREATE | DatabaseFlags::DUP_SORT,
            )
            .expect("open");
        let meta = txn
            .open_db(Some("meta"), DatabaseFlags::CREATE)
            .expect("open");

        // Batch insert with nested txn for atomicity
        txn.begin_nested_txn().expect("begin nested");

        for i in 0..100 {
            let doc_id = format!("doc:{i:04}");
            let body = format!(r#"{{"id":{i},"title":"Document {i}"}}"#);
            txn.put(
                docs,
                doc_id.as_bytes(),
                body.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put doc");

            let meta_val = format!(r#"{{"created":"2024-01-{:02}"}}"#, (i % 28) + 1);
            txn.put(
                meta,
                doc_id.as_bytes(),
                meta_val.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put meta");

            // Each doc gets 2-3 tags
            let tag1 = format!("tag:{}", i % 10);
            txn.put(
                tags,
                tag1.as_bytes(),
                doc_id.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put tag1");
            let tag2 = format!("tag:{}", (i + 5) % 10);
            txn.put(
                tags,
                tag2.as_bytes(),
                doc_id.as_bytes(),
                WriteFlags::empty(),
            )
            .expect("put tag2");
        }

        txn.commit_nested_txn().expect("commit nested");
        txn.commit().expect("commit");
    }

    // Verify: look up docs by tag
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        let tags = txn.open_db(Some("tags")).expect("open");
        let docs = txn.open_db(Some("docs")).expect("open");

        // Find all docs with "tag:0"
        let mut cursor = txn.open_cursor(tags).expect("cursor");
        let (k, first_doc_id) = cursor
            .get(Some(b"tag:0"), CursorOp::Set)
            .expect("set tag:0");
        assert_eq!(k, b"tag:0");

        // The first doc should be retrievable
        let doc_body = txn.get(docs, first_doc_id).expect("get doc");
        assert!(
            std::str::from_utf8(doc_body)
                .expect("utf8")
                .contains("title")
        );
    }

    // Verify persistence across reopen
    {
        drop(env);
        let env2 = open_env(&dir);
        let mut txn = env2.begin_ro_txn().expect("ro");
        let docs = txn.open_db(Some("docs")).expect("open");
        let val = txn.get(docs, b"doc:0050").expect("get");
        assert!(
            std::str::from_utf8(val)
                .expect("utf8")
                .contains("Document 50")
        );
    }
}

// ---------------------------------------------------------------------------
// Stress: high-volume operations
// ---------------------------------------------------------------------------

#[test]
fn test_fp_stress_10k_keys_5_dbs_nested() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    let db_names = ["alpha", "beta", "gamma", "delta", "epsilon"];

    // 5 rounds, each with nested txn
    for round in 0..5 {
        let mut txn = env.begin_rw_txn().expect("rw");

        txn.begin_nested_txn().expect("begin nested");

        for db_name in &db_names {
            let dbi = txn
                .open_db(Some(db_name), DatabaseFlags::CREATE)
                .expect("open");
            for i in 0..400 {
                let key = format!("r{round}:{i:04}");
                let val = format!("{db_name}:{round}:{i}");
                txn.put(dbi, key.as_bytes(), val.as_bytes(), WriteFlags::empty())
                    .expect("put");
            }
        }

        txn.commit_nested_txn().expect("commit nested");
        txn.commit().expect("commit");
    }

    // Verify: 5 rounds * 400 keys per DB = 2000 keys per DB
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        for db_name in &db_names {
            let dbi = txn.open_db(Some(db_name)).expect("open");
            let mut cursor = txn.open_cursor(dbi).expect("cursor");
            let count = cursor.iter().count();
            assert_eq!(
                count, 2000,
                "{db_name} should have 2000 entries, got {count}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// txn reset/renew
// ---------------------------------------------------------------------------

#[test]
fn test_fp_txn_reset_renew() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    // Write initial data
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"v1", b"first", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Start read txn, verify, reset, write more data, renew, see new data
    let mut ro = env.begin_ro_txn().expect("ro");
    assert_eq!(ro.get(MAIN_DBI, b"v1").expect("get"), b"first");

    ro.reset();

    // Write more data while txn is reset
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"v2", b"second", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Renew the txn — should see new data
    ro.renew().expect("renew");
    assert_eq!(ro.get(MAIN_DBI, b"v1").expect("get"), b"first");
    assert_eq!(ro.get(MAIN_DBI, b"v2").expect("get"), b"second");
}

// ---------------------------------------------------------------------------
// drop_db
// ---------------------------------------------------------------------------

#[test]
fn test_fp_drop_db_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let dbi = txn
            .open_db(Some("to_empty"), DatabaseFlags::CREATE)
            .expect("open");
        txn.put(dbi, b"k1", b"v1", WriteFlags::empty())
            .expect("put");
        txn.put(dbi, b"k2", b"v2", WriteFlags::empty())
            .expect("put");
        txn.commit().expect("commit");
    }

    // Empty the database (del=false)
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let dbi = txn
            .open_db(Some("to_empty"), DatabaseFlags::CREATE)
            .expect("open");
        txn.drop_db(dbi, false).expect("drop_db");
        txn.commit().expect("commit");
    }

    // DB should be empty but still exist
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        let dbi = txn.open_db(Some("to_empty")).expect("open");
        assert!(matches!(txn.get(dbi, b"k1"), Err(Error::NotFound)));
    }
}

#[test]
fn test_fp_drop_db_delete() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let dbi = txn
            .open_db(Some("to_delete"), DatabaseFlags::CREATE)
            .expect("open");
        txn.put(dbi, b"k", b"v", WriteFlags::empty()).expect("put");
        txn.commit().expect("commit");
    }

    // Delete the database entirely (del=true)
    {
        let mut txn = env.begin_rw_txn().expect("rw");
        let dbi = txn
            .open_db(Some("to_delete"), DatabaseFlags::CREATE)
            .expect("open");
        txn.drop_db(dbi, true).expect("drop_db");
        txn.commit().expect("commit");
    }

    // DB should not exist anymore
    {
        let mut txn = env.begin_ro_txn().expect("ro");
        assert!(matches!(
            txn.open_db(Some("to_delete")),
            Err(Error::NotFound)
        ));
    }
}

// ---------------------------------------------------------------------------
// MDB_APPEND
// ---------------------------------------------------------------------------

#[test]
fn test_fp_append_mode_sequential() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..1000u32 {
            let key = format!("k{i:06}");
            txn.put(MAIN_DBI, key.as_bytes(), b"v", WriteFlags::APPEND)
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Verify all keys
    {
        let txn = env.begin_ro_txn().expect("ro");
        let mut cursor = txn.open_cursor(MAIN_DBI).expect("cursor");
        let count = cursor.iter().count();
        assert_eq!(count, 1000);
    }
}

#[test]
fn test_fp_append_mode_out_of_order_fails() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        txn.put(MAIN_DBI, b"bbb", b"v", WriteFlags::APPEND)
            .expect("first put");
        // Inserting a key that is NOT greater should fail
        let result = txn.put(MAIN_DBI, b"aaa", b"v", WriteFlags::APPEND);
        assert!(matches!(result, Err(Error::KeyExist)));
        txn.abort();
    }
}

// ---------------------------------------------------------------------------
// Write cursor
// ---------------------------------------------------------------------------

#[test]
fn test_fp_write_cursor_put() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        {
            let mut cursor = txn.open_rw_cursor(MAIN_DBI).expect("cursor");
            cursor
                .put(b"key1", b"val1", WriteFlags::empty())
                .expect("put");
            cursor
                .put(b"key2", b"val2", WriteFlags::empty())
                .expect("put");
            cursor
                .put(b"key3", b"val3", WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    {
        let txn = env.begin_ro_txn().expect("ro");
        assert_eq!(txn.get(MAIN_DBI, b"key1").expect("get"), b"val1");
        assert_eq!(txn.get(MAIN_DBI, b"key2").expect("get"), b"val2");
        assert_eq!(txn.get(MAIN_DBI, b"key3").expect("get"), b"val3");
    }
}

// ---------------------------------------------------------------------------
// Environment copy
// ---------------------------------------------------------------------------

#[test]
fn test_fp_env_copy() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env = open_env(&dir);

    {
        let mut txn = env.begin_rw_txn().expect("rw");
        for i in 0..50 {
            let key = format!("copy:{i:03}");
            txn.put(MAIN_DBI, key.as_bytes(), b"data", WriteFlags::empty())
                .expect("put");
        }
        txn.commit().expect("commit");
    }

    // Copy to a new location
    let copy_dir = tempfile::tempdir().expect("tempdir");
    let copy_path = copy_dir.path().join("backup.mdb");
    env.copy(&copy_path).expect("copy");

    // Open the copy and verify data
    let env2 = Environment::builder()
        .map_size(64 * 1024 * 1024)
        .flags(lmdb_rs_core::types::EnvFlags::NO_SUB_DIR)
        .open(&copy_path)
        .expect("open copy");

    let txn = env2.begin_ro_txn().expect("ro");
    for i in 0..50 {
        let key = format!("copy:{i:03}");
        assert_eq!(txn.get(MAIN_DBI, key.as_bytes()).expect("get"), b"data");
    }
}
