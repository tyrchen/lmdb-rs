//! Thin FFI wrapper around `lmdb-master-sys`, exposed as a safe, ergonomic
//! API shaped to mirror `lmdb-rs-core` for apples-to-apples benchmarking.
//!
//! This crate intentionally contains the **only** `unsafe` blocks in the
//! benchmark stack. The [`CEnv`] / [`CRoTxn`] / [`CRwTxn`] / [`CCursor`]
//! types hand out `&[u8]` slices that live for the transaction's lifetime,
//! matching the zero-copy contract of the Rust implementation.
//!
//! Everything is intentionally minimal: we only expose what the benches
//! actually call. No env flag mapping, no comparator customization — the
//! C defaults are what `lmdb-rs-core` also uses.

#![allow(clippy::missing_safety_doc)]

use std::{
    ffi::{CString, c_void},
    path::Path,
    ptr,
};

use lmdb_master_sys as sys;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CError {
    pub code: i32,
    pub ctx: &'static str,
}

impl std::fmt::Display for CError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C LMDB error in {}: code={}", self.ctx, self.code)
    }
}

impl std::error::Error for CError {}

pub type CResult<T> = Result<T, CError>;

fn chk(rc: i32, ctx: &'static str) -> CResult<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(CError { code: rc, ctx })
    }
}

// ---------------------------------------------------------------------------
// Bit flags that we re-export so benches can stay pure-Rust
// ---------------------------------------------------------------------------

pub mod flags {
    /// `MDB_NOSYNC` — skip fsync on commit.
    pub const ENV_NOSYNC: u32 = 0x10000;
    /// `MDB_WRITEMAP` — use a writeable mmap instead of a read-only one.
    pub const ENV_WRITEMAP: u32 = 0x80000;

    pub const DB_CREATE: u32 = 0x40000;
    pub const DB_DUPSORT: u32 = 0x04;
    pub const DB_DUPFIXED: u32 = 0x10;

    pub const PUT_APPEND: u32 = 0x20000;
    pub const PUT_RESERVE: u32 = 0x10000;
    pub const PUT_NOOVERWRITE: u32 = 0x10;
}

// ---------------------------------------------------------------------------
// CEnv
// ---------------------------------------------------------------------------

/// Owning handle for `MDB_env`.
pub struct CEnv {
    env: *mut sys::MDB_env,
}

// `MDB_env` is designed for shared access across threads; sync/send is safe
// as long as callers respect the single-writer rule.
unsafe impl Send for CEnv {}
unsafe impl Sync for CEnv {}

impl CEnv {
    /// Open an environment. The destination `path` must be an existing directory.
    pub fn open(path: &Path, map_size: usize, max_dbs: u32, flags: u32) -> CResult<Self> {
        let mut env: *mut sys::MDB_env = ptr::null_mut();
        unsafe {
            chk(sys::mdb_env_create(&mut env), "env_create")?;
            if let Err(e) = (|| -> CResult<()> {
                chk(sys::mdb_env_set_mapsize(env, map_size), "set_mapsize")?;
                if max_dbs > 0 {
                    chk(sys::mdb_env_set_maxdbs(env, max_dbs), "set_maxdbs")?;
                }
                let cpath =
                    CString::new(path.to_str().expect("utf-8 path")).expect("no nul in path");
                // mode 0o644 — same default lmdb uses.
                chk(
                    sys::mdb_env_open(env, cpath.as_ptr(), flags, 0o644),
                    "env_open",
                )?;
                Ok(())
            })() {
                sys::mdb_env_close(env);
                return Err(e);
            }
        }
        Ok(Self { env })
    }

    pub fn sync(&self, force: bool) -> CResult<()> {
        unsafe { chk(sys::mdb_env_sync(self.env, i32::from(force)), "env_sync") }
    }

    pub fn copy(&self, dst: &Path) -> CResult<()> {
        let c = CString::new(dst.to_str().expect("utf-8 path")).expect("no nul");
        unsafe { chk(sys::mdb_env_copy(self.env, c.as_ptr()), "env_copy") }
    }

    pub fn copy_compact(&self, dst: &Path) -> CResult<()> {
        let c = CString::new(dst.to_str().expect("utf-8 path")).expect("no nul");
        const MDB_CP_COMPACT: u32 = 1;
        unsafe {
            chk(
                sys::mdb_env_copy2(self.env, c.as_ptr(), MDB_CP_COMPACT),
                "env_copy2",
            )
        }
    }

    pub fn begin_ro(&self) -> CResult<CRoTxn<'_>> {
        const MDB_RDONLY: u32 = 0x20000;
        let mut txn: *mut sys::MDB_txn = ptr::null_mut();
        unsafe {
            chk(
                sys::mdb_txn_begin(self.env, ptr::null_mut(), MDB_RDONLY, &mut txn),
                "txn_begin_ro",
            )?;
        }
        Ok(CRoTxn { txn, _env: self })
    }

    pub fn begin_rw(&self) -> CResult<CRwTxn<'_>> {
        let mut txn: *mut sys::MDB_txn = ptr::null_mut();
        unsafe {
            chk(
                sys::mdb_txn_begin(self.env, ptr::null_mut(), 0, &mut txn),
                "txn_begin_rw",
            )?;
        }
        Ok(CRwTxn {
            txn,
            _env: self,
            finished: false,
        })
    }

    pub fn raw(&self) -> *mut sys::MDB_env {
        self.env
    }
}

impl Drop for CEnv {
    fn drop(&mut self) {
        if !self.env.is_null() {
            unsafe { sys::mdb_env_close(self.env) }
        }
    }
}

// ---------------------------------------------------------------------------
// CRoTxn
// ---------------------------------------------------------------------------

pub struct CRoTxn<'env> {
    txn: *mut sys::MDB_txn,
    _env: &'env CEnv,
}

impl<'env> CRoTxn<'env> {
    /// Open or look up the MAIN (unnamed) DB.
    pub fn main_dbi(&self) -> CResult<u32> {
        let mut dbi: sys::MDB_dbi = 0;
        unsafe {
            chk(
                sys::mdb_dbi_open(self.txn, ptr::null(), 0, &mut dbi),
                "dbi_open_main",
            )?;
        }
        Ok(dbi)
    }

    /// Open a named DB (must already exist — no CREATE flag).
    pub fn open_named(&self, name: &str) -> CResult<u32> {
        let c = CString::new(name).expect("no nul in name");
        let mut dbi: sys::MDB_dbi = 0;
        unsafe {
            chk(
                sys::mdb_dbi_open(self.txn, c.as_ptr(), 0, &mut dbi),
                "dbi_open_named",
            )?;
        }
        Ok(dbi)
    }

    pub fn get<'t>(&'t self, dbi: u32, key: &[u8]) -> CResult<&'t [u8]> {
        let mut k = mk_val(key);
        let mut v = empty_val();
        unsafe {
            chk(sys::mdb_get(self.txn, dbi, &mut k, &mut v), "get")?;
            Ok(val_as_slice(&v))
        }
    }

    pub fn open_cursor(&self, dbi: u32) -> CResult<CCursor<'_>> {
        let mut cur: *mut sys::MDB_cursor = ptr::null_mut();
        unsafe {
            chk(
                sys::mdb_cursor_open(self.txn, dbi, &mut cur),
                "cursor_open_ro",
            )?;
        }
        Ok(CCursor {
            cur,
            _marker: std::marker::PhantomData,
        })
    }
}

impl Drop for CRoTxn<'_> {
    fn drop(&mut self) {
        // mdb_txn_abort also works for RO txns and is the usual drop path.
        unsafe { sys::mdb_txn_abort(self.txn) }
    }
}

// ---------------------------------------------------------------------------
// CRwTxn
// ---------------------------------------------------------------------------

pub struct CRwTxn<'env> {
    txn: *mut sys::MDB_txn,
    _env: &'env CEnv,
    finished: bool,
}

impl<'env> CRwTxn<'env> {
    pub fn main_dbi(&self) -> CResult<u32> {
        let mut dbi: sys::MDB_dbi = 0;
        unsafe {
            chk(
                sys::mdb_dbi_open(self.txn, ptr::null(), 0, &mut dbi),
                "dbi_open_main",
            )?;
        }
        Ok(dbi)
    }

    /// Open (or create) a named DB with the given flags.
    pub fn open_named(&self, name: &str, db_flags: u32) -> CResult<u32> {
        let c = CString::new(name).expect("no nul in name");
        let mut dbi: sys::MDB_dbi = 0;
        unsafe {
            chk(
                sys::mdb_dbi_open(self.txn, c.as_ptr(), db_flags, &mut dbi),
                "dbi_open_named",
            )?;
        }
        Ok(dbi)
    }

    pub fn put(&mut self, dbi: u32, key: &[u8], data: &[u8], put_flags: u32) -> CResult<()> {
        let mut k = mk_val(key);
        let mut v = mk_val(data);
        unsafe {
            chk(
                sys::mdb_put(self.txn, dbi, &mut k, &mut v, put_flags),
                "put",
            )
        }
    }

    pub fn del(&mut self, dbi: u32, key: &[u8]) -> CResult<()> {
        let mut k = mk_val(key);
        unsafe { chk(sys::mdb_del(self.txn, dbi, &mut k, ptr::null_mut()), "del") }
    }

    pub fn commit(mut self) -> CResult<()> {
        self.finished = true;
        unsafe { chk(sys::mdb_txn_commit(self.txn), "txn_commit") }
    }

    pub fn abort(mut self) {
        self.finished = true;
        unsafe { sys::mdb_txn_abort(self.txn) }
    }

    pub fn raw_txn(&self) -> *mut sys::MDB_txn {
        self.txn
    }
}

impl Drop for CRwTxn<'_> {
    fn drop(&mut self) {
        if !self.finished {
            unsafe { sys::mdb_txn_abort(self.txn) }
        }
    }
}

// ---------------------------------------------------------------------------
// CCursor — read-only cursor. RW-cursor ops are issued through mdb_put/mdb_del
// which gives parity with how we've wired lmdb-rs-core's bench paths.
// ---------------------------------------------------------------------------

pub struct CCursor<'txn> {
    cur: *mut sys::MDB_cursor,
    _marker: std::marker::PhantomData<&'txn ()>,
}

/// Cursor operations mirroring LMDB's `MDB_cursor_op`.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum COp {
    First = 0,
    GetCurrent = 4,
    Last = 6,
    Next = 8,
    NextDup = 9,
    Prev = 12,
    Set = 15,
    SetRange = 17,
}

impl<'txn> CCursor<'txn> {
    /// Fetch `(key, data)` using the given cursor op.
    ///
    /// When the op does not accept an input key (e.g. First/Next), pass `None`.
    pub fn get(&mut self, key: Option<&[u8]>, op: COp) -> CResult<(&'txn [u8], &'txn [u8])> {
        let mut k = match key {
            Some(k) => mk_val(k),
            None => empty_val(),
        };
        let mut v = empty_val();
        unsafe {
            chk(
                sys::mdb_cursor_get(self.cur, &mut k, &mut v, op as u32),
                "cursor_get",
            )?;
            Ok((val_as_slice(&k), val_as_slice(&v)))
        }
    }
}

impl Drop for CCursor<'_> {
    fn drop(&mut self) {
        unsafe { sys::mdb_cursor_close(self.cur) }
    }
}

// ---------------------------------------------------------------------------
// MDB_val helpers
// ---------------------------------------------------------------------------

fn mk_val(bytes: &[u8]) -> sys::MDB_val {
    sys::MDB_val {
        mv_size: bytes.len(),
        mv_data: bytes.as_ptr() as *mut c_void,
    }
}

fn empty_val() -> sys::MDB_val {
    sys::MDB_val {
        mv_size: 0,
        mv_data: ptr::null_mut(),
    }
}

unsafe fn val_as_slice<'a>(v: &sys::MDB_val) -> &'a [u8] {
    if v.mv_size == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(v.mv_data as *const u8, v.mv_size) }
    }
}

// ---------------------------------------------------------------------------
// Tests — behavior parity checks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn test_should_roundtrip_put_get() {
        let dir = tempdir();
        let env = CEnv::open(dir.path(), 10 * 1024 * 1024, 0, 0).expect("open");
        {
            let mut txn = env.begin_rw().expect("rw");
            let dbi = txn.main_dbi().expect("dbi");
            txn.put(dbi, b"hello", b"world", 0).expect("put");
            txn.commit().expect("commit");
        }
        {
            let txn = env.begin_ro().expect("ro");
            let dbi = txn.main_dbi().expect("dbi");
            assert_eq!(txn.get(dbi, b"hello").expect("get"), b"world");
        }
    }

    #[test]
    fn test_should_iter_cursor_first_next() {
        let dir = tempdir();
        let env = CEnv::open(dir.path(), 10 * 1024 * 1024, 0, 0).expect("open");
        {
            let mut txn = env.begin_rw().expect("rw");
            let dbi = txn.main_dbi().expect("dbi");
            for i in 0u32..64 {
                let k = i.to_be_bytes();
                let v = [0xAA; 8];
                txn.put(dbi, &k, &v, 0).expect("put");
            }
            txn.commit().expect("commit");
        }
        let txn = env.begin_ro().expect("ro");
        let dbi = txn.main_dbi().expect("dbi");
        let mut cursor = txn.open_cursor(dbi).expect("cursor");
        let (_k, _v) = cursor.get(None, COp::First).expect("first");
        let mut count = 1u32;
        while cursor.get(None, COp::Next).is_ok() {
            count += 1;
        }
        assert_eq!(count, 64);
    }

    #[test]
    fn test_should_return_notfound_on_missing_key() {
        let dir = tempdir();
        let env = CEnv::open(dir.path(), 10 * 1024 * 1024, 0, 0).expect("open");
        let txn = env.begin_ro().expect("ro");
        let dbi = txn.main_dbi().expect("dbi");
        let err = txn.get(dbi, b"nope").unwrap_err();
        assert_eq!(err.code, -30798); // MDB_NOTFOUND
    }
}
