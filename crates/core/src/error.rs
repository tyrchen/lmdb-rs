//! Error types for lmdb-rs.

/// Result type alias for lmdb-rs operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during lmdb-rs operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Key/data pair already exists.
    #[error("key/data pair already exists")]
    KeyExist,

    /// No matching key/data pair found.
    #[error("no matching key/data pair found")]
    NotFound,

    /// Requested page not found — database may be corrupted.
    #[error("requested page not found (corrupted database)")]
    PageNotFound,

    /// Located page was wrong type — database may be corrupted.
    #[error("located page was wrong type (corrupted database)")]
    Corrupted,

    /// Environment had a fatal error, must be reopened.
    #[error("environment had fatal error")]
    Panic,

    /// Database file version does not match library version.
    #[error("database version mismatch")]
    VersionMismatch,

    /// File is not a valid LMDB database.
    #[error("file is not a valid database")]
    Invalid,

    /// Environment map size limit reached.
    #[error("environment mapsize limit reached")]
    MapFull,

    /// Maximum number of named databases reached.
    #[error("max databases limit reached")]
    DbsFull,

    /// Maximum number of readers reached.
    #[error("max readers limit reached")]
    ReadersFull,

    /// Transaction has too many dirty pages.
    #[error("transaction has too many dirty pages")]
    TxnFull,

    /// Internal error: cursor stack limit reached.
    #[error("cursor stack limit reached")]
    CursorFull,

    /// Internal error: page has no more space.
    #[error("page has no more space")]
    PageFull,

    /// Database contents grew beyond environment map size.
    #[error("database contents grew beyond mapsize")]
    MapResized,

    /// Incompatible operation or database flags changed.
    #[error("incompatible operation or database flags changed")]
    Incompatible,

    /// Invalid reuse of reader lock table slot.
    #[error("invalid reuse of reader lock table slot")]
    BadReaderSlot,

    /// Transaction must abort, has a child, or is invalid.
    #[error("transaction must abort, has child, or is invalid")]
    BadTxn,

    /// Unsupported key/data size.
    #[error("unsupported key/data size")]
    BadValSize,

    /// Database handle was closed/changed unexpectedly.
    #[error("database handle was closed/changed unexpectedly")]
    BadDbi,

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
