use thiserror::Error;

/// Errors surfaced by the knot library.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("payload too large: {0}")]
    PayloadTooLarge(String),

    #[error("model error: {0}")]
    Model(String),

    /// Issue #37: the routed checkpoint has no source directory and this
    /// deployment configured no checkpoint to fall back to. Serve maps it
    /// to 503 — a configuration problem, not an engine bug.
    #[error(
        "checkpoint {0:?} is not configured; this deployment has no fallback checkpoint (set KNOT_MODEL_DIR or KNOT_DEFAULT_MODEL)"
    )]
    CheckpointUnavailable(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "candle")]
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Model(e.to_string())
    }
}
