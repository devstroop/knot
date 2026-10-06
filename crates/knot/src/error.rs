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
