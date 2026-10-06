//! knot — production-oriented Rust decision engine.
//!
//! Jev/Laya-compatible wire protocol, non-autoregressive "System 1" decisions
//! over typed questions (`choice` / `score` / `noul`).

pub mod error;
pub mod integrity;
pub mod lang;
pub mod protocol;
pub mod pyjson;
pub mod router;
pub mod runtime;

#[cfg(feature = "candle")]
pub mod candle_runtime;
#[cfg(any(feature = "onnx", feature = "candle"))]
pub mod engine;
#[cfg(feature = "tokenizer")]
pub mod prompt;
#[cfg(feature = "tokenizer")]
pub mod shortlist;

pub use error::{Error, Result};
