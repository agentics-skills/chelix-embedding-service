//! HTTP API surface for the embedding provider.

pub mod api;
mod engine_api;
pub mod pool;
pub mod queue;
mod wire;

pub use {engine_api::EmbeddingEngine, wire::*};
