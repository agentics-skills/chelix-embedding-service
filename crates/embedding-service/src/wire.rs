//! HTTP wire types for the embedding provider.

use serde::{Deserialize, Serialize};

pub const EMBEDDING_SERVICE_EMBED_PATH: &str = "/v1/embed";
pub const EMBEDDING_SERVICE_HEALTH_PATH: &str = "/health";

/// Background index jobs. Lower than [`EMBEDDING_PRIORITY_SEARCH`].
pub const EMBEDDING_PRIORITY_INDEX: u32 = 0;
/// Interactive search jobs. Served before waiting index jobs.
pub const EMBEDDING_PRIORITY_SEARCH: u32 = 1;
/// Maximum JSON body size accepted by `POST /v1/embed`.
pub const EMBEDDING_MAX_BODY_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelMetadata {
    pub model_name: String,
    pub dimensions: usize,
    pub provider_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub text: String,
    #[serde(default)]
    pub priority: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingServiceError {
    pub error: String,
}
