use {anyhow::Result, async_trait::async_trait};

use crate::EmbeddingModelMetadata;

#[async_trait]
pub trait EmbeddingEngine: Send + Sync {
    async fn embed(&self, text: &str, priority: u32) -> Result<Vec<f32>>;
    fn metadata(&self) -> &EmbeddingModelMetadata;
}
