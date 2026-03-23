//! Local embedding provider using ONNX models via fastembed.
//!
//! Enable with the `local-embeddings` feature flag. Downloads the model
//! from HuggingFace Hub on first use (~22 MB for the default AllMiniLML6V2).

use std::sync::Mutex;

use crate::error::{AlayaError, Result};
use crate::provider::EmbeddingProvider;

pub use fastembed::EmbeddingModel;

/// Local embedding provider backed by ONNX Runtime via [`fastembed`].
///
/// This provider runs inference entirely on-device — no external API calls
/// required. The model weights are downloaded once from HuggingFace Hub and
/// cached locally.
///
/// # Default model
///
/// [`EmbeddingModel::AllMiniLML6V2`] — 384 dimensions, fast, ~22 MB download.
///
/// The inner model is wrapped in a [`Mutex`] because
/// [`fastembed::TextEmbedding::embed`] requires `&mut self`, while the
/// [`EmbeddingProvider`] trait takes `&self` (and is `Send + Sync`).
pub struct LocalEmbeddingProvider {
    model: Mutex<fastembed::TextEmbedding>,
    dimensions: usize,
}

impl LocalEmbeddingProvider {
    /// Create a provider with the default model ([`EmbeddingModel::AllMiniLML6V2`], 384 dimensions).
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create a provider with a specific embedding model.
    pub fn with_model(model: EmbeddingModel) -> Result<Self> {
        let mut text_embedding = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(model).with_show_download_progress(false),
        )
        .map_err(|e| AlayaError::InvalidInput(format!("Failed to load embedding model: {e}")))?;

        // Determine dimensions by embedding a probe string.
        let test = text_embedding
            .embed(vec!["test"], None)
            .map_err(|e| AlayaError::InvalidInput(format!("Failed to determine dimensions: {e}")))?;
        let dimensions = test.first().map(|v| v.len()).unwrap_or(384);

        Ok(Self {
            model: Mutex::new(text_embedding),
            dimensions,
        })
    }

    /// Return the number of dimensions produced by the underlying model.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl EmbeddingProvider for LocalEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| AlayaError::InvalidInput(format!("Lock poisoned: {e}")))?;
        let results = model
            .embed(vec![text], None)
            .map_err(|e| AlayaError::InvalidInput(format!("Embedding failed: {e}")))?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| AlayaError::InvalidInput("No embedding returned".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let mut model = self
            .model
            .lock()
            .map_err(|e| AlayaError::InvalidInput(format!("Lock poisoned: {e}")))?;
        let results = model
            .embed(owned, None)
            .map_err(|e| AlayaError::InvalidInput(format!("Batch embedding failed: {e}")))?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download (~22 MB)
    fn test_local_embedding_produces_vector() {
        let provider = LocalEmbeddingProvider::new().unwrap();
        let embedding = provider.embed("Hello, world!").unwrap();
        assert!(!embedding.is_empty());
        assert_eq!(embedding.len(), 384); // AllMiniLML6V2 dimensions
    }

    #[test]
    #[ignore] // Requires model download
    fn test_local_embedding_consistent() {
        let provider = LocalEmbeddingProvider::new().unwrap();
        let e1 = provider.embed("test text").unwrap();
        let e2 = provider.embed("test text").unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_local_embedding_different_texts_differ() {
        let provider = LocalEmbeddingProvider::new().unwrap();
        let e1 = provider.embed("cat").unwrap();
        let e2 = provider.embed("quantum physics").unwrap();
        assert_ne!(e1, e2);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_local_embedding_batch() {
        let provider = LocalEmbeddingProvider::new().unwrap();
        let results = provider.embed_batch(&["hello", "world"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 384);
    }

    #[test]
    fn test_dimensions_type_exists() {
        // Compile-time verification that the type and its public API exist.
        // Cannot instantiate without a model download, so just confirm size is non-zero.
        assert!(std::mem::size_of::<LocalEmbeddingProvider>() > 0);
    }
}
