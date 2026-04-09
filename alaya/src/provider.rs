use crate::error::Result;
use crate::types::*;

/// The agent provides this trait to support intelligent consolidation.
/// Alaya never calls an LLM directly — the agent owns the LLM connection.
pub trait ConsolidationProvider {
    /// Extract semantic knowledge from a batch of episodes.
    fn extract_knowledge(&self, episodes: &[Episode]) -> Result<Vec<NewSemanticNode>>;

    /// Extract behavioral impressions from an interaction.
    fn extract_impressions(&self, interaction: &Interaction) -> Result<Vec<NewImpression>>;

    /// Detect whether two semantic nodes contradict each other.
    fn detect_contradiction(&self, a: &SemanticNode, b: &SemanticNode) -> Result<bool>;
}

/// A no-op provider for when no LLM is available.
/// Consolidation and perfuming simply skip the LLM-dependent steps.
pub struct NoOpProvider;

impl ConsolidationProvider for NoOpProvider {
    fn extract_knowledge(&self, _episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        Ok(vec![])
    }

    fn extract_impressions(&self, _interaction: &Interaction) -> Result<Vec<NewImpression>> {
        Ok(vec![])
    }

    fn detect_contradiction(&self, _a: &SemanticNode, _b: &SemanticNode) -> Result<bool> {
        Ok(false)
    }
}

/// Trait for automatic embedding generation.
///
/// Implement this to auto-embed episodes, semantic nodes, and queries.
/// When no provider is set, embeddings must be provided manually.
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for the given text.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts. Default implementation
    /// calls `embed()` for each text sequentially.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Trait for automatic knowledge extraction from episodes.
///
/// Implement this with your preferred LLM (Haiku, GPT-4o-mini, local Ollama)
/// to enable auto-consolidation. When set on Alaya, the MCP server
/// will automatically extract facts instead of prompting the agent.
pub trait ExtractionProvider: Send + Sync {
    /// Extract semantic knowledge from a batch of episodes.
    fn extract(&self, episodes: &[Episode]) -> Result<Vec<NewSemanticNode>>;
}

/// Mock extraction provider for tests. Returns pre-configured nodes.
pub struct MockExtractionProvider {
    nodes: Vec<NewSemanticNode>,
}

impl MockExtractionProvider {
    pub fn new(nodes: Vec<NewSemanticNode>) -> Self {
        Self { nodes }
    }

    pub fn empty() -> Self {
        Self { nodes: vec![] }
    }
}

impl ExtractionProvider for MockExtractionProvider {
    fn extract(&self, _episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        Ok(self.nodes.clone())
    }
}

impl ExtractionProvider for NoOpProvider {
    fn extract(&self, _episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        Ok(vec![])
    }
}

/// Mock embedding provider for tests. Returns deterministic embeddings
/// based on a hash of the input text.
pub struct MockEmbeddingProvider {
    dim: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Simple hash-based deterministic embedding
        let hash = text
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let emb: Vec<f32> = (0..self.dim)
            .map(|i| {
                let val = ((hash.wrapping_mul((i as u64).wrapping_add(1))) % 1000) as f32 / 1000.0;
                val * 2.0 - 1.0 // normalize to [-1, 1]
            })
            .collect();
        Ok(emb)
    }
}

#[cfg(test)]
pub struct MockProvider {
    pub knowledge: Vec<NewSemanticNode>,
    pub impressions: Vec<NewImpression>,
}

#[cfg(test)]
impl MockProvider {
    pub fn empty() -> Self {
        Self {
            knowledge: vec![],
            impressions: vec![],
        }
    }

    pub fn with_knowledge(knowledge: Vec<NewSemanticNode>) -> Self {
        Self {
            knowledge,
            impressions: vec![],
        }
    }

    pub fn with_impressions(impressions: Vec<NewImpression>) -> Self {
        Self {
            knowledge: vec![],
            impressions,
        }
    }
}

#[cfg(test)]
impl ConsolidationProvider for MockProvider {
    fn extract_knowledge(&self, _episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        Ok(self.knowledge.clone())
    }

    fn extract_impressions(&self, _interaction: &Interaction) -> Result<Vec<NewImpression>> {
        Ok(self.impressions.clone())
    }

    fn detect_contradiction(&self, _a: &SemanticNode, _b: &SemanticNode) -> Result<bool> {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedding_provider_embed() {
        let provider = MockEmbeddingProvider::new(3);
        let emb = provider.embed("hello world").unwrap();
        assert_eq!(emb.len(), 3);
    }

    #[test]
    fn test_mock_embedding_provider_batch() {
        let provider = MockEmbeddingProvider::new(4);
        let results = provider.embed_batch(&["hello", "world"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 4);
        assert_eq!(results[1].len(), 4);
    }

    #[test]
    fn test_mock_embedding_provider_deterministic() {
        let provider = MockEmbeddingProvider::new(4);
        let emb1 = provider.embed("same text").unwrap();
        let emb2 = provider.embed("same text").unwrap();
        assert_eq!(emb1, emb2, "same input should produce same embedding");
    }

    #[test]
    fn test_mock_embedding_provider_different_inputs() {
        let provider = MockEmbeddingProvider::new(4);
        let emb1 = provider.embed("hello").unwrap();
        let emb2 = provider.embed("world").unwrap();
        assert_ne!(
            emb1, emb2,
            "different inputs should produce different embeddings"
        );
    }

    #[test]
    fn test_noop_provider_extract_impressions() {
        let provider = NoOpProvider;
        let interaction = Interaction {
            text: "I prefer dark mode".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
        };
        let result = provider.extract_impressions(&interaction).unwrap();
        assert!(
            result.is_empty(),
            "NoOpProvider should return empty impressions"
        );
    }

    #[test]
    fn test_noop_provider_detect_contradiction() {
        let provider = NoOpProvider;
        let a = SemanticNode {
            id: NodeId(1),
            content: "User likes Rust".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![],
            created_at: 1000,
            last_corroborated: 1000,
            corroboration_count: 1,
            category_id: None,
        };
        let b = SemanticNode {
            id: NodeId(2),
            content: "User dislikes Rust".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![],
            created_at: 2000,
            last_corroborated: 2000,
            corroboration_count: 1,
            category_id: None,
        };
        let result = provider.detect_contradiction(&a, &b).unwrap();
        assert!(
            !result,
            "NoOpProvider should always return false for contradictions"
        );
    }

    #[test]
    fn test_mock_extraction_provider_returns_configured_nodes() {
        let nodes = vec![NewSemanticNode {
            content: "User likes Rust".into(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![],
            embedding: None,
        }];
        let provider = MockExtractionProvider::new(nodes.clone());
        let result = provider.extract(&[]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "User likes Rust");
    }

    #[test]
    fn test_mock_extraction_provider_empty() {
        let provider = MockExtractionProvider::empty();
        let result = provider.extract(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_noop_extraction_returns_empty() {
        let provider = NoOpProvider;
        let result = ExtractionProvider::extract(&provider, &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_extraction_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockExtractionProvider>();
    }

    #[test]
    fn test_noop_provider_extract_knowledge() {
        let provider = NoOpProvider;
        let result = provider.extract_knowledge(&[]).unwrap();
        assert!(
            result.is_empty(),
            "NoOpProvider.extract_knowledge should return empty"
        );
    }

    #[test]
    fn test_mock_provider_with_knowledge() {
        let nodes = vec![NewSemanticNode {
            content: "test fact".into(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![],
            embedding: None,
        }];
        let provider = MockProvider::with_knowledge(nodes);
        let result = provider.extract_knowledge(&[]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "test fact");
    }

    #[test]
    fn test_mock_provider_with_impressions() {
        let imps = vec![NewImpression {
            domain: "style".to_string(),
            observation: "prefers bullet points".to_string(),
            valence: 0.9,
        }];
        let provider = MockProvider::with_impressions(imps);
        let interaction = Interaction {
            text: "test".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
        };
        let result = provider.extract_impressions(&interaction).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].domain, "style");
    }

    #[test]
    fn test_mock_provider_empty() {
        let provider = MockProvider::empty();
        let result = provider.extract_knowledge(&[]).unwrap();
        assert!(result.is_empty());
        let interaction = Interaction {
            text: "test".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
        };
        let imps = provider.extract_impressions(&interaction).unwrap();
        assert!(imps.is_empty());
    }

    #[test]
    fn test_mock_provider_detect_contradiction() {
        let provider = MockProvider::empty();
        let a = SemanticNode {
            id: NodeId(1),
            content: "a".into(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![],
            created_at: 0,
            last_corroborated: 0,
            corroboration_count: 1,
            category_id: None,
        };
        let b = SemanticNode {
            id: NodeId(2),
            content: "b".into(),
            node_type: SemanticType::Fact,
            confidence: 0.8,
            source_episodes: vec![],
            created_at: 0,
            last_corroborated: 0,
            corroboration_count: 1,
            category_id: None,
        };
        assert!(!provider.detect_contradiction(&a, &b).unwrap());
    }

    #[test]
    fn test_embedding_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockEmbeddingProvider>();
    }
}
