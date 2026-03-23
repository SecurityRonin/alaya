use std::collections::HashMap;

use super::Knowledge;
use crate::db;
use crate::error::{AlayaError, Result};
use crate::types::*;
use crate::{lifecycle, retrieval, store};

impl Knowledge<'_> {
    /// Hybrid retrieval: BM25 + vector + graph activation -> RRF -> rerank.
    ///
    /// ```
    /// use alaya::{Alaya, NewEpisode, Role, EpisodeContext, Query};
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// alaya.episodes().store(&NewEpisode {
    ///     content: "Rust has zero-cost abstractions.".to_string(),
    ///     role: Role::User,
    ///     session_id: "s1".to_string(),
    ///     timestamp: 1700000000,
    ///     context: EpisodeContext::default(),
    ///     embedding: None,
    /// }).unwrap();
    ///
    /// let results = alaya.knowledge().query(&Query::simple("Rust")).unwrap();
    /// assert!(!results.is_empty());
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn query(&self, q: &Query) -> Result<Vec<ScoredMemory>> {
        if q.text.trim().is_empty() {
            return Err(AlayaError::InvalidInput(
                "query text must not be empty".into(),
            ));
        }
        if q.max_results == 0 {
            return Err(AlayaError::InvalidInput(
                "max_results must be greater than 0".into(),
            ));
        }

        const MAX_TEXT_BYTES: usize = 100 * 1024;
        if q.text.len() > MAX_TEXT_BYTES {
            return Err(AlayaError::InvalidInput(format!(
                "query text exceeds 100KB limit ({} bytes)",
                q.text.len()
            )));
        }

        // Auto-embed query text if no embedding provided and provider is set
        if q.embedding.is_none() {
            if let Some(provider) = self.embedding_provider {
                let mut q2 = q.clone();
                q2.embedding = provider.embed(&q.text).ok();
                return retrieval::pipeline::execute_query(self.conn, &q2);
            }
        }
        retrieval::pipeline::execute_query(self.conn, q)
    }

    /// Provider-less consolidation: store pre-extracted semantic knowledge.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn learn(&self, nodes: Vec<NewSemanticNode>) -> Result<ConsolidationReport> {
        const MAX_CONTENT_BYTES: usize = 100 * 1024;
        for node in &nodes {
            if node.content.len() > MAX_CONTENT_BYTES {
                return Err(AlayaError::InvalidInput(format!(
                    "semantic node content exceeds 100KB limit ({} bytes)",
                    node.content.len()
                )));
            }
        }

        db::transact(self.conn, |tx| {
            lifecycle::consolidation::learn_direct(tx, nodes)
        })
    }

    /// Get semantic knowledge nodes with optional filtering.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn filter(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>> {
        let f = filter.unwrap_or_default();
        match f.node_type {
            Some(nt) => {
                store::semantic::find_by_type(self.conn, nt, f.limit.unwrap_or(100) as u32)
            }
            None => {
                // Return all types, ordered by confidence
                let mut all = Vec::new();
                for nt in &[
                    SemanticType::Fact,
                    SemanticType::Relationship,
                    SemanticType::Event,
                    SemanticType::Concept,
                ] {
                    let mut nodes = store::semantic::find_by_type(
                        self.conn,
                        *nt,
                        f.limit.unwrap_or(100) as u32,
                    )?;
                    all.append(&mut nodes);
                }
                if let Some(min_conf) = f.min_confidence {
                    all.retain(|n| n.confidence >= min_conf);
                }
                if let Some(ref cat_label) = f.category {
                    all.retain(|n| {
                        store::categories::get_node_category(self.conn, n.id)
                            .ok()
                            .flatten()
                            .map(|c| c.label == *cat_label)
                            .unwrap_or(false)
                    });
                }
                all.sort_by(|a, b| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if let Some(limit) = f.limit {
                    all.truncate(limit);
                }
                Ok(all)
            }
        }
    }

    /// Count semantic knowledge nodes grouped by type.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn breakdown(&self) -> Result<HashMap<SemanticType, u64>> {
        store::semantic::count_nodes_by_type(self.conn)
    }
}

#[cfg(test)]
mod tests {
    use crate::testutil::fixtures::*;
    use crate::Alaya;

    #[test]
    fn query_returns_stored_episodes() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya.episodes().store(&episode("Rust has zero-cost abstractions")).unwrap();

        let results = alaya
            .knowledge()
            .query(&crate::Query::simple("Rust"))
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn query_rejects_empty_text() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.knowledge().query(&crate::Query::simple(""));
        assert!(result.is_err());
    }

    #[test]
    fn query_rejects_zero_max_results() {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut q = crate::Query::simple("test");
        q.max_results = 0;
        assert!(alaya.knowledge().query(&q).is_err());
    }

    #[test]
    fn learn_creates_nodes() {
        let alaya = Alaya::open_in_memory().unwrap();
        // Store an episode first so learn has source episodes
        alaya.episodes().store(&episode("The sky is blue")).unwrap();

        let report = alaya.knowledge().learn(vec![]).unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn filter_returns_empty_initially() {
        let alaya = Alaya::open_in_memory().unwrap();
        let nodes = alaya.knowledge().filter(None).unwrap();
        assert!(nodes.is_empty());
    }

    #[test]
    fn filter_returns_inserted_nodes() {
        let alaya = Alaya::open_in_memory().unwrap();
        insert_semantic_node(alaya.raw_conn(), "test fact", 0.9);

        let nodes = alaya.knowledge().filter(None).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].content, "test fact");
    }

    #[test]
    fn learn_rejects_oversized_content() {
        let alaya = Alaya::open_in_memory().unwrap();
        let node = crate::NewSemanticNode {
            content: "x".repeat(100 * 1024 + 1),
            node_type: crate::SemanticType::Fact,
            confidence: 0.9,
            source_episodes: vec![],
            embedding: None,
        };
        let result = alaya.knowledge().learn(vec![node]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("100KB"), "error should mention limit: {err}");
    }

    #[test]
    fn query_rejects_oversized_text() {
        let alaya = Alaya::open_in_memory().unwrap();
        let q = crate::Query {
            text: "x".repeat(100 * 1024 + 1),
            ..crate::Query::simple("x")
        };
        let result = alaya.knowledge().query(&q);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("100KB"), "error should mention limit: {err}");
    }

    #[test]
    fn breakdown_empty() {
        let alaya = Alaya::open_in_memory().unwrap();
        let bd = alaya.knowledge().breakdown().unwrap();
        assert!(bd.is_empty());
    }

    #[test]
    fn breakdown_counts_by_type() {
        let alaya = Alaya::open_in_memory().unwrap();
        insert_semantic_node(alaya.raw_conn(), "fact 1", 0.9);
        insert_typed_node(alaya.raw_conn(), "rel 1", "relationship", 0.8);

        let bd = alaya.knowledge().breakdown().unwrap();
        assert_eq!(bd.get(&crate::SemanticType::Fact), Some(&1));
        assert_eq!(bd.get(&crate::SemanticType::Relationship), Some(&1));
    }
}
