//! # Alaya
//!
//! A neuroscience and Buddhist psychology-inspired memory engine for conversational AI agents.
//!
//! Alaya (Sanskrit: *alaya-vijnana*, "storehouse consciousness") provides three
//! memory stores, a Hebbian graph overlay, hybrid retrieval with spreading
//! activation, and adaptive lifecycle processes — all without coupling to any
//! specific LLM or agent framework.
//!
//! # Quick Start
//!
//! ```
//! use alaya::{Alaya, NewEpisode, Role, EpisodeContext, Query};
//!
//! let alaya = Alaya::open_in_memory().unwrap();
//!
//! // Store an episode
//! alaya.episodes().store(&NewEpisode {
//!     content: "Rust has zero-cost abstractions.".to_string(),
//!     role: Role::User,
//!     session_id: "session-1".to_string(),
//!     timestamp: 1700000000,
//!     context: EpisodeContext::default(),
//!     embedding: None,
//! }).unwrap();
//!
//! // Query memories
//! let results = alaya.knowledge().query(&Query::simple("Rust")).unwrap();
//! assert!(!results.is_empty());
//! ```

pub(crate) mod db;
pub(crate) mod decay;
pub(crate) mod error;
pub(crate) mod graph;
pub(crate) mod lifecycle;
pub mod managers;
pub(crate) mod provider;
pub(crate) mod retrieval;
pub(crate) mod schema;
pub(crate) mod store;
pub(crate) mod types;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "llm")]
pub mod extraction;

#[cfg(feature = "async")]
pub mod async_store;

#[cfg(feature = "local-embeddings")]
pub mod local_embeddings;

#[cfg(test)]
pub(crate) mod testutil;

use rusqlite::Connection;
use std::path::Path;

pub use error::{AlayaError, Result};
pub use provider::{
    ConsolidationProvider, EmbeddingProvider, ExtractionProvider, MockEmbeddingProvider,
    MockExtractionProvider, NoOpProvider,
};
pub use store::export::{ExportData, ExportReport, ImportReport};
pub use types::*;

#[cfg(feature = "llm")]
pub use extraction::LlmExtractionProvider;

#[cfg(feature = "local-embeddings")]
pub use local_embeddings::LocalEmbeddingProvider;

/// The main entry point. Owns a SQLite connection and exposes the full
/// store / query / lifecycle API via focused sub-managers.
pub struct Alaya {
    conn: Connection,
    embedding_provider: Option<Box<dyn EmbeddingProvider>>,
    extraction_provider: Option<Box<dyn ExtractionProvider>>,
    conflict_strategy: ConflictStrategy,
}

impl Alaya {
    /// Open (or create) a persistent database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = schema::open_db(path.as_ref().to_str().unwrap_or("alaya.db"))?;
        Ok(Self {
            conn,
            embedding_provider: None,
            extraction_provider: None,
            conflict_strategy: ConflictStrategy::default(),
        })
    }

    /// Open an ephemeral in-memory database (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = schema::open_memory_db()?;
        Ok(Self {
            conn,
            embedding_provider: None,
            extraction_provider: None,
            conflict_strategy: ConflictStrategy::default(),
        })
    }

    #[cfg(feature = "sqlcipher")]
    #[cfg(not(tarpaulin_include))]
    pub fn open_encrypted(path: impl AsRef<Path>, key: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "key", key)?;
        conn.execute_batch("SELECT count(*) FROM sqlite_master")
            .map_err(|_| {
                AlayaError::InvalidInput("wrong encryption key or not an encrypted database".into())
            })?;
        schema::initialize(&conn)?;
        Ok(Self {
            conn,
            embedding_provider: None,
            extraction_provider: None,
            conflict_strategy: ConflictStrategy::default(),
        })
    }

    /// Set an embedding provider for automatic embedding generation.
    pub fn set_embedding_provider(&mut self, provider: Box<dyn EmbeddingProvider>) {
        self.embedding_provider = Some(provider);
    }

    /// Set an extraction provider for automatic knowledge extraction.
    pub fn set_extraction_provider(&mut self, provider: Box<dyn ExtractionProvider>) {
        self.extraction_provider = Some(provider);
    }

    /// Configure the conflict resolution strategy.
    pub fn set_conflict_strategy(&mut self, strategy: ConflictStrategy) {
        self.conflict_strategy = strategy;
    }

    #[cfg(feature = "sqlcipher")]
    #[cfg(not(tarpaulin_include))]
    pub fn rekey(&self, new_key: &str) -> Result<()> {
        self.conn.pragma_update(None, "rekey", new_key)?;
        Ok(())
    }

    /// Expose the raw SQLite connection for test-only DB corruption scenarios.
    #[cfg(test)]
    pub(crate) fn raw_conn(&self) -> &Connection {
        &self.conn
    }

    // --- Sub-manager accessors ---

    pub fn episodes(&self) -> managers::Episodes<'_> {
        managers::Episodes {
            conn: &self.conn,
            embedding_provider: self.embedding_provider.as_deref(),
        }
    }

    pub fn knowledge(&self) -> managers::Knowledge<'_> {
        managers::Knowledge {
            conn: &self.conn,
            embedding_provider: self.embedding_provider.as_deref(),
        }
    }

    pub fn lifecycle(&self) -> managers::Lifecycle<'_> {
        managers::Lifecycle {
            conn: &self.conn,
            extraction_provider: self.extraction_provider.as_deref(),
            conflict_strategy: self.conflict_strategy,
        }
    }

    pub fn graph(&self) -> managers::Graph<'_> {
        managers::Graph { conn: &self.conn }
    }

    pub fn admin(&self) -> managers::Admin<'_> {
        managers::Admin { conn: &self.conn }
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    #[test]
    fn test_full_lifecycle() {
        let alaya = Alaya::open_in_memory().unwrap();

        // Store some episodes
        for i in 0..5 {
            alaya
                .episodes()
                .store(&NewEpisode {
                    content: format!("message about Rust programming {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                })
                .unwrap();
        }

        let status = alaya.admin().status().unwrap();
        assert_eq!(status.episode_count, 5);

        // Query
        let results = alaya
            .knowledge()
            .query(&Query::simple("Rust programming"))
            .unwrap();
        assert!(!results.is_empty());

        // Lifecycle with no-op provider
        let noop = NoOpProvider;
        let _cr = alaya.lifecycle().consolidate(&noop).unwrap();
        let _tr = alaya.lifecycle().transform().unwrap();
        let _fr = alaya.lifecycle().forget().unwrap();
    }

    #[test]
    fn test_purge_all() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya
            .episodes()
            .store(&NewEpisode {
                content: "hello".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        assert_eq!(alaya.admin().status().unwrap().episode_count, 1);
        alaya.admin().purge(PurgeFilter::All).unwrap();
        assert_eq!(alaya.admin().status().unwrap().episode_count, 0);
    }

    #[test]
    fn test_open_persistent_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let alaya = Alaya::open(&path).unwrap();

        alaya
            .episodes()
            .store(&NewEpisode {
                content: "persistent test".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        assert_eq!(alaya.admin().status().unwrap().episode_count, 1);

        // Drop and reopen -- data should persist
        drop(alaya);
        let alaya2 = Alaya::open(&path).unwrap();
        assert_eq!(alaya2.admin().status().unwrap().episode_count, 1);
    }

    #[test]
    fn test_store_episode_rejects_empty_content() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.episodes().store(&NewEpisode {
            content: "".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        });
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), AlayaError::InvalidInput(_)),
            "empty content should return InvalidInput"
        );
    }

    #[test]
    fn test_store_episode_rejects_empty_session_id() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.episodes().store(&NewEpisode {
            content: "hello".to_string(),
            role: Role::User,
            session_id: "".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        });
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), AlayaError::InvalidInput(_)),
            "empty session_id should return InvalidInput"
        );
    }

    #[test]
    fn test_query_rejects_empty_text() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.knowledge().query(&Query {
            text: "".to_string(),
            embedding: None,
            context: QueryContext::default(),
            max_results: 5,
            boost_categories: None,
            boost_weights: None,
        });
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), AlayaError::InvalidInput(_)),
            "empty query text should return InvalidInput"
        );
    }

    #[test]
    fn test_query_rejects_zero_max_results() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.knowledge().query(&Query {
            text: "hello".to_string(),
            embedding: None,
            context: QueryContext::default(),
            max_results: 0,
            boost_categories: None,
            boost_weights: None,
        });
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), AlayaError::InvalidInput(_)),
            "zero max_results should return InvalidInput"
        );
    }

    #[test]
    fn test_store_episode_with_embedding_is_atomic() {
        let alaya = Alaya::open_in_memory().unwrap();

        let id = alaya
            .episodes()
            .store(&NewEpisode {
                content: "atomic test".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: Some(vec![1.0, 0.0, 0.0]),
            })
            .unwrap();

        let status = alaya.admin().status().unwrap();
        assert_eq!(status.episode_count, 1);
        assert_eq!(status.embedding_count, 1);
        assert!(id.0 > 0);
    }

    // -----------------------------------------------------------------------
    // Task 5: API-level tests for lib.rs public interface
    // -----------------------------------------------------------------------

    /// Helper: create a simple interaction in a given domain
    fn make_interaction(text: &str, session: &str, ts: i64) -> Interaction {
        Interaction {
            text: text.to_string(),
            role: Role::User,
            session_id: session.to_string(),
            timestamp: ts,
            context: EpisodeContext::default(),
        }
    }

    /// Helper: create a simple new episode
    fn make_new_episode(content: &str, session: &str, ts: i64) -> NewEpisode {
        NewEpisode {
            content: content.to_string(),
            role: Role::User,
            session_id: session.to_string(),
            timestamp: ts,
            context: EpisodeContext::default(),
            embedding: None,
        }
    }

    #[test]
    fn test_preferences_with_domain_filter() {
        let alaya = Alaya::open_in_memory().unwrap();

        // MockProvider returns 1 impression in domain "style" per perfume call
        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "style".to_string(),
            observation: "prefers dark mode".to_string(),
            valence: 1.0,
        }]);

        // Perfume 6 times to exceed CRYSTALLIZATION_THRESHOLD (5)
        for i in 0..6 {
            let interaction =
                make_interaction(&format!("style interaction {i}"), "s1", 1000 + i * 100);
            alaya.lifecycle().perfume(&interaction, &provider).unwrap();
        }

        // Preferences for "style" domain should be non-empty
        let style_prefs = alaya.admin().preferences(Some("style")).unwrap();
        assert!(
            !style_prefs.is_empty(),
            "should have crystallized a style preference"
        );

        // Preferences for a nonexistent domain should be empty
        let none_prefs = alaya.admin().preferences(Some("nonexistent")).unwrap();
        assert!(
            none_prefs.is_empty(),
            "nonexistent domain should have no preferences"
        );
    }

    #[test]
    fn test_preferences_without_filter() {
        let alaya = Alaya::open_in_memory().unwrap();

        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "style".to_string(),
            observation: "prefers bullet points".to_string(),
            valence: 0.8,
        }]);

        for i in 0..6 {
            let interaction =
                make_interaction(&format!("bullet interaction {i}"), "s1", 2000 + i * 100);
            alaya.lifecycle().perfume(&interaction, &provider).unwrap();
        }

        // No domain filter -- should return all preferences
        let all_prefs = alaya.admin().preferences(None).unwrap();
        assert!(
            !all_prefs.is_empty(),
            "preferences(None) should return all crystallized preferences"
        );
    }

    #[test]
    fn test_knowledge_with_type_filter() {
        let alaya = Alaya::open_in_memory().unwrap();

        let mut ep_ids = Vec::new();
        for i in 0..5 {
            let id = alaya
                .episodes()
                .store(&make_new_episode(
                    &format!("knowledge episode {i}"),
                    "s1",
                    1000 + i * 100,
                ))
                .unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![
            NewSemanticNode {
                content: "User likes Rust".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: ep_ids.clone(),
                embedding: None,
            },
            NewSemanticNode {
                content: "User is friends with Alice".to_string(),
                node_type: SemanticType::Relationship,
                confidence: 0.8,
                source_episodes: ep_ids,
                embedding: None,
            },
        ]);

        let report = alaya.lifecycle().consolidate(&provider).unwrap();
        assert_eq!(report.nodes_created, 2);

        // Filter for Facts only
        let facts = alaya
            .knowledge()
            .filter(Some(KnowledgeFilter {
                node_type: Some(SemanticType::Fact),
                ..Default::default()
            }))
            .unwrap();
        assert!(!facts.is_empty(), "should have at least one Fact");
        for f in &facts {
            assert_eq!(f.node_type, SemanticType::Fact);
        }

        // Filter for Relationship only
        let rels = alaya
            .knowledge()
            .filter(Some(KnowledgeFilter {
                node_type: Some(SemanticType::Relationship),
                ..Default::default()
            }))
            .unwrap();
        assert!(!rels.is_empty(), "should have at least one Relationship");
        for r in &rels {
            assert_eq!(r.node_type, SemanticType::Relationship);
        }
    }

    #[test]
    fn test_knowledge_with_min_confidence() {
        let alaya = Alaya::open_in_memory().unwrap();

        let mut ep_ids = Vec::new();
        for i in 0..5 {
            let id = alaya
                .episodes()
                .store(&make_new_episode(
                    &format!("confidence episode {i}"),
                    "s1",
                    1000 + i * 100,
                ))
                .unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![
            NewSemanticNode {
                content: "High confidence fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: ep_ids.clone(),
                embedding: None,
            },
            NewSemanticNode {
                content: "Low confidence fact".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.3,
                source_episodes: ep_ids,
                embedding: None,
            },
        ]);

        alaya.lifecycle().consolidate(&provider).unwrap();

        // Filter with min_confidence 0.7 (no node_type filter => goes through the None branch)
        let filtered = alaya
            .knowledge()
            .filter(Some(KnowledgeFilter {
                min_confidence: Some(0.7),
                ..Default::default()
            }))
            .unwrap();

        assert!(
            !filtered.is_empty(),
            "should have at least one node above 0.7 confidence"
        );
        for node in &filtered {
            assert!(
                node.confidence >= 0.7,
                "node confidence {} should be >= 0.7",
                node.confidence
            );
        }
    }

    #[test]
    fn test_knowledge_with_limit() {
        let alaya = Alaya::open_in_memory().unwrap();

        let mut ep_ids = Vec::new();
        for i in 0..5 {
            let id = alaya
                .episodes()
                .store(&make_new_episode(
                    &format!("limit episode {i}"),
                    "s1",
                    1000 + i * 100,
                ))
                .unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![
            NewSemanticNode {
                content: "Node A".to_string(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: ep_ids.clone(),
                embedding: None,
            },
            NewSemanticNode {
                content: "Node B".to_string(),
                node_type: SemanticType::Concept,
                confidence: 0.8,
                source_episodes: ep_ids.clone(),
                embedding: None,
            },
            NewSemanticNode {
                content: "Node C".to_string(),
                node_type: SemanticType::Event,
                confidence: 0.7,
                source_episodes: ep_ids,
                embedding: None,
            },
        ]);

        alaya.lifecycle().consolidate(&provider).unwrap();

        // Request limit of 1 (no node_type filter)
        let limited = alaya
            .knowledge()
            .filter(Some(KnowledgeFilter {
                limit: Some(1),
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(limited.len(), 1, "limit(1) should return exactly 1 node");
    }

    #[test]
    fn test_neighbors_with_links() {
        let alaya = Alaya::open_in_memory().unwrap();

        let id1 = alaya
            .episodes()
            .store(&make_new_episode("first msg", "s1", 1000))
            .unwrap();

        let mut ep2 = make_new_episode("second msg", "s1", 2000);
        ep2.context.preceding_episode = Some(id1);
        let id2 = alaya.episodes().store(&ep2).unwrap();

        let mut ep3 = make_new_episode("third msg", "s1", 3000);
        ep3.context.preceding_episode = Some(id2);
        let _id3 = alaya.episodes().store(&ep3).unwrap();

        let neighbors = alaya.graph().neighbors(NodeRef::Episode(id1), 2).unwrap();
        assert!(
            !neighbors.is_empty(),
            "episode with temporal links should have neighbors"
        );
    }

    #[test]
    fn test_neighbors_without_links() {
        let alaya = Alaya::open_in_memory().unwrap();

        let id = alaya
            .episodes()
            .store(&make_new_episode("isolated msg", "s1", 1000))
            .unwrap();

        let neighbors = alaya.graph().neighbors(NodeRef::Episode(id), 2).unwrap();
        assert!(
            neighbors.is_empty(),
            "isolated node should have no neighbors"
        );
    }

    #[test]
    fn test_perfume_dedicated() {
        let alaya = Alaya::open_in_memory().unwrap();

        let provider = MockProvider::with_impressions(vec![
            NewImpression {
                domain: "tone".to_string(),
                observation: "prefers formal tone".to_string(),
                valence: 0.9,
            },
            NewImpression {
                domain: "format".to_string(),
                observation: "prefers markdown".to_string(),
                valence: 0.7,
            },
        ]);

        let interaction = make_interaction("Please use formal markdown", "s1", 1000);
        let report = alaya.lifecycle().perfume(&interaction, &provider).unwrap();

        assert_eq!(
            report.impressions_stored, 2,
            "should store both impressions"
        );
        let status = alaya.admin().status().unwrap();
        assert_eq!(status.impression_count, 2);
    }

    #[test]
    fn test_transform_dedicated() {
        let alaya = Alaya::open_in_memory().unwrap();

        alaya
            .episodes()
            .store(&make_new_episode("ep1", "s1", 1000))
            .unwrap();
        alaya
            .episodes()
            .store(&make_new_episode("ep2", "s1", 2000))
            .unwrap();

        graph::links::create_link(
            alaya.raw_conn(),
            NodeRef::Episode(EpisodeId(1)),
            NodeRef::Episode(EpisodeId(2)),
            LinkType::Topical,
            0.01,
        )
        .unwrap();

        assert_eq!(alaya.admin().status().unwrap().link_count, 1);

        let report = alaya.lifecycle().transform().unwrap();
        assert!(report.links_pruned > 0, "weak link should have been pruned");
        assert_eq!(alaya.admin().status().unwrap().link_count, 0);
    }

    #[test]
    fn test_forget_dedicated() {
        let alaya = Alaya::open_in_memory().unwrap();

        let id1 = alaya
            .episodes()
            .store(&make_new_episode("remember me", "s1", 1000))
            .unwrap();
        let id2 = alaya
            .episodes()
            .store(&make_new_episode("remember me too", "s1", 2000))
            .unwrap();

        let s1_before =
            store::strengths::get_strength(alaya.raw_conn(), NodeRef::Episode(id1)).unwrap();
        let s2_before =
            store::strengths::get_strength(alaya.raw_conn(), NodeRef::Episode(id2)).unwrap();
        assert!((s1_before.retrieval_strength - 1.0).abs() < 0.01);
        assert!((s2_before.retrieval_strength - 1.0).abs() < 0.01);

        for _ in 0..5 {
            alaya.lifecycle().forget().unwrap();
        }

        let s1_after =
            store::strengths::get_strength(alaya.raw_conn(), NodeRef::Episode(id1)).unwrap();
        let s2_after =
            store::strengths::get_strength(alaya.raw_conn(), NodeRef::Episode(id2)).unwrap();

        assert!(
            s1_after.retrieval_strength < s1_before.retrieval_strength,
            "retrieval strength should decay: {} -> {}",
            s1_before.retrieval_strength,
            s1_after.retrieval_strength,
        );
        assert!(
            s2_after.retrieval_strength < s2_before.retrieval_strength,
            "retrieval strength should decay: {} -> {}",
            s2_before.retrieval_strength,
            s2_after.retrieval_strength,
        );
    }

    #[test]
    fn test_purge_session() {
        let alaya = Alaya::open_in_memory().unwrap();

        alaya
            .episodes()
            .store(&make_new_episode("s1 msg1", "s1", 1000))
            .unwrap();
        alaya
            .episodes()
            .store(&make_new_episode("s1 msg2", "s1", 2000))
            .unwrap();
        alaya
            .episodes()
            .store(&make_new_episode("s2 msg1", "s2", 3000))
            .unwrap();

        assert_eq!(alaya.admin().status().unwrap().episode_count, 3);

        let report = alaya
            .admin()
            .purge(PurgeFilter::Session("s1".to_string()))
            .unwrap();
        assert_eq!(report.episodes_deleted, 2);

        assert_eq!(alaya.admin().status().unwrap().episode_count, 1);

        let results = alaya.knowledge().query(&Query::simple("s2 msg1")).unwrap();
        assert!(!results.is_empty(), "s2 episodes should still be queryable");
    }

    #[test]
    fn test_purge_older_than() {
        let alaya = Alaya::open_in_memory().unwrap();

        alaya
            .episodes()
            .store(&make_new_episode("old episode", "s1", 1000))
            .unwrap();
        alaya
            .episodes()
            .store(&make_new_episode("new episode", "s1", 2000))
            .unwrap();

        assert_eq!(alaya.admin().status().unwrap().episode_count, 2);

        let report = alaya.admin().purge(PurgeFilter::OlderThan(1500)).unwrap();
        assert_eq!(
            report.episodes_deleted, 1,
            "should delete the episode at ts=1000"
        );

        assert_eq!(
            alaya.admin().status().unwrap().episode_count,
            1,
            "only the newer episode should remain"
        );

        let results = alaya
            .knowledge()
            .query(&Query::simple("new episode"))
            .unwrap();
        assert!(
            !results.is_empty(),
            "the newer episode should still be queryable"
        );
    }

    #[test]
    fn test_neighbors_depth_zero() {
        let alaya = Alaya::open_in_memory().unwrap();

        let id1 = alaya
            .episodes()
            .store(&make_new_episode("first msg", "s1", 1000))
            .unwrap();
        let mut ep2 = make_new_episode("second msg", "s1", 2000);
        ep2.context.preceding_episode = Some(id1);
        alaya.episodes().store(&ep2).unwrap();

        let neighbors = alaya.graph().neighbors(NodeRef::Episode(id1), 0).unwrap();
        assert!(neighbors.is_empty(), "depth=0 should return no neighbors");
    }

    #[test]
    fn test_forget_archives_weak_nodes() {
        let alaya = Alaya::open_in_memory().unwrap();

        let id = alaya
            .episodes()
            .store(&make_new_episode("archival test", "s1", 1000))
            .unwrap();
        assert_eq!(alaya.admin().status().unwrap().episode_count, 1);

        alaya.raw_conn().execute(
            "UPDATE node_strengths SET storage_strength = 0.05, retrieval_strength = 0.01 WHERE node_id = ?1",
            [id.0],
        ).unwrap();

        let report = alaya.lifecycle().forget().unwrap();
        assert!(
            report.nodes_archived > 0,
            "node with low storage+retrieval should be archived"
        );
        assert_eq!(
            alaya.admin().status().unwrap().episode_count,
            0,
            "archived episode should be deleted"
        );
    }

    // -----------------------------------------------------------------------
    // Task 7: categories() and node_category() API tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_categories_api() {
        let alaya = Alaya::open_in_memory().unwrap();
        let cats = alaya.admin().categories(None).unwrap();
        assert!(cats.is_empty());
    }

    #[test]
    fn test_node_category_api() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.admin().node_category(NodeId(999)).unwrap();
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Task 10: purge(All) clears categories
    // -----------------------------------------------------------------------

    #[test]
    fn test_purge_all_clears_categories() {
        let alaya = Alaya::open_in_memory().unwrap();

        alaya.raw_conn().execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
             VALUES ('proto', 'fact', 0.8, 1000, 1000)",
            [],
        ).unwrap();
        let proto_id = NodeId(alaya.raw_conn().last_insert_rowid());

        store::categories::store_category(alaya.raw_conn(), "test-cat", proto_id, None, None)
            .unwrap();
        assert_eq!(alaya.admin().categories(None).unwrap().len(), 1);

        alaya.admin().purge(PurgeFilter::All).unwrap();
        assert!(
            alaya.admin().categories(None).unwrap().is_empty(),
            "purge(All) should delete all categories"
        );
    }

    #[test]
    fn test_knowledge_filter_by_category() {
        let alaya = Alaya::open_in_memory().unwrap();

        let mut ep_ids = Vec::new();
        for i in 0..5 {
            let id = alaya
                .episodes()
                .store(&make_new_episode(
                    &format!("category filter ep {i}"),
                    "s1",
                    1000 + i * 100,
                ))
                .unwrap();
            ep_ids.push(id);
        }

        let provider = MockProvider::with_knowledge(vec![NewSemanticNode {
            content: "User likes Rust".to_string(),
            node_type: SemanticType::Fact,
            confidence: 0.9,
            source_episodes: ep_ids,
            embedding: None,
        }]);
        alaya.lifecycle().consolidate(&provider).unwrap();

        let filtered = alaya
            .knowledge()
            .filter(Some(KnowledgeFilter {
                category: Some("nonexistent-cat".to_string()),
                ..Default::default()
            }))
            .unwrap();
        assert!(
            filtered.is_empty(),
            "filtering by nonexistent category should return empty"
        );

        let all = alaya.knowledge().filter(None).unwrap();
        assert!(!all.is_empty(), "without filter should find nodes");
    }

    // -----------------------------------------------------------------------
    // EmbeddingProvider integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_embedding_provider_auto_embeds_episode() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_embedding_provider(Box::new(MockEmbeddingProvider::new(4)));

        alaya
            .episodes()
            .store(&NewEpisode {
                content: "auto-embedded episode".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        let status = alaya.admin().status().unwrap();
        assert_eq!(
            status.embedding_count, 1,
            "should have auto-embedded the episode"
        );
    }

    #[test]
    fn test_explicit_embedding_overrides_provider() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_embedding_provider(Box::new(MockEmbeddingProvider::new(4)));

        let explicit_emb = vec![1.0, 2.0, 3.0, 4.0];
        alaya
            .episodes()
            .store(&NewEpisode {
                content: "explicit embedding".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: Some(explicit_emb.clone()),
            })
            .unwrap();

        let status = alaya.admin().status().unwrap();
        assert_eq!(status.embedding_count, 1);
    }

    #[test]
    fn test_no_provider_preserves_v1_behavior() {
        let alaya = Alaya::open_in_memory().unwrap();

        alaya
            .episodes()
            .store(&NewEpisode {
                content: "no provider episode".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        let status = alaya.admin().status().unwrap();
        assert_eq!(status.embedding_count, 0, "no auto-embed without provider");
    }

    #[test]
    fn test_embedding_provider_auto_embeds_query() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_embedding_provider(Box::new(MockEmbeddingProvider::new(4)));

        alaya
            .episodes()
            .store(&NewEpisode {
                content: "Rust programming language".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        let results = alaya
            .knowledge()
            .query(&Query::simple("Rust programming"))
            .unwrap();
        assert!(
            !results.is_empty(),
            "query with auto-embedded text should return results"
        );
    }

    #[test]
    fn test_perfume_crystallization_dedicated() {
        let alaya = Alaya::open_in_memory().unwrap();

        let provider = MockProvider::with_impressions(vec![NewImpression {
            domain: "verbosity".to_string(),
            observation: "prefers concise answers".to_string(),
            valence: 0.9,
        }]);

        for i in 0..4 {
            let interaction = make_interaction(&format!("msg {i}"), "s1", 1000 + i * 100);
            let report = alaya.lifecycle().perfume(&interaction, &provider).unwrap();
            assert_eq!(
                report.preferences_crystallized, 0,
                "should not crystallize below threshold"
            );
        }
        assert!(alaya
            .admin()
            .preferences(Some("verbosity"))
            .unwrap()
            .is_empty());

        let interaction = make_interaction("msg 4", "s1", 1400);
        let report = alaya.lifecycle().perfume(&interaction, &provider).unwrap();
        assert_eq!(
            report.preferences_crystallized, 1,
            "5th impression should trigger crystallization"
        );

        let prefs = alaya.admin().preferences(Some("verbosity")).unwrap();
        assert_eq!(prefs.len(), 1);
        assert_eq!(prefs[0].domain, "verbosity");

        let interaction = make_interaction("msg 5", "s1", 1500);
        let report = alaya.lifecycle().perfume(&interaction, &provider).unwrap();
        assert_eq!(report.preferences_crystallized, 0);
        assert_eq!(
            report.preferences_reinforced, 1,
            "should reinforce existing preference"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage: subcategories(), node_content variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_subcategories_empty() {
        let alaya = Alaya::open_in_memory().unwrap();
        let subs = alaya.admin().subcategories(CategoryId(999)).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn test_node_content_episode() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya
            .episodes()
            .store(&make_new_episode("hello world", "s1", 1000))
            .unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Episode(EpisodeId(1)))
            .unwrap();
        assert_eq!(content, Some("hello world".to_string()));
    }

    #[test]
    fn test_node_content_semantic() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya
            .raw_conn()
            .execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
                 VALUES ('semantic test content', 'fact', 0.8, 1000, 1000)",
                [],
            )
            .unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Semantic(NodeId(1)))
            .unwrap();
        assert_eq!(content, Some("semantic test content".to_string()));
    }

    #[test]
    fn test_node_content_category() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya
            .raw_conn()
            .execute(
                "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated)
                 VALUES ('proto', 'fact', 0.8, 1000, 1000)",
                [],
            )
            .unwrap();
        store::categories::store_category(alaya.raw_conn(), "test-category", NodeId(1), None, None)
            .unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Category(CategoryId(1)))
            .unwrap();
        assert_eq!(content, Some("test-category".to_string()));
    }

    #[test]
    fn test_node_content_preference_fallback() {
        let alaya = Alaya::open_in_memory().unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Preference(PreferenceId(42)))
            .unwrap();
        assert_eq!(content, Some("preference#42".to_string()));
    }

    #[test]
    fn test_node_content_not_found() {
        let alaya = Alaya::open_in_memory().unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Episode(EpisodeId(999)))
            .unwrap();
        assert!(content.is_none());

        let content = alaya
            .admin()
            .node_content(NodeRef::Semantic(NodeId(999)))
            .unwrap();
        assert!(content.is_none());

        let content = alaya
            .admin()
            .node_content(NodeRef::Category(CategoryId(999)))
            .unwrap();
        assert!(content.is_none());
    }

    #[test]
    fn test_truncate_label_long_string() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya
            .episodes()
            .store(&make_new_episode(
                "this is a very long content string that exceeds thirty characters easily",
                "s1",
                1000,
            ))
            .unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Episode(EpisodeId(1)))
            .unwrap();
        let label = content.unwrap();
        assert!(
            label.ends_with("..."),
            "long content should be truncated with ..., got: {label}",
        );
        assert!(
            label.len() <= 33,
            "truncated label should be at most 33 chars, got {}",
            label.len()
        );
    }

    #[test]
    fn test_node_content_semantic_not_found_explicit() {
        let alaya = Alaya::open_in_memory().unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Semantic(NodeId(9999)))
            .unwrap();
        assert!(content.is_none());
    }

    #[test]
    fn test_node_content_category_not_found_explicit() {
        let alaya = Alaya::open_in_memory().unwrap();
        let content = alaya
            .admin()
            .node_content(NodeRef::Category(CategoryId(9999)))
            .unwrap();
        assert!(content.is_none());
    }

    #[test]
    fn test_node_category_not_found_explicit() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.admin().node_category(NodeId(99999)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_set_extraction_provider_enables_auto_consolidate() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        assert!(alaya.lifecycle().auto_consolidate().is_err());

        alaya.set_extraction_provider(Box::new(MockExtractionProvider::new(vec![
            NewSemanticNode {
                content: "User prefers Rust".into(),
                node_type: SemanticType::Fact,
                confidence: 0.9,
                source_episodes: vec![],
                embedding: None,
            },
        ])));

        alaya
            .episodes()
            .store(&NewEpisode {
                content: "I really like Rust".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        let report = alaya.lifecycle().auto_consolidate().unwrap();
        assert_eq!(report.nodes_created, 1);
    }

    #[test]
    fn test_auto_consolidate_without_provider_errors() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.lifecycle().auto_consolidate();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("extraction provider"),
            "Error should mention extraction provider: {err_msg}"
        );
    }

    #[test]
    fn test_auto_consolidate_no_unconsolidated_episodes() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_extraction_provider(Box::new(MockExtractionProvider::empty()));
        let report = alaya.lifecycle().auto_consolidate().unwrap();
        assert_eq!(report.nodes_created, 0);
    }

    #[test]
    fn test_auto_consolidate_learns_extracted_nodes() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_extraction_provider(Box::new(MockExtractionProvider::new(vec![
            NewSemanticNode {
                content: "Fact one".into(),
                node_type: SemanticType::Fact,
                confidence: 0.85,
                source_episodes: vec![],
                embedding: None,
            },
            NewSemanticNode {
                content: "Relationship two".into(),
                node_type: SemanticType::Relationship,
                confidence: 0.7,
                source_episodes: vec![],
                embedding: None,
            },
        ])));

        for i in 0..3 {
            alaya
                .episodes()
                .store(&NewEpisode {
                    content: format!("Episode {i}"),
                    role: Role::User,
                    session_id: "s1".into(),
                    timestamp: 1000 + i as i64,
                    context: EpisodeContext::default(),
                    embedding: None,
                })
                .unwrap();
        }

        let report = alaya.lifecycle().auto_consolidate().unwrap();
        assert_eq!(report.nodes_created, 2);

        let knowledge = alaya.knowledge().filter(None).unwrap();
        assert_eq!(knowledge.len(), 2);
    }

    #[test]
    fn test_dream_empty_store_without_interaction() {
        let alaya = Alaya::open_in_memory().unwrap();
        let noop = NoOpProvider;
        let report = alaya.lifecycle().dream(&noop, None).unwrap();

        assert_eq!(report.consolidation.episodes_processed, 0);
        assert!(report.perfuming.is_none());
        assert_eq!(report.transformation.duplicates_merged, 0);
        assert_eq!(report.forgetting.nodes_decayed, 0);
    }

    #[test]
    fn test_dream_with_interaction() {
        let alaya = Alaya::open_in_memory().unwrap();
        let noop = NoOpProvider;

        let interaction = Interaction {
            text: "I prefer dark mode".to_string(),
            role: Role::User,
            session_id: "s1".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
        };

        let report = alaya.lifecycle().dream(&noop, Some(&interaction)).unwrap();

        assert!(report.perfuming.is_some());
        let perf = report.perfuming.unwrap();
        assert_eq!(perf.impressions_stored, 0);
    }

    #[test]
    fn test_dream_runs_full_lifecycle() {
        let alaya = Alaya::open_in_memory().unwrap();

        for i in 0..5 {
            alaya
                .episodes()
                .store(&NewEpisode {
                    content: format!("episode about topic {i}"),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                })
                .unwrap();
        }

        let mock = MockProvider::empty();
        let report = alaya.lifecycle().dream(&mock, None).unwrap();

        assert!(report.consolidation.episodes_processed > 0);
    }

    #[test]
    fn test_dream_report_type_has_all_fields() {
        let report = DreamReport::default();
        let _c: ConsolidationReport = report.consolidation;
        let _p: Option<PerfumingReport> = report.perfuming;
        let _t: TransformationReport = report.transformation;
        let _f: ForgettingReport = report.forgetting;
    }

    #[cfg(feature = "sqlcipher")]
    #[test]
    fn test_open_encrypted_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("encrypted.db");

        let alaya = Alaya::open_encrypted(&path, "test-key-123").unwrap();
        alaya
            .episodes()
            .store(&NewEpisode {
                content: "secret data".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
        drop(alaya);

        let alaya2 = Alaya::open_encrypted(&path, "test-key-123").unwrap();
        assert_eq!(alaya2.admin().status().unwrap().episode_count, 1);
    }

    #[cfg(feature = "sqlcipher")]
    #[test]
    fn test_open_encrypted_wrong_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("encrypted.db");

        let alaya = Alaya::open_encrypted(&path, "correct-key").unwrap();
        alaya
            .episodes()
            .store(&NewEpisode {
                content: "secret".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
        drop(alaya);

        let result = Alaya::open_encrypted(&path, "wrong-key");
        assert!(result.is_err());
    }

    #[cfg(feature = "sqlcipher")]
    #[test]
    fn test_rekey() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rekey.db");

        let alaya = Alaya::open_encrypted(&path, "old-key").unwrap();
        alaya
            .episodes()
            .store(&NewEpisode {
                content: "rekey test".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();
        alaya.rekey("new-key").unwrap();
        drop(alaya);

        assert!(Alaya::open_encrypted(&path, "old-key").is_err());

        let alaya2 = Alaya::open_encrypted(&path, "new-key").unwrap();
        assert_eq!(alaya2.admin().status().unwrap().episode_count, 1);
    }

    // -----------------------------------------------------------------------
    // Conflict resolution API tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconcile_default_strategy() {
        let alaya = Alaya::open_in_memory().unwrap();
        let report = alaya.lifecycle().reconcile().unwrap();
        assert_eq!(report.conflicts_detected, 0);
    }

    #[test]
    fn test_set_conflict_strategy() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya.set_conflict_strategy(ConflictStrategy::Confidence);
        let report = alaya.lifecycle().reconcile().unwrap();
        assert_eq!(report.conflicts_detected, 0);
    }

    #[test]
    fn test_conflicts_empty_store() {
        let alaya = Alaya::open_in_memory().unwrap();
        let conflicts = alaya.lifecycle().conflicts().unwrap();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_resolve_conflict_manual() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya
            .knowledge()
            .learn(vec![
                NewSemanticNode {
                    content: "user prefers dark mode".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.9,
                    source_episodes: vec![],
                    embedding: Some(vec![0.9, 0.1, 0.0]),
                },
                NewSemanticNode {
                    content: "user prefers light mode".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.8,
                    source_episodes: vec![],
                    embedding: Some(vec![0.85, 0.15, 0.0]),
                },
            ])
            .unwrap();

        let mut alaya = alaya;
        alaya.set_conflict_strategy(ConflictStrategy::Manual);
        alaya.lifecycle().reconcile().unwrap();

        let conflicts = alaya.lifecycle().conflicts().unwrap();
        assert_eq!(conflicts.len(), 1);

        let winner = conflicts[0].node_a;
        alaya
            .lifecycle()
            .resolve_conflict(conflicts[0].id, winner)
            .unwrap();

        let remaining = alaya.lifecycle().conflicts().unwrap();
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_resolve_conflict_picks_node_b_winner() {
        let mut alaya = Alaya::open_in_memory().unwrap();
        alaya
            .knowledge()
            .learn(vec![
                NewSemanticNode {
                    content: "user uses vim".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.9,
                    source_episodes: vec![],
                    embedding: Some(vec![0.9, 0.1, 0.0]),
                },
                NewSemanticNode {
                    content: "user uses emacs".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.8,
                    source_episodes: vec![],
                    embedding: Some(vec![0.85, 0.15, 0.0]),
                },
            ])
            .unwrap();

        alaya.set_conflict_strategy(ConflictStrategy::Manual);
        alaya.lifecycle().reconcile().unwrap();
        let conflicts = alaya.lifecycle().conflicts().unwrap();
        assert_eq!(conflicts.len(), 1);

        let winner = conflicts[0].node_b;
        alaya
            .lifecycle()
            .resolve_conflict(conflicts[0].id, winner)
            .unwrap();

        let remaining = alaya.lifecycle().conflicts().unwrap();
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_node_content_db_error_propagates() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya
            .raw_conn()
            .execute_batch("DROP TABLE episodes")
            .unwrap();
        let result = alaya.admin().node_content(NodeRef::Episode(EpisodeId(1)));
        assert!(result.is_err(), "should propagate DB error, not NotFound");
        assert!(
            !matches!(result, Ok(None)),
            "DB error should not become Ok(None)"
        );
    }
}
