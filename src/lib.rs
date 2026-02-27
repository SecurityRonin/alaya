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
//! use alaya::{AlayaStore, NewEpisode, Role, EpisodeContext, Query};
//!
//! let store = AlayaStore::open_in_memory().unwrap();
//!
//! // Store an episode
//! store.store_episode(&NewEpisode {
//!     content: "Rust has zero-cost abstractions.".to_string(),
//!     role: Role::User,
//!     session_id: "session-1".to_string(),
//!     timestamp: 1700000000,
//!     context: EpisodeContext::default(),
//!     embedding: None,
//! }).unwrap();
//!
//! // Query memories
//! let results = store.query(&Query::simple("Rust")).unwrap();
//! assert!(!results.is_empty());
//! ```

pub(crate) mod error;
pub(crate) mod types;
pub(crate) mod schema;
pub(crate) mod store;
pub(crate) mod graph;
pub(crate) mod retrieval;
pub(crate) mod lifecycle;
pub(crate) mod provider;

use rusqlite::Connection;
use std::path::Path;

pub use error::{AlayaError, Result};
pub use provider::{ConsolidationProvider, NoOpProvider};
pub use types::*;

/// The main entry point. Owns a SQLite connection and exposes the full
/// store / query / lifecycle API.
///
/// # Examples
///
/// ```
/// let store = alaya::AlayaStore::open_in_memory().unwrap();
/// let status = store.status().unwrap();
/// assert_eq!(status.episode_count, 0);
/// ```
pub struct AlayaStore {
    conn: Connection,
}

impl AlayaStore {
    /// Open (or create) a persistent database at `path`.
    ///
    /// # Examples
    ///
    /// ```
    /// let dir = tempfile::tempdir().unwrap();
    /// let store = alaya::AlayaStore::open(dir.path().join("test.db")).unwrap();
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = schema::open_db(path.as_ref().to_str().unwrap_or("alaya.db"))?;
        Ok(Self { conn })
    }

    /// Open an ephemeral in-memory database (useful for tests).
    ///
    /// # Examples
    ///
    /// ```
    /// let store = alaya::AlayaStore::open_in_memory().unwrap();
    /// ```
    pub fn open_in_memory() -> Result<Self> {
        let conn = schema::open_memory_db()?;
        Ok(Self { conn })
    }

    // -----------------------------------------------------------------------
    // Write path
    // -----------------------------------------------------------------------

    /// Store a conversation episode with full context.
    ///
    /// # Errors
    ///
    /// Returns [`AlayaError::InvalidInput`] if `content` or `session_id` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use alaya::{AlayaStore, NewEpisode, Role, EpisodeContext};
    ///
    /// let store = AlayaStore::open_in_memory().unwrap();
    /// let id = store.store_episode(&NewEpisode {
    ///     content: "The user prefers dark mode.".to_string(),
    ///     role: Role::User,
    ///     session_id: "session-1".to_string(),
    ///     timestamp: 1700000000,
    ///     context: EpisodeContext::default(),
    ///     embedding: None,
    /// }).unwrap();
    /// assert!(id.0 > 0);
    /// ```
    pub fn store_episode(&self, episode: &NewEpisode) -> Result<EpisodeId> {
        if episode.content.trim().is_empty() {
            return Err(AlayaError::InvalidInput("episode content must not be empty".into()));
        }
        if episode.session_id.trim().is_empty() {
            return Err(AlayaError::InvalidInput("session_id must not be empty".into()));
        }

        let tx = schema::begin_immediate(&self.conn)?;

        let id = store::episodic::store_episode(&tx, episode)?;

        if let Some(ref emb) = episode.embedding {
            store::embeddings::store_embedding(&tx, "episode", id.0, emb, "")?;
        }

        store::strengths::init_strength(&tx, NodeRef::Episode(id))?;

        if let Some(prev) = episode.context.preceding_episode {
            graph::links::create_link(
                &tx,
                NodeRef::Episode(prev),
                NodeRef::Episode(id),
                LinkType::Temporal,
                0.5,
            )?;
        }

        tx.commit()?;
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Read path
    // -----------------------------------------------------------------------

    /// Hybrid retrieval: BM25 + vector + graph activation -> RRF -> rerank.
    ///
    /// # Errors
    ///
    /// Returns [`AlayaError::InvalidInput`] if `text` is empty or `max_results` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use alaya::{AlayaStore, NewEpisode, Role, EpisodeContext, Query};
    ///
    /// let store = AlayaStore::open_in_memory().unwrap();
    /// store.store_episode(&NewEpisode {
    ///     content: "Rust has zero-cost abstractions.".to_string(),
    ///     role: Role::User,
    ///     session_id: "s1".to_string(),
    ///     timestamp: 1000,
    ///     context: EpisodeContext::default(),
    ///     embedding: None,
    /// }).unwrap();
    ///
    /// let results = store.query(&Query::simple("Rust")).unwrap();
    /// assert!(!results.is_empty());
    /// ```
    pub fn query(&self, q: &Query) -> Result<Vec<ScoredMemory>> {
        if q.text.trim().is_empty() {
            return Err(AlayaError::InvalidInput("query text must not be empty".into()));
        }
        if q.max_results == 0 {
            return Err(AlayaError::InvalidInput("max_results must be greater than 0".into()));
        }

        retrieval::pipeline::execute_query(&self.conn, q)
    }

    /// Get crystallized preferences, optionally filtered by domain.
    ///
    /// # Examples
    ///
    /// ```
    /// let store = alaya::AlayaStore::open_in_memory().unwrap();
    /// let prefs = store.preferences(None).unwrap();
    /// assert!(prefs.is_empty());
    /// ```
    pub fn preferences(&self, domain: Option<&str>) -> Result<Vec<Preference>> {
        store::implicit::get_preferences(&self.conn, domain)
    }

    /// Get semantic knowledge nodes with optional filtering.
    ///
    /// # Examples
    ///
    /// ```
    /// let store = alaya::AlayaStore::open_in_memory().unwrap();
    /// let nodes = store.knowledge(None).unwrap();
    /// assert!(nodes.is_empty());
    /// ```
    pub fn knowledge(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>> {
        let f = filter.unwrap_or_default();
        match f.node_type {
            Some(nt) => store::semantic::find_by_type(
                &self.conn,
                nt,
                f.limit.unwrap_or(100) as u32,
            ),
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
                        &self.conn,
                        *nt,
                        f.limit.unwrap_or(100) as u32,
                    )?;
                    all.append(&mut nodes);
                }
                if let Some(min_conf) = f.min_confidence {
                    all.retain(|n| n.confidence >= min_conf);
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

    /// Get graph neighbors of a node up to `depth` hops.
    ///
    /// # Examples
    ///
    /// ```
    /// use alaya::{AlayaStore, NodeRef, EpisodeId};
    ///
    /// let store = AlayaStore::open_in_memory().unwrap();
    /// let neighbors = store.neighbors(NodeRef::Episode(EpisodeId(1)), 2).unwrap();
    /// assert!(neighbors.is_empty());
    /// ```
    pub fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>> {
        let result = graph::activation::spread_activation(
            &self.conn,
            &[node],
            depth,
            0.05,
            0.6,
        )?;
        let mut pairs: Vec<(NodeRef, f32)> = result
            .into_iter()
            .filter(|(nr, _)| *nr != node)
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(pairs)
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Run consolidation: episodic -> semantic (CLS replay).
    ///
    /// The provider extracts knowledge from episodes. Use [`NoOpProvider`]
    /// if no LLM is available.
    ///
    /// # Examples
    ///
    /// ```
    /// use alaya::{AlayaStore, NoOpProvider};
    ///
    /// let store = AlayaStore::open_in_memory().unwrap();
    /// let report = store.consolidate(&NoOpProvider).unwrap();
    /// assert_eq!(report.nodes_created, 0);
    /// ```
    pub fn consolidate(&self, provider: &dyn ConsolidationProvider) -> Result<ConsolidationReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let report = lifecycle::consolidation::consolidate(&tx, provider)?;
        tx.commit()?;
        Ok(report)
    }

    /// Run perfuming: extract impressions, crystallize preferences (vasana).
    ///
    /// # Examples
    ///
    /// ```
    /// use alaya::{AlayaStore, NoOpProvider, Interaction, Role, EpisodeContext};
    ///
    /// let store = AlayaStore::open_in_memory().unwrap();
    /// let interaction = Interaction {
    ///     text: "I prefer dark themes.".to_string(),
    ///     role: Role::User,
    ///     session_id: "s1".to_string(),
    ///     timestamp: 1000,
    ///     context: EpisodeContext::default(),
    /// };
    /// let report = store.perfume(&interaction, &NoOpProvider).unwrap();
    /// ```
    pub fn perfume(
        &self,
        interaction: &Interaction,
        provider: &dyn ConsolidationProvider,
    ) -> Result<PerfumingReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let report = lifecycle::perfuming::perfume(&tx, interaction, provider)?;
        tx.commit()?;
        Ok(report)
    }

    /// Run transformation: dedup, prune, decay (asraya-paravrtti).
    ///
    /// # Examples
    ///
    /// ```
    /// let store = alaya::AlayaStore::open_in_memory().unwrap();
    /// let report = store.transform().unwrap();
    /// assert_eq!(report.duplicates_merged, 0);
    /// ```
    pub fn transform(&self) -> Result<TransformationReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let report = lifecycle::transformation::transform(&tx)?;
        tx.commit()?;
        Ok(report)
    }

    /// Run forgetting: decay retrieval strengths, archive weak nodes (Bjork).
    ///
    /// # Examples
    ///
    /// ```
    /// let store = alaya::AlayaStore::open_in_memory().unwrap();
    /// let report = store.forget().unwrap();
    /// assert_eq!(report.nodes_decayed, 0);
    /// ```
    pub fn forget(&self) -> Result<ForgettingReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let report = lifecycle::forgetting::forget(&tx)?;
        tx.commit()?;
        Ok(report)
    }

    // -----------------------------------------------------------------------
    // Admin
    // -----------------------------------------------------------------------

    /// Counts across all stores.
    ///
    /// # Examples
    ///
    /// ```
    /// let store = alaya::AlayaStore::open_in_memory().unwrap();
    /// let status = store.status().unwrap();
    /// assert_eq!(status.episode_count, 0);
    /// assert_eq!(status.semantic_node_count, 0);
    /// ```
    pub fn status(&self) -> Result<MemoryStatus> {
        Ok(MemoryStatus {
            episode_count: store::episodic::count_episodes(&self.conn)?,
            semantic_node_count: store::semantic::count_nodes(&self.conn)?,
            preference_count: store::implicit::count_preferences(&self.conn)?,
            impression_count: store::implicit::count_impressions(&self.conn)?,
            link_count: graph::links::count_links(&self.conn)?,
            embedding_count: store::embeddings::count_embeddings(&self.conn)?,
        })
    }

    /// Purge data matching the filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use alaya::{AlayaStore, NewEpisode, Role, EpisodeContext, PurgeFilter};
    ///
    /// let store = AlayaStore::open_in_memory().unwrap();
    /// store.store_episode(&NewEpisode {
    ///     content: "temporary".to_string(),
    ///     role: Role::User,
    ///     session_id: "s1".to_string(),
    ///     timestamp: 1000,
    ///     context: EpisodeContext::default(),
    ///     embedding: None,
    /// }).unwrap();
    ///
    /// store.purge(PurgeFilter::All).unwrap();
    /// assert_eq!(store.status().unwrap().episode_count, 0);
    /// ```
    pub fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let mut report = PurgeReport::default();
        match filter {
            PurgeFilter::Session(ref session_id) => {
                let eps = store::episodic::get_episodes_by_session(&tx, session_id)?;
                let ids: Vec<EpisodeId> = eps.iter().map(|e| e.id).collect();
                report.episodes_deleted = store::episodic::delete_episodes(&tx, &ids)? as u32;
            }
            PurgeFilter::OlderThan(ts) => {
                report.episodes_deleted = tx.execute(
                    "DELETE FROM episodes WHERE timestamp < ?1",
                    [ts],
                )? as u32;
            }
            PurgeFilter::All => {
                tx.execute_batch(
                    "DELETE FROM episodes;
                     DELETE FROM semantic_nodes;
                     DELETE FROM impressions;
                     DELETE FROM preferences;
                     DELETE FROM embeddings;
                     DELETE FROM links;
                     DELETE FROM node_strengths;",
                )?;
            }
        }
        tx.commit()?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    #[test]
    fn test_full_lifecycle() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Store some episodes
        for i in 0..5 {
            store
                .store_episode(&NewEpisode {
                    content: format!("message about Rust programming {}", i),
                    role: Role::User,
                    session_id: "s1".to_string(),
                    timestamp: 1000 + i * 100,
                    context: EpisodeContext::default(),
                    embedding: None,
                })
                .unwrap();
        }

        let status = store.status().unwrap();
        assert_eq!(status.episode_count, 5);

        // Query
        let results = store.query(&Query::simple("Rust programming")).unwrap();
        assert!(!results.is_empty());

        // Lifecycle with no-op provider
        let noop = NoOpProvider;
        let _cr = store.consolidate(&noop).unwrap();
        let _tr = store.transform().unwrap();
        let _fr = store.forget().unwrap();
    }

    #[test]
    fn test_purge_all() {
        let store = AlayaStore::open_in_memory().unwrap();
        store
            .store_episode(&NewEpisode {
                content: "hello".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        assert_eq!(store.status().unwrap().episode_count, 1);
        store.purge(PurgeFilter::All).unwrap();
        assert_eq!(store.status().unwrap().episode_count, 0);
    }

    #[test]
    fn test_open_persistent_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let store = AlayaStore::open(&path).unwrap();

        store
            .store_episode(&NewEpisode {
                content: "persistent test".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .unwrap();

        assert_eq!(store.status().unwrap().episode_count, 1);

        // Drop and reopen — data should persist
        drop(store);
        let store2 = AlayaStore::open(&path).unwrap();
        assert_eq!(store2.status().unwrap().episode_count, 1);
    }

    #[test]
    fn test_store_episode_rejects_empty_content() {
        let store = AlayaStore::open_in_memory().unwrap();
        let result = store.store_episode(&NewEpisode {
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
        let store = AlayaStore::open_in_memory().unwrap();
        let result = store.store_episode(&NewEpisode {
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
        let store = AlayaStore::open_in_memory().unwrap();
        let result = store.query(&Query {
            text: "".to_string(),
            embedding: None,
            context: QueryContext::default(),
            max_results: 5,
        });
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), AlayaError::InvalidInput(_)),
            "empty query text should return InvalidInput"
        );
    }

    #[test]
    fn test_query_rejects_zero_max_results() {
        let store = AlayaStore::open_in_memory().unwrap();
        let result = store.query(&Query {
            text: "hello".to_string(),
            embedding: None,
            context: QueryContext::default(),
            max_results: 0,
        });
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), AlayaError::InvalidInput(_)),
            "zero max_results should return InvalidInput"
        );
    }

    #[test]
    fn test_store_episode_with_embedding_is_atomic() {
        let store = AlayaStore::open_in_memory().unwrap();

        let id = store
            .store_episode(&NewEpisode {
                content: "atomic test".to_string(),
                role: Role::User,
                session_id: "s1".to_string(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: Some(vec![1.0, 0.0, 0.0]),
            })
            .unwrap();

        let status = store.status().unwrap();
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
        let store = AlayaStore::open_in_memory().unwrap();

        // MockProvider returns 1 impression in domain "style" per perfume call
        let provider = MockProvider::with_impressions(vec![
            NewImpression {
                domain: "style".to_string(),
                observation: "prefers dark mode".to_string(),
                valence: 1.0,
            },
        ]);

        // Perfume 6 times to exceed CRYSTALLIZATION_THRESHOLD (5)
        for i in 0..6 {
            let interaction = make_interaction(
                &format!("style interaction {}", i),
                "s1",
                1000 + i * 100,
            );
            store.perfume(&interaction, &provider).unwrap();
        }

        // Preferences for "style" domain should be non-empty
        let style_prefs = store.preferences(Some("style")).unwrap();
        assert!(!style_prefs.is_empty(), "should have crystallized a style preference");

        // Preferences for a nonexistent domain should be empty
        let none_prefs = store.preferences(Some("nonexistent")).unwrap();
        assert!(none_prefs.is_empty(), "nonexistent domain should have no preferences");
    }

    #[test]
    fn test_preferences_without_filter() {
        let store = AlayaStore::open_in_memory().unwrap();

        let provider = MockProvider::with_impressions(vec![
            NewImpression {
                domain: "style".to_string(),
                observation: "prefers bullet points".to_string(),
                valence: 0.8,
            },
        ]);

        for i in 0..6 {
            let interaction = make_interaction(
                &format!("bullet interaction {}", i),
                "s1",
                2000 + i * 100,
            );
            store.perfume(&interaction, &provider).unwrap();
        }

        // No domain filter — should return all preferences
        let all_prefs = store.preferences(None).unwrap();
        assert!(!all_prefs.is_empty(), "preferences(None) should return all crystallized preferences");
    }

    #[test]
    fn test_knowledge_with_type_filter() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Store 5 episodes (enough for consolidation, which requires >= 3 unconsolidated)
        let mut ep_ids = Vec::new();
        for i in 0..5 {
            let id = store.store_episode(&make_new_episode(
                &format!("knowledge episode {}", i),
                "s1",
                1000 + i * 100,
            )).unwrap();
            ep_ids.push(id);
        }

        // MockProvider returns nodes of type Fact and Relationship
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

        let report = store.consolidate(&provider).unwrap();
        assert_eq!(report.nodes_created, 2);

        // Filter for Facts only
        let facts = store.knowledge(Some(KnowledgeFilter {
            node_type: Some(SemanticType::Fact),
            ..Default::default()
        })).unwrap();
        assert!(!facts.is_empty(), "should have at least one Fact");
        for f in &facts {
            assert_eq!(f.node_type, SemanticType::Fact);
        }

        // Filter for Relationship only
        let rels = store.knowledge(Some(KnowledgeFilter {
            node_type: Some(SemanticType::Relationship),
            ..Default::default()
        })).unwrap();
        assert!(!rels.is_empty(), "should have at least one Relationship");
        for r in &rels {
            assert_eq!(r.node_type, SemanticType::Relationship);
        }
    }

    #[test]
    fn test_knowledge_with_min_confidence() {
        let store = AlayaStore::open_in_memory().unwrap();

        let mut ep_ids = Vec::new();
        for i in 0..5 {
            let id = store.store_episode(&make_new_episode(
                &format!("confidence episode {}", i),
                "s1",
                1000 + i * 100,
            )).unwrap();
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

        store.consolidate(&provider).unwrap();

        // Filter with min_confidence 0.7 (no node_type filter => goes through the None branch)
        let filtered = store.knowledge(Some(KnowledgeFilter {
            min_confidence: Some(0.7),
            ..Default::default()
        })).unwrap();

        assert!(!filtered.is_empty(), "should have at least one node above 0.7 confidence");
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
        let store = AlayaStore::open_in_memory().unwrap();

        let mut ep_ids = Vec::new();
        for i in 0..5 {
            let id = store.store_episode(&make_new_episode(
                &format!("limit episode {}", i),
                "s1",
                1000 + i * 100,
            )).unwrap();
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

        store.consolidate(&provider).unwrap();

        // Request limit of 1 (no node_type filter)
        let limited = store.knowledge(Some(KnowledgeFilter {
            limit: Some(1),
            ..Default::default()
        })).unwrap();
        assert_eq!(limited.len(), 1, "limit(1) should return exactly 1 node");
    }

    #[test]
    fn test_neighbors_with_links() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Store 3+ episodes with preceding_episode to create temporal links
        let id1 = store.store_episode(&make_new_episode("first msg", "s1", 1000)).unwrap();

        let mut ep2 = make_new_episode("second msg", "s1", 2000);
        ep2.context.preceding_episode = Some(id1);
        let id2 = store.store_episode(&ep2).unwrap();

        let mut ep3 = make_new_episode("third msg", "s1", 3000);
        ep3.context.preceding_episode = Some(id2);
        let _id3 = store.store_episode(&ep3).unwrap();

        // Spread activation from the first episode with depth 2
        let neighbors = store.neighbors(NodeRef::Episode(id1), 2).unwrap();
        assert!(!neighbors.is_empty(), "episode with temporal links should have neighbors");
    }

    #[test]
    fn test_neighbors_without_links() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Store a single episode with no links
        let id = store.store_episode(&make_new_episode("isolated msg", "s1", 1000)).unwrap();

        let neighbors = store.neighbors(NodeRef::Episode(id), 2).unwrap();
        assert!(neighbors.is_empty(), "isolated node should have no neighbors");
    }

    #[test]
    fn test_perfume_dedicated() {
        let store = AlayaStore::open_in_memory().unwrap();

        // MockProvider returns 2 impressions per perfume call
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
        let report = store.perfume(&interaction, &provider).unwrap();

        assert_eq!(report.impressions_stored, 2, "should store both impressions");
        let status = store.status().unwrap();
        assert_eq!(status.impression_count, 2);
    }

    #[test]
    fn test_transform_dedicated() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Store two episodes to have valid node refs for the link
        store.store_episode(&make_new_episode("ep1", "s1", 1000)).unwrap();
        store.store_episode(&make_new_episode("ep2", "s1", 2000)).unwrap();

        // Create a weak link directly via the graph module (weight 0.01, below prune threshold of 0.02)
        graph::links::create_link(
            &store.conn,
            NodeRef::Episode(EpisodeId(1)),
            NodeRef::Episode(EpisodeId(2)),
            LinkType::Topical,
            0.01,
        ).unwrap();

        assert_eq!(store.status().unwrap().link_count, 1);

        let report = store.transform().unwrap();
        assert!(report.links_pruned > 0, "weak link should have been pruned");
        assert_eq!(store.status().unwrap().link_count, 0);
    }

    #[test]
    fn test_forget_dedicated() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Store episodes (store_episode auto-initializes strength with retrieval_strength = 1.0)
        let id1 = store.store_episode(&make_new_episode("remember me", "s1", 1000)).unwrap();
        let id2 = store.store_episode(&make_new_episode("remember me too", "s1", 2000)).unwrap();

        // Check initial retrieval strength
        let s1_before = store::strengths::get_strength(&store.conn, NodeRef::Episode(id1)).unwrap();
        let s2_before = store::strengths::get_strength(&store.conn, NodeRef::Episode(id2)).unwrap();
        assert!((s1_before.retrieval_strength - 1.0).abs() < 0.01);
        assert!((s2_before.retrieval_strength - 1.0).abs() < 0.01);

        // Run forget multiple times to accumulate decay
        for _ in 0..5 {
            store.forget().unwrap();
        }

        let s1_after = store::strengths::get_strength(&store.conn, NodeRef::Episode(id1)).unwrap();
        let s2_after = store::strengths::get_strength(&store.conn, NodeRef::Episode(id2)).unwrap();

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
        let store = AlayaStore::open_in_memory().unwrap();

        // Store episodes in session "s1"
        store.store_episode(&make_new_episode("s1 msg1", "s1", 1000)).unwrap();
        store.store_episode(&make_new_episode("s1 msg2", "s1", 2000)).unwrap();

        // Store episodes in session "s2"
        store.store_episode(&make_new_episode("s2 msg1", "s2", 3000)).unwrap();

        assert_eq!(store.status().unwrap().episode_count, 3);

        // Purge session "s1"
        let report = store.purge(PurgeFilter::Session("s1".to_string())).unwrap();
        assert_eq!(report.episodes_deleted, 2);

        // s1 episodes gone, s2 remain
        assert_eq!(store.status().unwrap().episode_count, 1);

        // Verify the remaining episode is from s2
        let results = store.query(&Query::simple("s2 msg1")).unwrap();
        assert!(!results.is_empty(), "s2 episodes should still be queryable");
    }

    #[test]
    fn test_purge_older_than() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Store episodes with timestamps 1000 and 2000
        store.store_episode(&make_new_episode("old episode", "s1", 1000)).unwrap();
        store.store_episode(&make_new_episode("new episode", "s1", 2000)).unwrap();

        assert_eq!(store.status().unwrap().episode_count, 2);

        // Purge episodes older than 1500
        let report = store.purge(PurgeFilter::OlderThan(1500)).unwrap();
        assert_eq!(report.episodes_deleted, 1, "should delete the episode at ts=1000");

        assert_eq!(store.status().unwrap().episode_count, 1, "only the newer episode should remain");

        // Verify the remaining episode is the newer one
        let results = store.query(&Query::simple("new episode")).unwrap();
        assert!(!results.is_empty(), "the newer episode should still be queryable");
    }

    #[test]
    fn test_neighbors_depth_zero() {
        let store = AlayaStore::open_in_memory().unwrap();

        // Create a chain with temporal links
        let id1 = store.store_episode(&make_new_episode("first msg", "s1", 1000)).unwrap();
        let mut ep2 = make_new_episode("second msg", "s1", 2000);
        ep2.context.preceding_episode = Some(id1);
        store.store_episode(&ep2).unwrap();

        // depth=0 means zero hops -- should find no neighbors
        let neighbors = store.neighbors(NodeRef::Episode(id1), 0).unwrap();
        assert!(neighbors.is_empty(), "depth=0 should return no neighbors");
    }

    #[test]
    fn test_forget_archives_weak_nodes() {
        let store = AlayaStore::open_in_memory().unwrap();

        let id = store.store_episode(&make_new_episode("archival test", "s1", 1000)).unwrap();
        assert_eq!(store.status().unwrap().episode_count, 1);

        // Directly set storage_strength below the archive threshold (0.1).
        // init_strength sets storage=0.5, which never decreases through normal
        // forget() cycles, so we must set it manually to test the archival path.
        store.conn.execute(
            "UPDATE node_strengths SET storage_strength = 0.05, retrieval_strength = 0.01 WHERE node_id = ?1",
            [id.0],
        ).unwrap();

        // A single forget pass should now archive this node
        let report = store.forget().unwrap();
        assert!(report.nodes_archived > 0, "node with low storage+retrieval should be archived");
        assert_eq!(store.status().unwrap().episode_count, 0, "archived episode should be deleted");
    }

    #[test]
    fn test_perfume_crystallization_dedicated() {
        let store = AlayaStore::open_in_memory().unwrap();

        let provider = MockProvider::with_impressions(vec![
            NewImpression {
                domain: "verbosity".to_string(),
                observation: "prefers concise answers".to_string(),
                valence: 0.9,
            },
        ]);

        // Perfume below threshold -- no crystallization
        for i in 0..4 {
            let interaction = make_interaction(&format!("msg {}", i), "s1", 1000 + i * 100);
            let report = store.perfume(&interaction, &provider).unwrap();
            assert_eq!(report.preferences_crystallized, 0, "should not crystallize below threshold");
        }
        assert!(store.preferences(Some("verbosity")).unwrap().is_empty());

        // Perfume past threshold (5th impression triggers crystallization)
        let interaction = make_interaction("msg 4", "s1", 1400);
        let report = store.perfume(&interaction, &provider).unwrap();
        assert_eq!(report.preferences_crystallized, 1, "5th impression should trigger crystallization");

        let prefs = store.preferences(Some("verbosity")).unwrap();
        assert_eq!(prefs.len(), 1);
        assert_eq!(prefs[0].domain, "verbosity");

        // 6th perfume should reinforce, not crystallize again
        let interaction = make_interaction("msg 5", "s1", 1500);
        let report = store.perfume(&interaction, &provider).unwrap();
        assert_eq!(report.preferences_crystallized, 0);
        assert_eq!(report.preferences_reinforced, 1, "should reinforce existing preference");
    }

}
