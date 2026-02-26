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
}
