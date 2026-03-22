use crate::error::{AlayaError, Result};
use crate::provider::{ConsolidationProvider, EmbeddingProvider, ExtractionProvider, NoOpProvider};
use crate::types::*;
#[allow(deprecated)]
use crate::AlayaStore;
use std::path::Path;
use std::thread::JoinHandle;
use tokio::sync::{mpsc, oneshot};

// ---------------------------------------------------------------------------
// Reply type alias
// ---------------------------------------------------------------------------

type Reply<T> = oneshot::Sender<Result<T>>;

// ---------------------------------------------------------------------------
// Request enum
// ---------------------------------------------------------------------------

enum Request {
    StoreEpisode {
        episode: NewEpisode,
        reply: Reply<EpisodeId>,
    },
    Query {
        query: Query,
        reply: Reply<Vec<ScoredMemory>>,
    },
    Status {
        reply: Reply<MemoryStatus>,
    },
    Consolidate {
        reply: Reply<ConsolidationReport>,
    },
    Learn {
        nodes: Vec<NewSemanticNode>,
        reply: Reply<ConsolidationReport>,
    },
    AutoConsolidate {
        reply: Reply<ConsolidationReport>,
    },
    Perfume {
        interaction: Interaction,
        reply: Reply<PerfumingReport>,
    },
    Transform {
        reply: Reply<TransformationReport>,
    },
    Forget {
        reply: Reply<ForgettingReport>,
    },
    Dream {
        interaction: Option<Interaction>,
        reply: Reply<DreamReport>,
    },
    Preferences {
        domain: Option<String>,
        reply: Reply<Vec<Preference>>,
    },
    Knowledge {
        filter: Option<KnowledgeFilter>,
        reply: Reply<Vec<SemanticNode>>,
    },
    Categories {
        min_stability: Option<f32>,
        reply: Reply<Vec<Category>>,
    },
    Subcategories {
        parent_id: CategoryId,
        reply: Reply<Vec<Category>>,
    },
    NodeCategory {
        node_id: NodeId,
        reply: Reply<Option<Category>>,
    },
    Neighbors {
        node: NodeRef,
        depth: u32,
        reply: Reply<Vec<(NodeRef, f32)>>,
    },
    StrongestLink {
        reply: Reply<Option<(NodeRef, NodeRef, f32)>>,
    },
    NodeContent {
        node: NodeRef,
        reply: Reply<Option<String>>,
    },
    KnowledgeBreakdown {
        reply: Reply<std::collections::HashMap<SemanticType, u64>>,
    },
    EpisodesBySession {
        session_id: String,
        reply: Reply<Vec<Episode>>,
    },
    UnconsolidatedEpisodes {
        limit: u32,
        reply: Reply<Vec<Episode>>,
    },
    Purge {
        filter: PurgeFilter,
        reply: Reply<PurgeReport>,
    },
    SetConsolidationProvider {
        provider: Box<dyn ConsolidationProvider + Send>,
    },
    SetEmbeddingProvider {
        provider: Box<dyn EmbeddingProvider + Send>,
    },
    SetExtractionProvider {
        provider: Box<dyn ExtractionProvider + Send>,
    },
    Reconcile {
        reply: Reply<ReconcileReport>,
    },
    Conflicts {
        reply: Reply<Vec<Conflict>>,
    },
    ResolveConflict {
        conflict_id: ConflictId,
        winner_id: NodeId,
        reply: Reply<()>,
    },
    SetConflictStrategy {
        strategy: ConflictStrategy,
    },
    #[cfg(feature = "sqlcipher")]
    Rekey {
        new_key: String,
        reply: Reply<()>,
    },
    Shutdown,
}

// ---------------------------------------------------------------------------
// Actor loop
// ---------------------------------------------------------------------------

fn run_actor(mut store: AlayaStore, rx: mpsc::Receiver<Request>) {
    let mut consolidation_provider: Box<dyn ConsolidationProvider + Send> = Box::new(NoOpProvider);

    // blocking_recv() is correct here: actor runs on a std::thread, not in a
    // tokio runtime, so we use the blocking variant.
    let mut rx = rx;
    while let Some(req) = rx.blocking_recv() {
        match req {
            Request::StoreEpisode { episode, reply } => {
                let _ = reply.send(store.store_episode(&episode));
            }
            Request::Query { query, reply } => {
                let _ = reply.send(store.query(&query));
            }
            Request::Status { reply } => {
                let _ = reply.send(store.status());
            }
            Request::Consolidate { reply } => {
                let _ = reply.send(store.consolidate(consolidation_provider.as_ref()));
            }
            Request::Learn { nodes, reply } => {
                let _ = reply.send(store.learn(nodes));
            }
            Request::AutoConsolidate { reply } => {
                let _ = reply.send(store.auto_consolidate());
            }
            Request::Perfume { interaction, reply } => {
                let _ = reply.send(store.perfume(&interaction, consolidation_provider.as_ref()));
            }
            Request::Transform { reply } => {
                let _ = reply.send(store.transform());
            }
            Request::Forget { reply } => {
                let _ = reply.send(store.forget());
            }
            Request::Dream { interaction, reply } => {
                let inter_ref = interaction.as_ref();
                let _ = reply.send(store.dream(consolidation_provider.as_ref(), inter_ref));
            }
            Request::Preferences { domain, reply } => {
                let _ = reply.send(store.preferences(domain.as_deref()));
            }
            Request::Knowledge { filter, reply } => {
                #[allow(deprecated)]
                let _ = reply.send(store.knowledge_nodes(filter));
            }
            Request::Categories {
                min_stability,
                reply,
            } => {
                let _ = reply.send(store.categories(min_stability));
            }
            Request::Subcategories { parent_id, reply } => {
                let _ = reply.send(store.subcategories(parent_id));
            }
            Request::NodeCategory { node_id, reply } => {
                let _ = reply.send(store.node_category(node_id));
            }
            Request::Neighbors { node, depth, reply } => {
                let _ = reply.send(store.neighbors(node, depth));
            }
            Request::StrongestLink { reply } => {
                let _ = reply.send(store.strongest_link());
            }
            Request::NodeContent { node, reply } => {
                let _ = reply.send(store.node_content(node));
            }
            Request::KnowledgeBreakdown { reply } => {
                let _ = reply.send(store.knowledge_breakdown());
            }
            Request::EpisodesBySession { session_id, reply } => {
                let _ = reply.send(store.episodes_by_session(&session_id));
            }
            Request::UnconsolidatedEpisodes { limit, reply } => {
                let _ = reply.send(store.unconsolidated_episodes(limit));
            }
            Request::Purge { filter, reply } => {
                let _ = reply.send(store.purge(filter));
            }
            Request::SetConsolidationProvider { provider } => {
                consolidation_provider = provider;
            }
            Request::SetEmbeddingProvider { provider } => {
                store.set_embedding_provider(provider);
            }
            Request::SetExtractionProvider { provider } => {
                store.set_extraction_provider(provider);
            }
            Request::Reconcile { reply } => {
                let _ = reply.send(store.reconcile());
            }
            Request::Conflicts { reply } => {
                let _ = reply.send(store.conflicts());
            }
            Request::ResolveConflict {
                conflict_id,
                winner_id,
                reply,
            } => {
                let _ = reply.send(store.resolve_conflict(conflict_id, winner_id));
            }
            Request::SetConflictStrategy { strategy } => {
                store.set_conflict_strategy(strategy);
            }
            #[cfg(feature = "sqlcipher")]
            Request::Rekey { new_key, reply } => {
                let _ = reply.send(store.rekey(&new_key));
            }
            Request::Shutdown => break,
        }
    }
}

// ---------------------------------------------------------------------------
// AsyncAlayaStore
// ---------------------------------------------------------------------------

/// Async wrapper around [`AlayaStore`] using the actor pattern.
///
/// A dedicated `std::thread` owns the [`AlayaStore`] and its SQLite
/// connection. All public methods send a [`Request`] over a
/// `tokio::sync::mpsc` channel and await a `oneshot` reply.
///
/// The struct is `Send + Sync` and can be shared via `Arc<AsyncAlayaStore>`.
pub struct AsyncAlayaStore {
    tx: mpsc::Sender<Request>,
    handle: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl AsyncAlayaStore {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Open (or create) a persistent database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let store = AlayaStore::open(path)?;
        Ok(Self::spawn(store))
    }

    /// Open an in-memory database (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let store = AlayaStore::open_in_memory()?;
        Ok(Self::spawn(store))
    }

    /// Open (or create) an encrypted database at `path` (requires `sqlcipher` feature).
    #[cfg(feature = "sqlcipher")]
    #[cfg(not(tarpaulin_include))]
    pub fn open_encrypted(path: impl AsRef<Path>, key: &str) -> Result<Self> {
        let store = AlayaStore::open_encrypted(path, key)?;
        Ok(Self::spawn(store))
    }

    fn spawn(store: AlayaStore) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let handle = std::thread::spawn(move || run_actor(store, rx));
        AsyncAlayaStore {
            tx,
            handle: std::sync::Mutex::new(Some(handle)),
        }
    }

    // -----------------------------------------------------------------------
    // Helper: send request and await reply
    // -----------------------------------------------------------------------

    async fn send<T>(&self, make_req: impl FnOnce(Reply<T>) -> Request) -> Result<T> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(make_req(tx))
            .await
            .map_err(|_| AlayaError::ActorDead)?;
        rx.await.map_err(|_| AlayaError::ActorDead)?
    }

    // -----------------------------------------------------------------------
    // Shutdown
    // -----------------------------------------------------------------------

    /// Gracefully shut down the actor thread and join it.
    pub async fn close(self) -> Result<()> {
        // Best-effort send; ignore error if already dead.
        let _ = self.tx.send(Request::Shutdown).await;
        let handle = self.handle.lock().unwrap().take();
        if let Some(h) = handle {
            tokio::task::spawn_blocking(move || h.join())
                .await
                .map_err(|_| AlayaError::ActorDead)?
                .map_err(|_| AlayaError::ActorDead)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Write path
    // -----------------------------------------------------------------------

    pub async fn store_episode(&self, episode: NewEpisode) -> Result<EpisodeId> {
        self.send(|reply| Request::StoreEpisode { episode, reply })
            .await
    }

    // -----------------------------------------------------------------------
    // Read path
    // -----------------------------------------------------------------------

    pub async fn query(&self, query: Query) -> Result<Vec<ScoredMemory>> {
        self.send(|reply| Request::Query { query, reply }).await
    }

    pub async fn status(&self) -> Result<MemoryStatus> {
        self.send(|reply| Request::Status { reply }).await
    }

    pub async fn preferences(&self, domain: Option<String>) -> Result<Vec<Preference>> {
        self.send(|reply| Request::Preferences { domain, reply })
            .await
    }

    pub async fn knowledge(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>> {
        self.send(|reply| Request::Knowledge { filter, reply })
            .await
    }

    pub async fn categories(&self, min_stability: Option<f32>) -> Result<Vec<Category>> {
        self.send(|reply| Request::Categories {
            min_stability,
            reply,
        })
        .await
    }

    pub async fn subcategories(&self, parent_id: CategoryId) -> Result<Vec<Category>> {
        self.send(|reply| Request::Subcategories { parent_id, reply })
            .await
    }

    pub async fn node_category(&self, node_id: NodeId) -> Result<Option<Category>> {
        self.send(|reply| Request::NodeCategory { node_id, reply })
            .await
    }

    pub async fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>> {
        self.send(|reply| Request::Neighbors { node, depth, reply })
            .await
    }

    pub async fn strongest_link(&self) -> Result<Option<(NodeRef, NodeRef, f32)>> {
        self.send(|reply| Request::StrongestLink { reply }).await
    }

    pub async fn node_content(&self, node: NodeRef) -> Result<Option<String>> {
        self.send(|reply| Request::NodeContent { node, reply })
            .await
    }

    pub async fn knowledge_breakdown(
        &self,
    ) -> Result<std::collections::HashMap<SemanticType, u64>> {
        self.send(|reply| Request::KnowledgeBreakdown { reply })
            .await
    }

    pub async fn episodes_by_session(&self, session_id: String) -> Result<Vec<Episode>> {
        self.send(|reply| Request::EpisodesBySession { session_id, reply })
            .await
    }

    pub async fn unconsolidated_episodes(&self, limit: u32) -> Result<Vec<Episode>> {
        self.send(|reply| Request::UnconsolidatedEpisodes { limit, reply })
            .await
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    pub async fn consolidate(&self) -> Result<ConsolidationReport> {
        self.send(|reply| Request::Consolidate { reply }).await
    }

    pub async fn learn(&self, nodes: Vec<NewSemanticNode>) -> Result<ConsolidationReport> {
        self.send(|reply| Request::Learn { nodes, reply }).await
    }

    pub async fn auto_consolidate(&self) -> Result<ConsolidationReport> {
        self.send(|reply| Request::AutoConsolidate { reply }).await
    }

    pub async fn perfume(&self, interaction: Interaction) -> Result<PerfumingReport> {
        self.send(|reply| Request::Perfume { interaction, reply })
            .await
    }

    pub async fn transform(&self) -> Result<TransformationReport> {
        self.send(|reply| Request::Transform { reply }).await
    }

    pub async fn forget(&self) -> Result<ForgettingReport> {
        self.send(|reply| Request::Forget { reply }).await
    }

    pub async fn dream(&self, interaction: Option<Interaction>) -> Result<DreamReport> {
        self.send(|reply| Request::Dream { interaction, reply })
            .await
    }

    pub async fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport> {
        self.send(|reply| Request::Purge { filter, reply }).await
    }

    pub async fn reconcile(&self) -> Result<ReconcileReport> {
        self.send(|reply| Request::Reconcile { reply }).await
    }

    pub async fn conflicts(&self) -> Result<Vec<Conflict>> {
        self.send(|reply| Request::Conflicts { reply }).await
    }

    pub async fn resolve_conflict(
        &self,
        conflict_id: ConflictId,
        winner_id: NodeId,
    ) -> Result<()> {
        self.send(|reply| Request::ResolveConflict {
            conflict_id,
            winner_id,
            reply,
        })
        .await
    }

    pub fn set_conflict_strategy(&self, strategy: ConflictStrategy) {
        let _ = self.tx.try_send(Request::SetConflictStrategy { strategy });
    }

    // -----------------------------------------------------------------------
    // Provider configuration
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Encryption (sqlcipher feature)
    // -----------------------------------------------------------------------

    /// Re-encrypt the database with a new key.
    #[cfg(all(feature = "sqlcipher", not(tarpaulin_include)))]
    pub async fn rekey(&self, new_key: &str) -> Result<()> {
        let new_key = new_key.to_string();
        self.send(|reply| Request::Rekey { new_key, reply }).await
    }

    pub async fn set_consolidation_provider(
        &self,
        provider: Box<dyn ConsolidationProvider + Send>,
    ) -> Result<()> {
        self.tx
            .send(Request::SetConsolidationProvider { provider })
            .await
            .map_err(|_| AlayaError::ActorDead)
    }

    pub async fn set_embedding_provider(
        &self,
        provider: Box<dyn EmbeddingProvider + Send>,
    ) -> Result<()> {
        self.tx
            .send(Request::SetEmbeddingProvider { provider })
            .await
            .map_err(|_| AlayaError::ActorDead)
    }

    pub async fn set_extraction_provider(
        &self,
        provider: Box<dyn ExtractionProvider + Send>,
    ) -> Result<()> {
        self.tx
            .send(Request::SetExtractionProvider { provider })
            .await
            .map_err(|_| AlayaError::ActorDead)
    }
}

// ---------------------------------------------------------------------------
// Drop: best-effort shutdown (no blocking, no panic)
// ---------------------------------------------------------------------------

impl Drop for AsyncAlayaStore {
    fn drop(&mut self) {
        let _ = self.tx.try_send(Request::Shutdown);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_open_in_memory_and_close() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_open_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("async_open.db");
        let store = AsyncAlayaStore::open(&path).unwrap();
        let status = store.status().await.unwrap();
        assert_eq!(status.episode_count, 0);
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_reconcile_and_conflicts() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();
        let report = store.reconcile().await.unwrap();
        assert_eq!(report.conflicts_detected, 0);
        let conflicts = store.conflicts().await.unwrap();
        assert!(conflicts.is_empty());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_resolve_conflict_and_set_strategy() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();

        // Learn two conflicting facts with similar embeddings
        store
            .learn(vec![
                NewSemanticNode {
                    content: "user likes vim".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.9,
                    source_episodes: vec![],
                    embedding: Some(vec![0.9, 0.1, 0.0]),
                },
                NewSemanticNode {
                    content: "user likes emacs".to_string(),
                    node_type: SemanticType::Fact,
                    confidence: 0.8,
                    source_episodes: vec![],
                    embedding: Some(vec![0.85, 0.15, 0.0]),
                },
            ])
            .await
            .unwrap();

        // Set manual strategy so reconcile detects but doesn't resolve
        store.set_conflict_strategy(ConflictStrategy::Manual);
        // Small delay to let the strategy message be processed
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        store.reconcile().await.unwrap();
        let conflicts = store.conflicts().await.unwrap();
        assert_eq!(conflicts.len(), 1, "should detect one conflict");

        // Resolve picking node_b
        let winner = conflicts[0].node_b;
        store
            .resolve_conflict(conflicts[0].id, winner)
            .await
            .unwrap();

        let remaining = store.conflicts().await.unwrap();
        assert!(remaining.is_empty(), "conflict should be resolved");

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_store_and_query() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();

        let episode = NewEpisode {
            content: "Rust has zero-cost abstractions.".to_string(),
            role: Role::User,
            session_id: "session-1".to_string(),
            timestamp: 1_700_000_000,
            context: EpisodeContext::default(),
            embedding: None,
        };
        store.store_episode(episode).await.unwrap();

        let results = store.query(Query::simple("Rust")).await.unwrap();
        assert!(!results.is_empty(), "expected at least one result");

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_status() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();
        let status = store.status().await.unwrap();
        assert_eq!(status.episode_count, 0);
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_dream_without_interaction() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();
        let report = store.dream(None).await.unwrap();
        assert!(report.perfuming.is_none());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_stores() {
        let store = std::sync::Arc::new(AsyncAlayaStore::open_in_memory().unwrap());
        let mut handles = vec![];
        for i in 0..10 {
            let s = store.clone();
            handles.push(tokio::spawn(async move {
                s.store_episode(NewEpisode {
                    content: format!("concurrent message {i}"),
                    role: Role::User,
                    session_id: "s1".into(),
                    timestamp: 1000 + i,
                    context: EpisodeContext::default(),
                    embedding: None,
                })
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let status = store.status().await.unwrap();
        assert_eq!(status.episode_count, 10);
    }

    #[tokio::test]
    async fn test_drop_without_close() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();
        drop(store);
        // If we reach here, Drop worked without blocking or panicking
    }

    #[tokio::test]
    async fn test_lifecycle_via_async() {
        let store = AsyncAlayaStore::open_in_memory().unwrap();
        let tr = store.transform().await.unwrap();
        assert_eq!(tr.duplicates_merged, 0);
        let fr = store.forget().await.unwrap();
        assert_eq!(fr.nodes_decayed, 0);
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_dead_after_close() {
        // Open a store, send Shutdown without consuming self, then verify
        // that subsequent calls return ActorDead once the actor exits.
        let store = AsyncAlayaStore::open_in_memory().unwrap();

        // Tell the actor to shut down; this leaves `store` alive so we can
        // call methods on it afterwards.
        store.tx.send(Request::Shutdown).await.unwrap();

        // Join the actor thread so we know it has exited and dropped its
        // receiver end of the channel.
        let handle = store.handle.lock().unwrap().take();
        if let Some(h) = handle {
            tokio::task::spawn_blocking(move || h.join())
                .await
                .unwrap()
                .unwrap();
        }

        // Now the receiver is gone; status() must return ActorDead.
        let err = store.status().await.unwrap_err();
        assert!(
            matches!(err, AlayaError::ActorDead),
            "expected ActorDead, got: {err}"
        );
    }

    #[cfg(feature = "sqlcipher")]
    #[tokio::test]
    async fn test_async_open_encrypted_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("async_enc.db");

        let store = AsyncAlayaStore::open_encrypted(&path, "async-key").unwrap();
        store
            .store_episode(NewEpisode {
                content: "async secret".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .await
            .unwrap();
        store.close().await.unwrap();

        let store2 = AsyncAlayaStore::open_encrypted(&path, "async-key").unwrap();
        let status = store2.status().await.unwrap();
        assert_eq!(status.episode_count, 1);
        store2.close().await.unwrap();
    }

    #[cfg(feature = "sqlcipher")]
    #[tokio::test]
    async fn test_async_rekey() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("async_rekey.db");

        let store = AsyncAlayaStore::open_encrypted(&path, "old-key").unwrap();
        store
            .store_episode(NewEpisode {
                content: "rekey data".into(),
                role: Role::User,
                session_id: "s1".into(),
                timestamp: 1000,
                context: EpisodeContext::default(),
                embedding: None,
            })
            .await
            .unwrap();
        store.rekey("new-key").await.unwrap();
        store.close().await.unwrap();

        // New key should work
        let store2 = AsyncAlayaStore::open_encrypted(&path, "new-key").unwrap();
        assert_eq!(store2.status().await.unwrap().episode_count, 1);
        store2.close().await.unwrap();
    }
}
