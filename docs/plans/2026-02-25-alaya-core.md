# Alaya Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the alaya memory engine crate — three stores, graph overlay, retrieval engine, and lifecycle processes.

**Architecture:** SQLite-backed (rusqlite + FTS5) with three logical stores (episodic, semantic, implicit), a Hebbian graph overlay, multi-signal retrieval with RRF fusion and spreading activation, and four background lifecycle processes (consolidation, perfuming, transformation, forgetting). LLM-agnostic via traits.

**Tech Stack:** Rust 2024 edition, rusqlite (bundled, with FTS5), serde/serde_json, tokio (async lifecycle), thiserror (errors).

---

### Task 1: Schema and Database Initialization

**Files:**
- Create: `src/schema.rs`
- Modify: `src/lib.rs`

**Step 1: Write the schema module**

Define all SQLite tables:
- `episodes` (id, content, role, session_id, timestamp, context_json)
- `episodes_fts` (FTS5 virtual table on content)
- `semantic_nodes` (id, content, node_type, confidence, source_episodes_json, created_at, last_corroborated, corroboration_count)
- `impressions` (id, domain, observation, valence, timestamp)
- `preferences` (id, domain, preference, confidence, evidence_count, first_observed, last_reinforced)
- `embeddings` (id, node_type, node_id, embedding BLOB, model, created_at)
- `links` (id, source_type, source_id, target_type, target_id, forward_weight, backward_weight, link_type, created_at, last_activated, activation_count)
- `node_strengths` (node_type, node_id, storage_strength, retrieval_strength, access_count, last_accessed)
- FTS5 sync triggers on episodes

Implement `open_db(path) -> Result<Connection>` with WAL mode, foreign keys, and full schema init.

**Step 2: Write tests for schema init**

Test that `open_db` creates all tables, FTS5 triggers work, and re-opening an existing DB is idempotent.

**Step 3: Commit**

---

### Task 2: Core Types

**Files:**
- Create: `src/types.rs`

Define all public types:
- `EpisodeId`, `NodeId`, `PreferenceId`, `ImpressionId`, `LinkId` (newtype i64s)
- `NodeRef` enum (Episode, Semantic, Preference)
- `Role` enum (User, Assistant, System)
- `SemanticType` enum (Fact, Relationship, Event, Concept)
- `LinkType` enum (Temporal, Topical, Entity, Causal, CoRetrieval)
- `NewEpisode`, `Episode`, `EpisodeContext`
- `SemanticNode`, `NewSemanticNode`
- `Impression`, `NewImpression`
- `Preference`
- `Link`
- `NodeStrength` (storage_strength, retrieval_strength)
- `ScoredMemory`
- `Query`, `QueryContext`
- `MemoryStatus`, `PurgeFilter`, `PurgeReport`
- `ConsolidationReport`, `PerfumingReport`, `TransformationReport`, `ForgettingReport`

**Step 1: Write the types module**

**Step 2: Commit**

---

### Task 3: Error Types

**Files:**
- Create: `src/error.rs`

Define `AlayaError` enum with variants: Db, NotFound, InvalidInput, Serialization, Provider.
Implement `From<rusqlite::Error>` and `From<serde_json::Error>`.
Define `pub type Result<T> = std::result::Result<T, AlayaError>`.

**Step 1: Write the error module**

**Step 2: Commit**

---

### Task 4: Episodic Store

**Files:**
- Create: `src/store/mod.rs`
- Create: `src/store/episodic.rs`

Implement:
- `store_episode(conn, episode) -> Result<EpisodeId>` — INSERT with context serialized as JSON
- `get_episode(conn, id) -> Result<Episode>`
- `get_episodes_by_session(conn, session_id) -> Result<Vec<Episode>>`
- `get_recent_episodes(conn, limit) -> Result<Vec<Episode>>`
- `get_unconsolidated_episodes(conn, limit) -> Result<Vec<Episode>>` — episodes not yet linked to a semantic node
- `delete_episodes(conn, ids) -> Result<u64>`
- `count_episodes(conn) -> Result<u64>`

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 5: Semantic Store

**Files:**
- Create: `src/store/semantic.rs`

Implement:
- `store_semantic_node(conn, node) -> Result<NodeId>`
- `get_semantic_node(conn, id) -> Result<SemanticNode>`
- `update_corroboration(conn, id) -> Result<()>` — increment count, update timestamp
- `find_by_type(conn, node_type, limit) -> Result<Vec<SemanticNode>>`
- `find_similar_nodes(conn, embedding, threshold) -> Result<Vec<(SemanticNode, f32)>>` — cosine similarity
- `supersede_node(conn, old_id, new_id) -> Result<()>` — mark old as superseded
- `delete_node(conn, id) -> Result<()>`
- `count_nodes(conn) -> Result<u64>`

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 6: Implicit Store (Vasana)

**Files:**
- Create: `src/store/implicit.rs`

Implement:
- `store_impression(conn, impression) -> Result<ImpressionId>`
- `get_impressions_by_domain(conn, domain, limit) -> Result<Vec<Impression>>`
- `count_impressions_by_domain(conn, domain) -> Result<u64>`
- `store_preference(conn, pref) -> Result<PreferenceId>`
- `get_preferences(conn, domain: Option<&str>) -> Result<Vec<Preference>>`
- `reinforce_preference(conn, id, new_evidence_count) -> Result<()>`
- `decay_preferences(conn, now, half_life_secs) -> Result<u64>` — reduce confidence of un-reinforced
- `prune_weak_preferences(conn, min_confidence) -> Result<u64>`
- `prune_old_impressions(conn, max_age_secs) -> Result<u64>`

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 7: Embedding Storage

**Files:**
- Create: `src/store/embeddings.rs`

Implement:
- `store_embedding(conn, node_type, node_id, embedding, model) -> Result<()>`
- `get_embedding(conn, node_type, node_id) -> Result<Option<Vec<f32>>>`
- `get_unembedded_episodes(conn, limit) -> Result<Vec<EpisodeId>>`
- `search_by_vector(conn, query_vec, node_type_filter, limit) -> Result<Vec<(NodeRef, f32)>>` — brute-force cosine sim
- `cosine_similarity(a, b) -> f32`
- `serialize_embedding(vec) -> Vec<u8>` / `deserialize_embedding(blob) -> Vec<f32>`

**Step 1: Write tests (cosine similarity, serialization roundtrip, search)**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 8: Node Strengths (Bjork Dual-Strength Model)

**Files:**
- Create: `src/store/strengths.rs`

Implement:
- `init_strength(conn, node_type, node_id) -> Result<()>`
- `get_strength(conn, node_type, node_id) -> Result<NodeStrength>`
- `on_access(conn, node_type, node_id) -> Result<()>` — boost retrieval to 1.0, increment storage
- `boost_retrieval(conn, node_type, node_id, factor) -> Result<()>`
- `suppress_retrieval(conn, node_type, node_id, factor) -> Result<()>` — RIF
- `decay_all_retrieval(conn, decay_rate) -> Result<u64>` — power-law decay
- `find_archivable(conn, storage_thresh, retrieval_thresh) -> Result<Vec<NodeRef>>`

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 9: Graph — Link Management

**Files:**
- Create: `src/graph/mod.rs`
- Create: `src/graph/links.rs`

Implement:
- `create_link(conn, link) -> Result<LinkId>`
- `get_or_create_link(conn, source, target, link_type) -> Result<LinkId>`
- `get_links_from(conn, node) -> Result<Vec<Link>>`
- `get_links_to(conn, node) -> Result<Vec<Link>>`
- `on_co_retrieval(conn, source, target) -> Result<()>` — Hebbian: strengthen forward weight
- `decay_links(conn, half_life_secs) -> Result<u64>`
- `prune_weak_links(conn, threshold) -> Result<u64>`
- `count_links(conn) -> Result<u64>`

**Step 1: Write tests (create, co-retrieval strengthening, decay, pruning)**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 10: Graph — Spreading Activation

**Files:**
- Create: `src/graph/activation.rs`

Implement:
- `spread_activation(conn, seeds, max_depth, threshold, decay_per_hop) -> Result<HashMap<NodeRef, f32>>`

The algorithm:
1. Seed activation from initial nodes (1.0 each)
2. For each hop (up to max_depth, default 2):
   - For each node above threshold:
     - Get outgoing links
     - Distribute activation proportional to edge weight, multiplied by decay
3. Return all nodes with activation above threshold

**Step 1: Write tests (single hop, multi-hop, decay, threshold cutoff)**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 11: Retrieval — BM25 via FTS5

**Files:**
- Create: `src/retrieval/mod.rs`
- Create: `src/retrieval/bm25.rs`

Implement:
- `search_bm25(conn, query, limit) -> Result<Vec<(EpisodeId, f64)>>` — FTS5 rank query, normalize scores to [0,1]

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 12: Retrieval — Vector Search

**Files:**
- Create: `src/retrieval/vector.rs`

Implement:
- `search_vector(conn, query_embedding, limit) -> Result<Vec<(NodeRef, f32)>>` — cosine similarity across all embeddings

(Thin wrapper around store/embeddings::search_by_vector for the retrieval pipeline interface)

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 13: Retrieval — Reciprocal Rank Fusion

**Files:**
- Create: `src/retrieval/fusion.rs`

Implement:
- `rrf_merge(result_sets: Vec<Vec<(NodeRef, f64)>>, k: u32) -> Vec<(NodeRef, f64)>`

Algorithm: `score(d) = sum(1.0 / (k + rank_i + 1))` across all result sets where d appears.
Default k=60 (standard RRF constant).

**Step 1: Write tests (single set, two sets, three sets, disjoint, overlapping)**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 14: Retrieval — Context-Weighted Reranking

**Files:**
- Create: `src/retrieval/rerank.rs`

Implement:
- `rerank(candidates, query_context, now) -> Vec<ScoredMemory>`

Scoring per candidate:
- `recency = exp(-age_days / 30.0)`
- `context_sim = jaccard(candidate.topics, query.topics) * 0.5 + sentiment_sim * 0.25 + entity_overlap * 0.25`
- `final_score = base_score * (1.0 + 0.3 * context_sim) * (1.0 + 0.2 * recency)`

Limit output to MAX_RESULTS (default 5).

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 15: Retrieval — Full Query Pipeline

**Files:**
- Create: `src/retrieval/pipeline.rs`

Wire together: BM25 + vector + graph activation -> RRF fusion -> context reranking -> RIF update.

Implement:
- `execute_query(conn, query) -> Result<Vec<ScoredMemory>>`

This is the main entry point for retrieval.

**Step 1: Write integration test with seeded data**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 16: Lifecycle — Consolidation

**Files:**
- Create: `src/lifecycle/mod.rs`
- Create: `src/lifecycle/consolidation.rs`

Implement:
- `consolidate(conn, provider) -> Result<ConsolidationReport>`

Algorithm:
1. Fetch unconsolidated episodes grouped by topic/entity
2. For groups >= CORROBORATION_THRESHOLD (3):
   - Call provider.extract_knowledge(episodes)
   - Store resulting semantic nodes
   - Link new nodes to source episodes
   - Create graph links between new node and existing related nodes
3. Return report (nodes_created, episodes_processed)

**Step 1: Write tests with mock ConsolidationProvider**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 17: Lifecycle — Perfuming

**Files:**
- Create: `src/lifecycle/perfuming.rs`

Implement:
- `perfume(conn, interaction, provider) -> Result<PerfumingReport>`

Algorithm:
1. Call provider.extract_impressions(interaction)
2. Store impressions
3. For each domain with count >= CRYSTALLIZATION_THRESHOLD (5):
   - Cluster recent impressions
   - If coherent cluster found: create or reinforce preference
4. Return report (impressions_stored, preferences_crystallized)

**Step 1: Write tests with mock provider**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 18: Lifecycle — Transformation

**Files:**
- Create: `src/lifecycle/transformation.rs`

Implement:
- `transform(conn) -> Result<TransformationReport>`

Algorithm:
1. Find and merge semantic duplicates (cosine sim > 0.95)
2. Prune weak graph links
3. Decay un-reinforced preferences
4. Prune dormant impressions older than MAX_IMPRESSION_AGE
5. Return report (merged, pruned_links, decayed_prefs, pruned_impressions)

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 19: Lifecycle — Forgetting

**Files:**
- Create: `src/lifecycle/forgetting.rs`

Implement:
- `forget(conn, decay_rate) -> Result<ForgettingReport>`

Algorithm:
1. Decay retrieval strength of all nodes (power-law)
2. Find archivable nodes (low storage AND low retrieval strength)
3. Delete archivable nodes and their embeddings/links
4. Run retrieval-induced forgetting sweep (accumulated suppressions)
5. Return report (decayed, archived, deleted)

**Step 1: Write tests**

**Step 2: Implement**

**Step 3: Commit**

---

### Task 20: Provider Traits

**Files:**
- Create: `src/provider.rs`

Define:
```rust
pub trait ConsolidationProvider: Send + Sync {
    async fn extract_knowledge(&self, episodes: &[Episode]) -> Result<Vec<NewSemanticNode>>;
    async fn extract_impressions(&self, interaction: &Interaction) -> Result<Vec<NewImpression>>;
    async fn detect_contradiction(&self, a: &SemanticNode, b: &SemanticNode) -> Result<bool>;
}
```

Also define `Interaction` struct (the input to perfuming — text, role, context).

**Step 1: Write the trait + Interaction type**

**Step 2: Write a `MockProvider` for tests**

**Step 3: Commit**

---

### Task 21: Public API — AlayaStore Struct

**Files:**
- Modify: `src/lib.rs`

Implement the `AlayaStore` struct that owns a `rusqlite::Connection` and provides
the public API methods, delegating to the internal modules.

```rust
pub struct AlayaStore { conn: Connection }

impl AlayaStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self>;
    pub fn open_in_memory() -> Result<Self>;
    pub fn store_episode(&self, episode: NewEpisode) -> Result<EpisodeId>;
    pub fn query(&self, q: Query) -> Result<Vec<ScoredMemory>>;
    pub fn preferences(&self, domain: Option<&str>) -> Result<Vec<Preference>>;
    pub fn knowledge(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>>;
    pub fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>>;
    pub fn consolidate(&self, provider: &dyn ConsolidationProvider) -> Result<ConsolidationReport>;
    pub fn perfume(&self, interaction: &Interaction, provider: &dyn ConsolidationProvider) -> Result<PerfumingReport>;
    pub fn transform(&self) -> Result<TransformationReport>;
    pub fn forget(&self) -> Result<ForgettingReport>;
    pub fn status(&self) -> Result<MemoryStatus>;
    pub fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport>;
}
```

**Step 1: Write integration tests exercising the full API**

**Step 2: Implement by wiring internal modules**

**Step 3: Commit**

---

### Task 22: README and Final Polish

**Files:**
- Create: `README.md`
- Create: `.gitignore`

Write README with: overview, architecture diagram, quick start, API reference, design philosophy.
Add .gitignore for Rust targets.

**Step 1: Write files**

**Step 2: Final cargo test + cargo clippy**

**Step 3: Commit all, push to GitHub**
