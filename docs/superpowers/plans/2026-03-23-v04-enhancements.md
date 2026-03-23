# v0.4 Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 10 features covering schema migration, richer queries, export, incremental consolidation, sqlite-vec vector search, local embeddings, hooks, MCP visualization, benchmarks, and conflict resolution.

**Architecture:** Each feature is a self-contained module behind its own feature flag (where appropriate) with TDD. Features are ordered by dependency: foundation first (schema migration), then core improvements (queries, export), then performance (vec search, embeddings), then lifecycle (consolidation, conflicts), then ecosystem (hooks, viz, benchmarks).

**Tech Stack:** Rust, rusqlite, serde, sqlite-vec (optional), fastembed (optional), criterion (dev)

---

### Task 1: Schema Migration Framework

**Files:**
- Modify: `alaya/src/schema.rs`
- Test: `alaya/src/schema.rs` (inline tests)

**Context:** Currently `schema.rs` hardcodes `PRAGMA user_version = 5` and uses ad-hoc `ALTER TABLE` migrations checked via `SELECT column_name ... LIMIT 0`. This needs a proper versioned migration system.

**Implementation:**
- Add a `migrations` array of `(version, sql)` tuples
- On `init_db()`, read current `user_version`, run any migrations with version > current, then set `user_version` to latest
- Refactor existing ad-hoc migrations (category_id, superseded_by) into the migration array
- Keep `CREATE TABLE IF NOT EXISTS` for fresh databases (version 0 → latest)
- Bump schema version to 6

**Tests (TDD):**
1. `test_migration_from_v5_to_v6` — open a v5 DB (created without new columns), verify migration runs
2. `test_fresh_db_gets_latest_version` — open fresh in-memory DB, verify user_version is latest
3. `test_idempotent_migration` — run init_db twice, verify no errors
4. `test_migration_preserves_data` — insert data at v5, migrate, verify data intact

---

### Task 2: Richer Query API

**Files:**
- Modify: `alaya/src/types.rs` (extend `QueryContext`)
- Modify: `alaya/src/retrieval/pipeline.rs` (apply new filters)
- Modify: `alaya/src/retrieval/bm25.rs` (temporal + session filters)
- Test: `alaya/tests/retrieval.rs` (integration tests)

**Context:** `QueryContext` currently has `topics`, `sentiment`, `mentioned_entities`, `current_timestamp`. Need to add temporal filters, session scoping, negative terms, and boost weights.

**Implementation:**
- Add to `QueryContext`:
  - `after_timestamp: Option<i64>` — only return memories after this time
  - `before_timestamp: Option<i64>` — only return memories before this time
  - `session_filter: Option<String>` — restrict to specific session
  - `exclude_terms: Vec<String>` — filter out results containing these terms
- Add `BoostWeights` struct: `{ bm25: f32, vector: f32, graph: f32 }` with `Default` (1.0, 1.0, 1.0)
- Add `boost_weights: Option<BoostWeights>` to `Query`
- Modify BM25 stage to apply temporal + session SQL WHERE clauses
- Modify RRF fusion to use boost weights as multipliers
- Post-filter results to exclude matches containing exclude_terms

**Tests (TDD):**
1. `test_temporal_filter_after` — store episodes at t=1000,2000,3000; query with after=1500, verify only t=2000,3000 returned
2. `test_temporal_filter_before` — query with before=2500, verify only t=1000,2000 returned
3. `test_temporal_filter_range` — query with after=1500 AND before=2500, verify only t=2000
4. `test_session_filter` — store in sessions s1,s2; query with session_filter=s1, verify only s1 results
5. `test_exclude_terms` — store "I like Rust" and "I like Python"; query "like" excluding "Python", verify only Rust result
6. `test_boost_weights_bm25_only` — set vector=0, graph=0; verify only BM25 results contribute
7. `test_default_boost_weights` — verify BoostWeights::default() is (1.0, 1.0, 1.0)

---

### Task 3: Export/Backup API

**Files:**
- Create: `alaya/src/store/export.rs`
- Modify: `alaya/src/store/mod.rs` (add `pub mod export;`)
- Modify: `alaya/src/managers/admin.rs` (add `export_json`, `import_json` methods)
- Modify: `alaya/src/lib.rs` (re-export `ExportReport`, `ImportReport`)
- Test: `alaya/src/store/export.rs` (unit tests)

**Context:** There's `import_claude_mem` and `import_claude_code` but no generic export. Need JSON export/import for portability.

**Implementation:**
- `ExportData` struct (serde): episodes, semantic_nodes, preferences, impressions, categories, links
- `export_json(conn, writer) -> Result<ExportReport>` — serialize all tables to JSON
- `import_json(conn, reader) -> Result<ImportReport>` — deserialize and INSERT, handling ID remapping
- `ExportReport { episodes, nodes, preferences, links, categories }` — counts
- `ImportReport { episodes_imported, nodes_imported, conflicts_skipped }` — counts
- Use `serde_json::to_writer_pretty` for human-readable output

**Tests (TDD):**
1. `test_export_empty_db` — export empty DB, verify valid JSON with zero counts
2. `test_export_roundtrip` — store data, export, open fresh DB, import, verify data matches
3. `test_export_report_counts` — verify report has correct counts
4. `test_import_into_nonempty_db` — import into DB that already has data, verify no duplicates
5. `test_import_remaps_ids` — verify imported data gets new IDs, not conflicting with existing

---

### Task 4: Streaming/Incremental Consolidation

**Files:**
- Modify: `alaya/src/lifecycle/consolidation.rs` (add batch_size parameter)
- Modify: `alaya/src/managers/lifecycle.rs` (add `consolidate_batch` method)
- Modify: `alaya/src/lib.rs` (expose `consolidate_batch` on `Lifecycle`)
- Test: inline in consolidation.rs + integration test

**Context:** `consolidate()` currently processes ALL unconsolidated episodes at once. Need a batch mode.

**Implementation:**
- Add `consolidate_batch(conn, provider, batch_size: u32) -> Result<ConsolidationReport>`
- Existing `consolidate()` calls `consolidate_batch(conn, provider, u32::MAX)`
- Query limits unconsolidated episodes to `batch_size`
- Add `Lifecycle::consolidate_batch(&self, provider, batch_size)` public method

**Tests (TDD):**
1. `test_consolidate_batch_limits_episodes` — store 10 episodes, consolidate_batch(3), verify only 3 processed
2. `test_consolidate_batch_multiple_rounds` — batch(3) twice on 5 episodes, verify all 5 processed
3. `test_consolidate_full_is_batch_max` — verify consolidate() processes all episodes
4. `test_consolidate_batch_zero` — batch(0) should process nothing

---

### Task 5: Embedded Vector Search (vec-sqlite feature)

**Files:**
- Modify: `alaya/Cargo.toml` (add `sqlite-vec` dep + feature flag)
- Modify: `alaya/src/schema.rs` (create vec0 virtual table when feature enabled)
- Create: `alaya/src/store/vec_search.rs` (vec0-based KNN search)
- Modify: `alaya/src/store/mod.rs` (add conditional module)
- Modify: `alaya/src/store/embeddings.rs` (dual-write to vec0 table when feature enabled)
- Modify: `alaya/src/retrieval/vector.rs` (use vec0 when available)
- Test: feature-gated tests

**Context:** Current vector search is O(n) brute-force scan. sqlite-vec provides a vec0 virtual table for faster KNN via MATCH queries. Feature-gated to avoid adding a dependency for users who don't need it.

**Implementation:**
- Feature flag: `vec-sqlite` in Cargo.toml, depends on `sqlite-vec = "0.1"`
- On schema init (when feature enabled): register sqlite-vec extension, create `CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(episode_id INTEGER PRIMARY KEY, embedding float[N])` — N determined at runtime from first embedding
- `store/vec_search.rs`: `knn_search(conn, query_vec, limit) -> Vec<(NodeRef, f32)>` using MATCH query
- When storing embeddings: also INSERT into vec0 table
- `retrieval/vector.rs`: dispatch to vec0 search when feature enabled, fall back to brute-force otherwise
- Use `zerocopy` for efficient f32-to-bytes conversion

**Tests (TDD):**
1. `test_vec_search_returns_nearest` — store 3 embeddings, query, verify nearest first
2. `test_vec_search_respects_limit` — store 5, query with limit=2, verify 2 results
3. `test_vec_search_matches_brute_force` — verify vec0 returns same results as brute-force
4. `test_vec_table_created_on_init` — verify vec0 virtual table exists after schema init

---

### Task 6: Embedded Model Support (local-embeddings feature)

**Files:**
- Modify: `alaya/Cargo.toml` (add `fastembed` dep + feature flag)
- Create: `alaya/src/local_embeddings.rs` (LocalEmbeddingProvider)
- Modify: `alaya/src/lib.rs` (conditional module + re-export)
- Test: feature-gated tests

**Context:** Currently embeddings require an external provider. `fastembed` provides local ONNX-based embedding models.

**Implementation:**
- Feature flag: `local-embeddings`, depends on `fastembed = "5"`
- `LocalEmbeddingProvider` struct wrapping `fastembed::TextEmbedding`
- Implements `EmbeddingProvider` trait: `embed(&self, text: &str) -> Result<Vec<f32>>`
- Constructor: `LocalEmbeddingProvider::new(model: Option<EmbeddingModel>) -> Result<Self>`
- Default model: `AllMiniLML6V2` (384 dimensions, fast, small download)
- `dimensions(&self) -> usize` helper method

**Tests (TDD):**
1. `test_local_embedding_produces_vector` — embed a string, verify non-empty Vec<f32>
2. `test_local_embedding_consistent` — embed same string twice, verify identical vectors
3. `test_local_embedding_dimensions` — verify output matches expected model dimensions
4. `test_local_embedding_different_texts_differ` — embed "cat" and "quantum physics", verify different vectors

---

### Task 7: Webhook/Callback Hooks

**Files:**
- Create: `alaya/src/hooks.rs`
- Modify: `alaya/src/lib.rs` (add module, re-export trait, add `set_hooks` method)
- Modify: `alaya/src/store/episodic.rs` (call hook after store)
- Modify: `alaya/src/lifecycle/consolidation.rs` (call hook after consolidation)
- Modify: `alaya/src/lifecycle/transformation.rs` (call hook on category formation)
- Test: `alaya/src/hooks.rs` (unit tests with mock hook)

**Implementation:**
- `MemoryHooks` trait with default no-op methods:
  - `on_episode_stored(&self, id: EpisodeId)`
  - `on_consolidated(&self, report: &ConsolidationReport)`
  - `on_preference_crystallized(&self, pref: &Preference)`
  - `on_category_formed(&self, cat: &Category)`
  - `on_forgotten(&self, report: &ForgettingReport)`
- `NoOpHooks` struct implementing the trait (all defaults)
- `Alaya` stores `Option<Box<dyn MemoryHooks>>`, calls hooks at appropriate points
- Thread-safety: `MemoryHooks: Send + Sync` (or not, since Alaya is !Send anyway)

**Tests (TDD):**
1. `test_hook_called_on_episode_store` — set mock hook, store episode, verify hook called with correct ID
2. `test_hook_called_on_consolidation` — consolidate, verify hook called with report
3. `test_no_hook_no_panic` — verify operations work without hooks set
4. `test_hook_receives_crystallized_preference` — trigger preference crystallization, verify hook

---

### Task 8: Memory Visualization MCP Tool

**Files:**
- Create: `alaya/src/mcp/visualization.rs`
- Modify: `alaya/src/mcp/mod.rs` (register new tool)
- Test: `alaya/src/mcp/visualization.rs` (unit tests)

**Context:** MCP tools are registered in `mod.rs` via match arms. Need a `visualize` tool that returns a Mermaid diagram.

**Implementation:**
- `visualize` MCP tool with parameters: `format` (mermaid|dot, default mermaid), `max_nodes` (default 50), `min_weight` (default 0.1)
- Query: episodes (recent N), semantic nodes, categories, links above min_weight
- Generate Mermaid flowchart: nodes as boxes, links as arrows with weight labels
- Color-code by type: episodes=blue, semantic=green, preferences=orange, categories=purple

**Tests (TDD):**
1. `test_visualize_empty_db` — verify returns valid empty Mermaid diagram
2. `test_visualize_with_data` — store episodes + links, verify Mermaid contains node IDs and edges
3. `test_visualize_respects_max_nodes` — store 100 nodes, max_nodes=10, verify output limited
4. `test_visualize_min_weight_filter` — create weak and strong links, verify only strong links shown

---

### Task 9: Benchmark Harness (criterion)

**Files:**
- Create: `alaya/benches/memory_bench.rs`
- Modify: `alaya/Cargo.toml` (add criterion dev-dependency + [[bench]] section)
- No feature flag needed (dev-only)

**Implementation:**
- Benchmarks using criterion:
  - `bench_store_episode` — measure episode storage throughput
  - `bench_query_bm25` — BM25-only query on 1000 episodes
  - `bench_query_hybrid` — BM25 + vector query on 1000 episodes
  - `bench_consolidate` — consolidation on 100 episodes
  - `bench_transform` — transformation on 500 nodes
  - `bench_cosine_similarity` — raw cosine computation on 384-dim vectors
- Setup function that pre-populates the DB with realistic data

**Tests:** Benchmarks aren't TDD per se, but we verify they compile and run.

---

### Task 10: Conflict Resolution Engine

**Files:**
- Create: `alaya/src/lifecycle/conflict_resolution.rs`
- Modify: `alaya/src/lifecycle/mod.rs` (add module)
- Modify: `alaya/src/managers/lifecycle.rs` (add `resolve_conflicts` method)
- Modify: `alaya/src/types.rs` (add `ConflictResolutionReport`, extend `ConflictStrategy`)
- Test: inline + integration tests

**Context:** Current conflict handling detects contradictions via `detect_contradiction` provider method and stores them in the `conflicts` table. Resolution is manual (set winner_id). Need an automated resolution engine.

**Implementation:**
- `ConflictResolutionReport { resolved: u32, strategy_used: ConflictStrategy }`
- `resolve_conflicts(conn, strategy) -> Result<ConflictResolutionReport>`
- Strategies:
  - `Recency` — newer node (higher `last_corroborated`) wins
  - `Confidence` — higher confidence node wins
  - `Corroboration` — higher `corroboration_count` wins (new variant)
  - `Manual` — skip automatic resolution
- Resolution: set `winner_id`, mark loser as `superseded_by` winner, update status to `resolved`
- Add `Corroboration` variant to `ConflictStrategy` enum

**Tests (TDD):**
1. `test_resolve_by_recency` — create conflict where node B is newer, resolve with Recency, verify B wins
2. `test_resolve_by_confidence` — create conflict where node A has higher confidence, resolve with Confidence, verify A wins
3. `test_resolve_by_corroboration` — node with more corroborations wins
4. `test_manual_strategy_skips` — Manual strategy resolves nothing
5. `test_resolution_sets_superseded_by` — verify loser node gets superseded_by = winner.id
6. `test_resolution_updates_conflict_status` — verify conflict status changes to Resolved
