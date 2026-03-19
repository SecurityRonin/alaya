# Async API, SQLCipher & Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add async actor-pattern API, SQLCipher encryption, and Python bindings to Alaya.

**Architecture:** Actor pattern wraps sync `AlayaStore` in a dedicated thread with channel-based messaging. SQLCipher swaps SQLite bundling via feature flag. Python bindings (PyO3/maturin) expose a full API mirror as a separate workspace crate.

**Tech Stack:** tokio (sync channels), rusqlite (bundled-sqlcipher-vendored), PyO3 0.24, maturin, abi3-py39

**Spec:** `docs/superpowers/specs/2026-03-18-async-sqlcipher-python-design.md`

---

## File Structure

### Phase 1: Async API
- Create: `src/async_store.rs` — `AsyncAlayaStore`, `Request` enum, actor loop
- Modify: `src/lib.rs` — add `pub mod async_store` behind feature gate
- Modify: `src/error.rs` — add `ActorDead` variant
- Modify: `Cargo.toml` — add `async` feature flag with tokio sync

### Phase 2: SQLCipher
- Modify: `Cargo.toml` — restructure features (bundled-sqlite default, sqlcipher)
- Modify: `src/lib.rs` — add `open_encrypted()`, `rekey()` behind cfg
- Modify: `src/async_store.rs` — add encrypted variants behind cfg

### Phase 3: Workspace Migration
- Create: root `Cargo.toml` (workspace)
- Move: current `Cargo.toml` → `alaya/Cargo.toml`
- Move: `src/`, `tests/`, `tarpaulin.toml` → `alaya/`
- Modify: `.github/workflows/ci.yml` — workspace-aware commands
- Modify: `.github/workflows/release.yml` — workspace-aware commands

### Phase 4: Python Bindings
- Create: `alaya-py/Cargo.toml`
- Create: `alaya-py/pyproject.toml`
- Create: `alaya-py/src/lib.rs` — PyO3 module, `Alaya` class
- Create: `alaya-py/src/types.rs` — Python type wrappers
- Create: `alaya-py/src/provider.rs` — `PyConsolidationProvider` bridge
- Create: `alaya-py/alaya.pyi` — type stubs
- Create: `alaya-py/tests/test_basic.py`
- Create: `alaya-py/tests/test_lifecycle.py`
- Create: `alaya-py/tests/test_provider.py`

### Phase 5: CI
- Modify: `.github/workflows/ci.yml` — async + sqlcipher test jobs
- Modify: `.github/workflows/release.yml` — maturin wheel builds, PyPI publish

---

## Phase 1: Async API

### Task 1: Add `ActorDead` error variant

**Files:**
- Modify: `src/error.rs:5-20`

- [ ] **Step 1: Write failing test**

Add to `src/error.rs` test module:

```rust
#[test]
fn test_display_actor_dead() {
    let e = AlayaError::ActorDead;
    assert_eq!(e.to_string(), "actor dead: message channel closed");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib test_display_actor_dead`
Expected: FAIL — `ActorDead` variant doesn't exist

- [ ] **Step 3: Implement**

Add variant to `AlayaError` enum in `src/error.rs`:

```rust
#[error("actor dead: message channel closed")]
ActorDead,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib test_display_actor_dead`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/error.rs
git commit -m "feat: add ActorDead error variant for async API"
```

---

### Task 2: Add `async` feature flag to Cargo.toml

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add the feature flag**

Add to `[features]` section:

```toml
async = ["dep:tokio-async"]
```

Add to `[dependencies]`:

```toml
# Async actor pattern (optional, for AsyncAlayaStore)
tokio-async = { package = "tokio", version = "1", features = ["sync"], optional = true }
```

Note: We use `tokio-async` as the dep name to avoid conflict with the existing `tokio` dep (used by `mcp` feature with `features = ["full"]`). Cargo unifies features when both are enabled.

Wait — actually, Cargo allows the same package as a dependency only once. The existing `tokio` dep is optional with `features = ["full"]`. We need to handle this differently. The correct approach:

The existing line is:
```toml
tokio = { version = "1", features = ["full"], optional = true }
```

Change the `async` feature to also activate the existing tokio dep:
```toml
async = ["dep:tokio"]
```

But this creates an issue — `mcp` already activates `dep:tokio` with `["full"]`. When `async` is enabled without `mcp`, we only need `sync`. However, Cargo unifies features, so if the consumer enables both, they get `full`. If they only enable `async`, they need at minimum `sync`. The `features = ["full"]` in the dep line is always applied when tokio is activated.

Simplest correct approach: keep the existing tokio dep as-is and just add `async` as another activator:

```toml
[features]
async = ["dep:tokio"]
mcp = ["dep:rmcp", "dep:tokio", "dep:schemars", "dep:anyhow"]
```

This means enabling `async` alone pulls in tokio with `["full"]`, which is slightly more than needed but harmless and avoids dep management complexity.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check --features async`
Expected: Compiles (no async_store module yet, just the feature flag)

- [ ] **Step 3: Verify existing features still work**

Run: `cargo check --features "mcp llm"`
Expected: Compiles without issues

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml
git commit -m "feat: add async feature flag for AsyncAlayaStore"
```

---

### Task 3: Create `async_store.rs` — Request enum and actor loop

**Files:**
- Create: `src/async_store.rs`
- Modify: `src/lib.rs` — add conditional module

- [ ] **Step 1: Write failing test for actor open/shutdown**

Create `src/async_store.rs` with the test module first:

```rust
#[cfg(feature = "async")]
use crate::error::{AlayaError, Result};
#[cfg(feature = "async")]
use crate::types::*;
#[cfg(feature = "async")]
use crate::AlayaStore;
#[cfg(feature = "async")]
use crate::provider::{ConsolidationProvider, EmbeddingProvider, ExtractionProvider};

#[cfg(feature = "async")]
use std::path::Path;
#[cfg(feature = "async")]
use std::thread::JoinHandle;
#[cfg(feature = "async")]
use tokio::sync::{mpsc, oneshot};

// Placeholder — will be filled in step 3
#[cfg(feature = "async")]
pub struct AsyncAlayaStore;

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_open_in_memory_and_close() {
        let store = AsyncAlayaStore::open_in_memory().await.unwrap();
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_store_and_query() {
        let store = AsyncAlayaStore::open_in_memory().await.unwrap();
        let eid = store.store_episode(NewEpisode {
            content: "Rust has zero-cost abstractions".into(),
            role: Role::User,
            session_id: "s1".into(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        }).await.unwrap();
        assert_eq!(eid.0, 1);

        let results = store.query(Query::simple("Rust")).await.unwrap();
        assert!(!results.is_empty());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_status() {
        let store = AsyncAlayaStore::open_in_memory().await.unwrap();
        let status = store.status().await.unwrap();
        assert_eq!(status.episode_count, 0);
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_dream_without_interaction() {
        let store = AsyncAlayaStore::open_in_memory().await.unwrap();
        let report = store.dream(None).await.unwrap();
        assert_eq!(report.consolidation.episodes_processed, 0);
        assert!(report.perfuming.is_none());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_dead_after_close() {
        let store = AsyncAlayaStore::open_in_memory().await.unwrap();
        store.close().await.unwrap();
        let result = store.status().await;
        assert!(matches!(result, Err(AlayaError::ActorDead)));
    }
}
```

Add to `src/lib.rs` after the `pub(crate) mod types;` line:

```rust
#[cfg(feature = "async")]
pub mod async_store;
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features async --lib test_open_in_memory_and_close`
Expected: FAIL — `AsyncAlayaStore` is a unit struct with no methods

- [ ] **Step 3: Implement the actor pattern**

Replace the placeholder in `src/async_store.rs` with the full implementation:

```rust
#[cfg(feature = "async")]
use crate::error::{AlayaError, Result};
#[cfg(feature = "async")]
use crate::types::*;
#[cfg(feature = "async")]
use crate::AlayaStore;
#[cfg(feature = "async")]
use crate::provider::{ConsolidationProvider, EmbeddingProvider, ExtractionProvider, NoOpProvider};

#[cfg(feature = "async")]
use std::path::Path;
#[cfg(feature = "async")]
use std::thread::JoinHandle;
#[cfg(feature = "async")]
use tokio::sync::{mpsc, oneshot};

#[cfg(feature = "async")]
type Reply<T> = oneshot::Sender<Result<T>>;

#[cfg(feature = "async")]
enum Request {
    // Core
    StoreEpisode { episode: NewEpisode, reply: Reply<EpisodeId> },
    Query { query: Query, reply: Reply<Vec<ScoredMemory>> },
    Status { reply: Reply<MemoryStatus> },

    // Lifecycle
    Consolidate { reply: Reply<ConsolidationReport> },
    Learn { nodes: Vec<NewSemanticNode>, reply: Reply<ConsolidationReport> },
    AutoConsolidate { reply: Reply<ConsolidationReport> },
    Perfume { interaction: Interaction, reply: Reply<PerfumingReport> },
    Transform { reply: Reply<TransformationReport> },
    Forget { reply: Reply<ForgettingReport> },
    Dream { interaction: Option<Interaction>, reply: Reply<DreamReport> },

    // Query
    Preferences { domain: Option<String>, reply: Reply<Vec<Preference>> },
    Knowledge { filter: Option<KnowledgeFilter>, reply: Reply<Vec<SemanticNode>> },
    Categories { min_stability: Option<f32>, reply: Reply<Vec<Category>> },
    Subcategories { parent_id: CategoryId, reply: Reply<Vec<Category>> },
    NodeCategory { node_id: NodeId, reply: Reply<Option<Category>> },
    Neighbors { node: NodeRef, depth: u32, reply: Reply<Vec<(NodeRef, f32)>> },
    StrongestLink { reply: Reply<Option<(NodeRef, NodeRef, f32)>> },
    NodeContent { node: NodeRef, reply: Reply<Option<String>> },
    KnowledgeBreakdown { reply: Reply<std::collections::HashMap<SemanticType, u64>> },
    EpisodesBySession { session_id: String, reply: Reply<Vec<Episode>> },
    UnconsolidatedEpisodes { limit: u32, reply: Reply<Vec<Episode>> },

    // Admin
    Purge { filter: PurgeFilter, reply: Reply<PurgeReport> },

    // Provider management
    SetConsolidationProvider { provider: Box<dyn ConsolidationProvider + Send> },
    SetEmbeddingProvider { provider: Box<dyn EmbeddingProvider + Send> },
    SetExtractionProvider { provider: Box<dyn ExtractionProvider + Send> },

    // Shutdown
    Shutdown,
}

#[cfg(feature = "async")]
pub struct AsyncAlayaStore {
    tx: mpsc::Sender<Request>,
    handle: Option<JoinHandle<()>>,
}

#[cfg(feature = "async")]
impl AsyncAlayaStore {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (tx, rx) = mpsc::channel(64);
        let handle = std::thread::spawn(move || {
            let store = AlayaStore::open(&path).expect("failed to open store");
            Self::run_actor(store, rx);
        });
        Ok(Self { tx, handle: Some(handle) })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let (tx, rx) = mpsc::channel(64);
        let handle = std::thread::spawn(move || {
            let store = AlayaStore::open_in_memory().expect("failed to open in-memory store");
            Self::run_actor(store, rx);
        });
        Ok(Self { tx, handle: Some(handle) })
    }

    pub async fn close(mut self) -> Result<()> {
        let _ = self.tx.send(Request::Shutdown).await;
        if let Some(handle) = self.handle.take() {
            tokio::task::spawn_blocking(move || {
                let _ = handle.join();
            }).await.map_err(|_| AlayaError::ActorDead)?;
        }
        Ok(())
    }

    fn run_actor(mut store: AlayaStore, mut rx: mpsc::Receiver<Request>) {
        // Provider stored in actor thread — default NoOp
        let mut provider: Box<dyn ConsolidationProvider + Send> = Box::new(NoOpProvider);

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
                    let _ = reply.send(store.consolidate(provider.as_ref()));
                }
                Request::Learn { nodes, reply } => {
                    let _ = reply.send(store.learn(nodes));
                }
                Request::AutoConsolidate { reply } => {
                    let _ = reply.send(store.auto_consolidate());
                }
                Request::Perfume { interaction, reply } => {
                    let _ = reply.send(store.perfume(&interaction, provider.as_ref()));
                }
                Request::Transform { reply } => {
                    let _ = reply.send(store.transform());
                }
                Request::Forget { reply } => {
                    let _ = reply.send(store.forget());
                }
                Request::Dream { interaction, reply } => {
                    let _ = reply.send(store.dream(provider.as_ref(), interaction.as_ref()));
                }
                Request::Preferences { domain, reply } => {
                    let _ = reply.send(store.preferences(domain.as_deref()));
                }
                Request::Knowledge { filter, reply } => {
                    let _ = reply.send(store.knowledge(filter));
                }
                Request::Categories { min_stability, reply } => {
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
                Request::SetConsolidationProvider { provider: p } => {
                    provider = p;
                }
                Request::SetEmbeddingProvider { provider: p } => {
                    store.set_embedding_provider(p);
                }
                Request::SetExtractionProvider { provider: p } => {
                    store.set_extraction_provider(p);
                }
                Request::Shutdown => break,
            }
        }
    }

    // -- Helper: send request and await reply --

    async fn send<T>(&self, make_req: impl FnOnce(Reply<T>) -> Request) -> Result<T> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(make_req(tx)).await.map_err(|_| AlayaError::ActorDead)?;
        rx.await.map_err(|_| AlayaError::ActorDead)?
    }

    // -- Public async methods --

    pub async fn store_episode(&self, episode: NewEpisode) -> Result<EpisodeId> {
        self.send(|reply| Request::StoreEpisode { episode, reply }).await
    }

    pub async fn query(&self, query: Query) -> Result<Vec<ScoredMemory>> {
        self.send(|reply| Request::Query { query, reply }).await
    }

    pub async fn status(&self) -> Result<MemoryStatus> {
        self.send(|reply| Request::Status { reply }).await
    }

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
        self.send(|reply| Request::Perfume { interaction, reply }).await
    }

    pub async fn transform(&self) -> Result<TransformationReport> {
        self.send(|reply| Request::Transform { reply }).await
    }

    pub async fn forget(&self) -> Result<ForgettingReport> {
        self.send(|reply| Request::Forget { reply }).await
    }

    pub async fn dream(&self, interaction: Option<Interaction>) -> Result<DreamReport> {
        self.send(|reply| Request::Dream { interaction, reply }).await
    }

    pub async fn preferences(&self, domain: Option<String>) -> Result<Vec<Preference>> {
        self.send(|reply| Request::Preferences { domain, reply }).await
    }

    pub async fn knowledge(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>> {
        self.send(|reply| Request::Knowledge { filter, reply }).await
    }

    pub async fn categories(&self, min_stability: Option<f32>) -> Result<Vec<Category>> {
        self.send(|reply| Request::Categories { min_stability, reply }).await
    }

    pub async fn subcategories(&self, parent_id: CategoryId) -> Result<Vec<Category>> {
        self.send(|reply| Request::Subcategories { parent_id, reply }).await
    }

    pub async fn node_category(&self, node_id: NodeId) -> Result<Option<Category>> {
        self.send(|reply| Request::NodeCategory { node_id, reply }).await
    }

    pub async fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>> {
        self.send(|reply| Request::Neighbors { node, depth, reply }).await
    }

    pub async fn strongest_link(&self) -> Result<Option<(NodeRef, NodeRef, f32)>> {
        self.send(|reply| Request::StrongestLink { reply }).await
    }

    pub async fn node_content(&self, node: NodeRef) -> Result<Option<String>> {
        self.send(|reply| Request::NodeContent { node, reply }).await
    }

    pub async fn knowledge_breakdown(&self) -> Result<std::collections::HashMap<SemanticType, u64>> {
        self.send(|reply| Request::KnowledgeBreakdown { reply }).await
    }

    pub async fn episodes_by_session(&self, session_id: String) -> Result<Vec<Episode>> {
        self.send(|reply| Request::EpisodesBySession { session_id, reply }).await
    }

    pub async fn unconsolidated_episodes(&self, limit: u32) -> Result<Vec<Episode>> {
        self.send(|reply| Request::UnconsolidatedEpisodes { limit, reply }).await
    }

    pub async fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport> {
        self.send(|reply| Request::Purge { filter, reply }).await
    }

    pub async fn set_consolidation_provider(&self, provider: Box<dyn ConsolidationProvider + Send>) {
        let _ = self.tx.send(Request::SetConsolidationProvider { provider }).await;
    }

    pub async fn set_embedding_provider(&self, provider: Box<dyn EmbeddingProvider + Send>) {
        let _ = self.tx.send(Request::SetEmbeddingProvider { provider }).await;
    }

    pub async fn set_extraction_provider(&self, provider: Box<dyn ExtractionProvider + Send>) {
        let _ = self.tx.send(Request::SetExtractionProvider { provider }).await;
    }
}

#[cfg(feature = "async")]
impl Drop for AsyncAlayaStore {
    fn drop(&mut self) {
        let _ = self.tx.try_send(Request::Shutdown);
        // Best-effort: do not join here to avoid blocking async runtime
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --features async --lib async_store`
Expected: All 5 tests pass

- [ ] **Step 5: Run full test suite**

Run: `cargo test --features "mcp llm async"`
Expected: All existing tests + 5 new tests pass

- [ ] **Step 6: Run clippy and fmt**

Run: `cargo clippy --all-targets --features "mcp llm async" -- -D warnings && cargo fmt -- --check`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add src/async_store.rs src/lib.rs
git commit -m "feat: add AsyncAlayaStore with actor pattern behind async feature"
```

---

### Task 4: Add concurrent access and edge case tests

**Files:**
- Modify: `src/async_store.rs` — add tests to test module

- [ ] **Step 1: Write additional tests**

Add to `src/async_store.rs` test module:

```rust
#[tokio::test]
async fn test_concurrent_stores() {
    let store = std::sync::Arc::new(AsyncAlayaStore::open_in_memory().await.unwrap());
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
            }).await.unwrap();
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
    // Verify Drop doesn't panic or block
    let store = AsyncAlayaStore::open_in_memory().await.unwrap();
    drop(store);
    // If we reach here, Drop worked
}

#[tokio::test]
async fn test_lifecycle_via_async() {
    let store = AsyncAlayaStore::open_in_memory().await.unwrap();
    let tr = store.transform().await.unwrap();
    assert_eq!(tr.duplicates_merged, 0);
    let fr = store.forget().await.unwrap();
    assert_eq!(fr.nodes_decayed, 0);
    store.close().await.unwrap();
}
```

Note: `AsyncAlayaStore` needs to derive `Clone` (by wrapping the sender in an `Arc`), OR the test should not use `Arc`. Since `mpsc::Sender` is already `Clone`, we can make `AsyncAlayaStore` cloneable by making `handle` an `Arc<Mutex<Option<JoinHandle>>>`. For simplicity, the concurrent test can wrap in `Arc` manually only if `AsyncAlayaStore` is `Send + Sync`. Since `mpsc::Sender` is `Send + Sync` and `Option<JoinHandle>` is `Send`, the struct should be `Send`. For the concurrent test, the simplest approach is to have the test create separate episode contents without needing to clone the store — instead, use a shared reference via `Arc`. This requires `AsyncAlayaStore` to be `Sync`, which it is if all fields are `Sync`. `mpsc::Sender` is `Send + Sync`. `Option<JoinHandle>` is `Send` but not `Sync`. Fix: wrap handle in `Mutex`.

Adjust the struct definition:

```rust
pub struct AsyncAlayaStore {
    tx: mpsc::Sender<Request>,
    handle: std::sync::Mutex<Option<JoinHandle<()>>>,
}
```

Update `close()` and `Drop` to lock the mutex.

- [ ] **Step 2: Run tests**

Run: `cargo test --features async --lib async_store`
Expected: All 8 tests pass

- [ ] **Step 3: Commit**

```bash
git add src/async_store.rs
git commit -m "test: add concurrent access and edge case tests for AsyncAlayaStore"
```

---

## Phase 2: SQLCipher

### Task 5: Restructure feature flags

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Restructure features**

Change the `[dependencies]` and `[features]` sections:

```toml
[dependencies]
rusqlite = { version = "0.32", features = ["modern_sqlite"] }

[features]
default = ["bundled-sqlite"]
bundled-sqlite = ["rusqlite/bundled"]
sqlcipher = ["rusqlite/bundled-sqlcipher-vendored"]
mcp = ["dep:rmcp", "dep:tokio", "dep:schemars", "dep:anyhow"]
llm = ["dep:ureq"]
tracing = ["dep:tracing", "dep:tracing-subscriber"]
async = ["dep:tokio"]
```

Key change: `rusqlite` no longer has `bundled` in its base features — it's activated by the default `bundled-sqlite` feature.

- [ ] **Step 2: Verify default features compile**

Run: `cargo check`
Expected: Compiles (bundled-sqlite is default)

- [ ] **Step 3: Verify sqlcipher compiles**

Run: `cargo check --features sqlcipher --no-default-features`
Expected: Compiles (bundled-sqlcipher-vendored activated)

- [ ] **Step 4: Verify existing feature combos**

Run: `cargo test --features "mcp llm"` and `cargo test`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml
git commit -m "refactor: restructure features for bundled-sqlite default and sqlcipher option"
```

---

### Task 6: Add `open_encrypted()` and `rekey()`

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add to `src/lib.rs` test module (these tests only run with sqlcipher feature):

```rust
#[cfg(feature = "sqlcipher")]
#[test]
fn test_open_encrypted_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encrypted.db");

    // Create encrypted DB
    let store = AlayaStore::open_encrypted(&path, "test-key-123").unwrap();
    store.store_episode(&NewEpisode {
        content: "secret data".into(),
        role: Role::User,
        session_id: "s1".into(),
        timestamp: 1000,
        context: EpisodeContext::default(),
        embedding: None,
    }).unwrap();
    drop(store);

    // Reopen with correct key
    let store2 = AlayaStore::open_encrypted(&path, "test-key-123").unwrap();
    assert_eq!(store2.status().unwrap().episode_count, 1);
}

#[cfg(feature = "sqlcipher")]
#[test]
fn test_open_encrypted_wrong_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encrypted.db");

    let store = AlayaStore::open_encrypted(&path, "correct-key").unwrap();
    store.store_episode(&NewEpisode {
        content: "secret".into(),
        role: Role::User,
        session_id: "s1".into(),
        timestamp: 1000,
        context: EpisodeContext::default(),
        embedding: None,
    }).unwrap();
    drop(store);

    // Wrong key should fail
    let result = AlayaStore::open_encrypted(&path, "wrong-key");
    assert!(result.is_err());
}

#[cfg(feature = "sqlcipher")]
#[test]
fn test_rekey() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rekey.db");

    let store = AlayaStore::open_encrypted(&path, "old-key").unwrap();
    store.store_episode(&NewEpisode {
        content: "rekey test".into(),
        role: Role::User,
        session_id: "s1".into(),
        timestamp: 1000,
        context: EpisodeContext::default(),
        embedding: None,
    }).unwrap();
    store.rekey("new-key").unwrap();
    drop(store);

    // Old key should fail
    assert!(AlayaStore::open_encrypted(&path, "old-key").is_err());

    // New key should work
    let store2 = AlayaStore::open_encrypted(&path, "new-key").unwrap();
    assert_eq!(store2.status().unwrap().episode_count, 1);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --features sqlcipher --no-default-features --lib test_open_encrypted`
Expected: FAIL — methods don't exist

- [ ] **Step 3: Implement**

Add to `src/lib.rs` inside `impl AlayaStore`:

```rust
#[cfg(feature = "sqlcipher")]
pub fn open_encrypted(path: impl AsRef<Path>, key: &str) -> Result<Self> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "key", key)?;
    // Verify the key works by reading from the database
    conn.execute_batch("SELECT count(*) FROM sqlite_master")?;
    schema::initialize(&conn)?;
    Ok(Self {
        conn,
        embedding_provider: None,
        extraction_provider: None,
    })
}

#[cfg(feature = "sqlcipher")]
pub fn rekey(&self, new_key: &str) -> Result<()> {
    self.conn.pragma_update(None, "rekey", new_key)?;
    Ok(())
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --features sqlcipher --no-default-features --lib test_open_encrypted test_rekey`
Expected: All 3 tests pass

- [ ] **Step 5: Verify default features still work**

Run: `cargo test`
Expected: All tests pass (sqlcipher code is cfg'd out)

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs
git commit -m "feat: add open_encrypted() and rekey() behind sqlcipher feature"
```

---

### Task 7: Add async encrypted variants

**Files:**
- Modify: `src/async_store.rs`

- [ ] **Step 1: Write failing tests**

Add to `src/async_store.rs` test module:

```rust
#[cfg(feature = "sqlcipher")]
#[tokio::test]
async fn test_async_open_encrypted_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("async_enc.db");

    let store = AsyncAlayaStore::open_encrypted(&path, "async-key").await.unwrap();
    store.store_episode(NewEpisode {
        content: "async secret".into(),
        role: Role::User,
        session_id: "s1".into(),
        timestamp: 1000,
        context: EpisodeContext::default(),
        embedding: None,
    }).await.unwrap();
    store.close().await.unwrap();

    let store2 = AsyncAlayaStore::open_encrypted(&path, "async-key").await.unwrap();
    let status = store2.status().await.unwrap();
    assert_eq!(status.episode_count, 1);
    store2.close().await.unwrap();
}
```

- [ ] **Step 2: Implement**

Add to `impl AsyncAlayaStore`:

```rust
#[cfg(feature = "sqlcipher")]
pub async fn open_encrypted(path: impl AsRef<Path>, key: &str) -> Result<Self> {
    let path = path.as_ref().to_path_buf();
    let key = key.to_string();
    let (tx, rx) = mpsc::channel(64);
    let handle = std::thread::spawn(move || {
        let store = AlayaStore::open_encrypted(&path, &key).expect("failed to open encrypted store");
        Self::run_actor(store, rx);
    });
    Ok(Self { tx, handle: std::sync::Mutex::new(Some(handle)) })
}

#[cfg(feature = "sqlcipher")]
pub async fn rekey(&self, new_key: &str) -> Result<()> {
    let (reply_tx, reply_rx) = oneshot::channel();
    self.tx.send(Request::Rekey {
        new_key: new_key.to_string(),
        reply: reply_tx,
    }).await.map_err(|_| AlayaError::ActorDead)?;
    reply_rx.await.map_err(|_| AlayaError::ActorDead)?
}
```

Also add `Rekey` variant to `Request`:
```rust
#[cfg(feature = "sqlcipher")]
Rekey { new_key: String, reply: Reply<()> },
```

And handle it in `run_actor`:
```rust
#[cfg(feature = "sqlcipher")]
Request::Rekey { new_key, reply } => {
    let _ = reply.send(store.rekey(&new_key));
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --features "async sqlcipher" --no-default-features --lib test_async_open_encrypted`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/async_store.rs
git commit -m "feat: add async encrypted open/rekey behind sqlcipher+async features"
```

---

## Phase 3: Workspace Migration

### Task 8: Migrate to Cargo workspace

**Files:**
- Create: root `Cargo.toml` (workspace)
- Move: all source files into `alaya/` subdirectory

- [ ] **Step 1: Create workspace structure**

```bash
# Create the alaya subdirectory
mkdir -p alaya

# Move source files
git mv src alaya/
git mv tests alaya/
git mv Cargo.toml alaya/
git mv tarpaulin.toml alaya/

# Create workspace root Cargo.toml
cat > Cargo.toml << 'WORKSPACE'
[workspace]
members = ["alaya"]
resolver = "2"
WORKSPACE
```

- [ ] **Step 2: Verify builds**

Run: `cargo test -p alaya --features "mcp llm async"`
Expected: All tests pass

Run: `cargo clippy -p alaya --all-targets --features "mcp llm async" -- -D warnings`
Expected: Clean

- [ ] **Step 3: Update CI workflows**

Update `.github/workflows/ci.yml` to use `cargo test -p alaya` and `cargo clippy -p alaya`. The workspace root `cargo test` also works but being explicit is clearer.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: migrate to Cargo workspace for multi-crate support"
```

---

## Phase 4: Python Bindings

### Task 9: Create `alaya-py` crate skeleton

**Files:**
- Create: `alaya-py/Cargo.toml`
- Create: `alaya-py/pyproject.toml`
- Create: `alaya-py/src/lib.rs` (minimal module)
- Modify: root `Cargo.toml` — add `alaya-py` to workspace

- [ ] **Step 1: Create crate files**

`alaya-py/Cargo.toml`:
```toml
[package]
name = "alaya-py"
version = "0.3.0"
edition = "2021"
publish = false

[lib]
name = "alaya"
crate-type = ["cdylib"]

[features]
default = ["encryption"]
encryption = ["alaya/sqlcipher"]

[dependencies]
alaya = { path = "../alaya", default-features = false }
pyo3 = { version = "0.24", features = ["abi3-py39", "extension-module"] }
```

`alaya-py/pyproject.toml`:
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "alaya"
requires-python = ">=3.9"
description = "A memory engine for conversational AI agents"
license = { text = "MIT" }
keywords = ["memory", "ai-agent", "knowledge-graph", "sqlite"]

[tool.maturin]
features = ["pyo3/abi3-py39"]
```

`alaya-py/src/lib.rs`:
```rust
use pyo3::prelude::*;

#[pymodule]
fn alaya(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Alaya>()?;
    Ok(())
}

#[pyclass]
struct Alaya {
    store: ::alaya::AlayaStore,
}

#[pymethods]
impl Alaya {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = ::alaya::AlayaStore::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { store })
    }

    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        let store = ::alaya::AlayaStore::open_in_memory()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { store })
    }
}
```

Update root `Cargo.toml`:
```toml
[workspace]
members = ["alaya", "alaya-py"]
resolver = "2"
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p alaya-py`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add alaya-py/ Cargo.toml
git commit -m "feat: add alaya-py crate skeleton with PyO3"
```

---

### Task 10: Implement Python type wrappers

**Files:**
- Create: `alaya-py/src/types.rs`
- Modify: `alaya-py/src/lib.rs` — add module, register types

- [ ] **Step 1: Create type wrappers**

`alaya-py/src/types.rs` — Python wrapper types for all Alaya types. Each Alaya type gets a PyO3 `#[pyclass]` wrapper with `#[getter]` methods for field access.

Key types to wrap: `Episode`, `ScoredMemory`, `SemanticNode`, `Preference`, `Impression`, `Category`, `MemoryStatus`, `DreamReport`, `ConsolidationReport`, `PerfumingReport`, `TransformationReport`, `ForgettingReport`, `PurgeReport`, `Query`, `NewEpisode`, `Interaction`, `Link`.

- [ ] **Step 2: Register types in module**

Add `mod types;` to `lib.rs` and register all classes with the module.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p alaya-py`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add alaya-py/src/types.rs alaya-py/src/lib.rs
git commit -m "feat: add Python type wrappers for all Alaya types"
```

---

### Task 11: Implement full Alaya Python class

**Files:**
- Modify: `alaya-py/src/lib.rs`

- [ ] **Step 1: Add all AlayaStore methods to the Alaya pyclass**

Mirror all 26 public methods. Key patterns:
- Methods taking `&str` params: accept Python `str`
- Methods returning `Vec<T>`: return Python `list`
- Methods returning `Option<T>`: return `T | None`
- Methods returning `Result<T>`: return `T`, raise on error
- `store_episode` takes keyword args
- `query` accepts `str` (simple) or `Query` object (advanced)
- Context manager: `__enter__` returns self, `__exit__` is no-op (SQLite handles cleanup)

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p alaya-py`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add alaya-py/src/lib.rs
git commit -m "feat: implement full Alaya Python class mirroring AlayaStore"
```

---

### Task 12: Implement Python provider bridge

**Files:**
- Create: `alaya-py/src/provider.rs`
- Modify: `alaya-py/src/lib.rs`

- [ ] **Step 1: Create PyConsolidationProvider**

```rust
use pyo3::prelude::*;
use alaya::{
    ConsolidationProvider, Episode, Interaction, NewSemanticNode, NewImpression, SemanticNode,
};
use alaya::Result;

pub struct PyConsolidationProvider {
    py_obj: PyObject,
}

impl PyConsolidationProvider {
    pub fn new(py_obj: PyObject) -> Self {
        Self { py_obj }
    }
}

impl ConsolidationProvider for PyConsolidationProvider {
    fn extract_knowledge(&self, episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        Python::with_gil(|py| {
            let py_episodes: Vec<PyObject> = episodes
                .iter()
                .map(|ep| crate::types::PyEpisode::from(ep.clone()).into_pyobject(py).unwrap().into_any().unbind())
                .collect();
            let result = self.py_obj
                .call_method1(py, "extract_knowledge", (py_episodes,))
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
            let py_list = result.downcast_bound::<pyo3::types::PyList>(py)
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
            let mut nodes = Vec::new();
            for item in py_list.iter() {
                let content: String = item.getattr("content")?.extract()
                    .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
                let source_episodes: Vec<i64> = item.getattr("source_episodes")?.extract()
                    .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
                nodes.push(NewSemanticNode {
                    content,
                    source_episodes: source_episodes.into_iter().map(EpisodeId).collect(),
                });
            }
            Ok(nodes)
        })
    }

    fn extract_impressions(&self, interaction: &Interaction) -> Result<Vec<NewImpression>> {
        Python::with_gil(|py| {
            let py_interaction = crate::types::PyInteraction::from(interaction.clone())
                .into_pyobject(py).unwrap().into_any().unbind();
            let result = self.py_obj
                .call_method1(py, "extract_impressions", (py_interaction,))
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
            let py_list = result.downcast_bound::<pyo3::types::PyList>(py)
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
            let mut impressions = Vec::new();
            for item in py_list.iter() {
                let content: String = item.getattr("content")?.extract()
                    .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
                impressions.push(NewImpression { content });
            }
            Ok(impressions)
        })
    }

    fn detect_contradiction(&self, a: &SemanticNode, b: &SemanticNode) -> Result<bool> {
        Python::with_gil(|py| {
            let py_a = crate::types::PySemanticNode::from(a.clone())
                .into_pyobject(py).unwrap().into_any().unbind();
            let py_b = crate::types::PySemanticNode::from(b.clone())
                .into_pyobject(py).unwrap().into_any().unbind();
            let result = self.py_obj
                .call_method1(py, "detect_contradiction", (py_a, py_b))
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))?;
            result.extract::<bool>(py)
                .map_err(|e| alaya::AlayaError::Provider(e.to_string()))
        })
    }
}
```

- [ ] **Step 2: Wire into Alaya class**

Add `set_consolidation_provider(provider)` method to the `Alaya` pyclass.

- [ ] **Step 3: Commit**

```bash
git add alaya-py/src/provider.rs alaya-py/src/lib.rs
git commit -m "feat: add Python-to-Rust provider bridge via PyConsolidationProvider"
```

---

### Task 13: Add Python type stubs and tests

**Files:**
- Create: `alaya-py/alaya.pyi`
- Create: `alaya-py/tests/test_basic.py`
- Create: `alaya-py/tests/test_lifecycle.py`
- Create: `alaya-py/tests/test_provider.py`

- [ ] **Step 1: Create type stubs**

`alaya-py/alaya.pyi` — type hints for IDE autocompletion:

```python
from typing import Optional

class Alaya:
    def __init__(self, path: str) -> None: ...
    @staticmethod
    def in_memory() -> "Alaya": ...
    @staticmethod
    def open_encrypted(path: str, key: str) -> "Alaya": ...
    def store_episode(self, content: str, role: str, session_id: str, timestamp: int, ...) -> int: ...
    def query(self, query: str | Query, max_results: int = 5) -> list[ScoredMemory]: ...
    def dream(self) -> DreamReport: ...
    def status(self) -> MemoryStatus: ...
    # ... (all methods)
```

- [ ] **Step 2: Create Python tests**

`alaya-py/tests/test_basic.py`:
```python
import alaya

def test_open_in_memory():
    store = alaya.Alaya.in_memory()
    status = store.status()
    assert status.episode_count == 0

def test_store_and_query():
    store = alaya.Alaya.in_memory()
    eid = store.store_episode(
        content="Rust has zero-cost abstractions",
        role="user",
        session_id="s1",
        timestamp=1000,
    )
    assert eid > 0
    results = store.query("Rust")
    assert len(results) > 0
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd alaya-py
pip install maturin
maturin develop
pytest tests/
```
Expected: All Python tests pass

- [ ] **Step 4: Commit**

```bash
git add alaya-py/alaya.pyi alaya-py/tests/
git commit -m "feat: add Python type stubs and test suite"
```

---

## Phase 5: CI

### Task 14: Add async and SQLCipher CI jobs

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add test jobs**

Add to the CI workflow:

```yaml
async-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo test -p alaya --features async

sqlcipher-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo test -p alaya --features sqlcipher --no-default-features
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add async and sqlcipher test jobs"
```

---

### Task 15: Add Python wheel builds and PyPI publishing

**Files:**
- Modify: `.github/workflows/release.yml`

- [ ] **Step 1: Add maturin wheel build job**

Add to release workflow:

```yaml
python-build:
  runs-on: ${{ matrix.os }}
  strategy:
    matrix:
      os: [ubuntu-latest, macos-14, macos-13, windows-latest]
  steps:
    - uses: actions/checkout@v4
    - uses: PyO3/maturin-action@v1
      with:
        working-directory: alaya-py
        args: --release --out dist
    - uses: actions/upload-artifact@v4
      with:
        name: wheels-${{ matrix.os }}
        path: alaya-py/dist/

python-publish:
  needs: python-build
  runs-on: ubuntu-latest
  permissions:
    id-token: write
  steps:
    - uses: actions/download-artifact@v4
      with:
        pattern: wheels-*
        merge-multiple: true
        path: dist/
    - uses: pypa/gh-action-pypi-publish@release/v1
      with:
        packages-dir: dist/
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add Python wheel builds and PyPI trusted publishing"
```

---

### Task 16: Final verification

- [ ] **Step 1: Run all feature combinations**

```bash
cargo test -p alaya
cargo test -p alaya --features mcp
cargo test -p alaya --features llm
cargo test -p alaya --features async
cargo test -p alaya --features "mcp llm async"
cargo test -p alaya --features sqlcipher --no-default-features
cargo test -p alaya --features "sqlcipher async" --no-default-features
```

- [ ] **Step 2: Run clippy across all combos**

```bash
cargo clippy -p alaya --all-targets --features "mcp llm async" -- -D warnings
cargo clippy -p alaya --all-targets --features "sqlcipher async" --no-default-features -- -D warnings
cargo clippy -p alaya-py --all-targets -- -D warnings
```

- [ ] **Step 3: Run Python tests**

```bash
cd alaya-py && maturin develop && pytest tests/ -v
```

- [ ] **Step 4: Verify formatting**

```bash
cargo fmt --all -- --check
```

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: final verification fixes"
```
