# P0 Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the Alaya v0.1.0 API surface for semver stability, transaction safety, input validation, encapsulation, and documentation before the first crates.io publish.

**Architecture:** Seven tasks addressing known gaps GAP-001 through GAP-005 plus dev-dependency and schema versioning foundations. Each task is independently committable. Tasks are ordered so each builds on the previous: dev-deps first (enables test infrastructure), schema versioning (foundation), `#[non_exhaustive]` (API contract), transactions (safety), validation (boundary), `pub(crate)` (encapsulation), doctests (documentation of the final hardened API).

**Tech Stack:** Rust 2021 edition, rusqlite 0.32 (bundled SQLite), tempfile (dev-dependency), `cargo test`, `cargo doc --test`

---

### Task 1: Add tempfile dev-dependency

**Files:**
- Modify: `Cargo.toml:18`

**Step 1: Add tempfile to dev-dependencies**

In `Cargo.toml`, change the empty `[dev-dependencies]` section:

```toml
[dev-dependencies]
tempfile = "3"
```

**Step 2: Write a test that uses tempfile to verify the dependency works**

Add to the bottom of `src/lib.rs`, inside the existing `#[cfg(test)] mod tests` block (after the `test_purge_all` test):

```rust
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
```

**Step 3: Run tests to verify**

Run: `cargo test test_open_persistent_db -- --nocapture`
Expected: PASS

**Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs
git commit -m "chore: add tempfile dev-dependency with persistent DB test"
```

---

### Task 2: Schema versioning with PRAGMA user_version

**Files:**
- Modify: `src/schema.rs:20-24` (init_db function)
- Test: `src/schema.rs` (tests module)

**Step 1: Write the failing test**

Add to `src/schema.rs` inside the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn test_schema_version_is_set() {
        let conn = open_memory_db().unwrap();
        let version: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(version, 1, "schema version should be 1 after init");
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_schema_version_is_set -- --nocapture`
Expected: FAIL with `assertion ... left: 0, right: 1`

**Step 3: Add PRAGMA user_version to init_db**

In `src/schema.rs`, inside `init_db`, after line 23 (`PRAGMA synchronous = NORMAL;`), add:

```rust
    conn.execute_batch("PRAGMA user_version = 1;")?;
```

So the PRAGMAs block becomes:

```rust
fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch("PRAGMA journal_mode = WAL;")?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    conn.execute_batch("PRAGMA synchronous = NORMAL;")?;
    conn.execute_batch("PRAGMA user_version = 1;")?;

    conn.execute_batch(
        "
        -- =================================================================
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_schema_version_is_set -- --nocapture`
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test`
Expected: All tests pass (including `test_idempotent_init`)

**Step 6: Commit**

```bash
git add src/schema.rs
git commit -m "feat: add schema versioning with PRAGMA user_version = 1"
```

---

### Task 3: Add #[non_exhaustive] to all public enums (GAP-001)

**Files:**
- Modify: `src/types.rs:27,66,93,123,330`
- Modify: `src/error.rs:3`

There are 6 public enums that need `#[non_exhaustive]`:

| Enum | File | Line |
|------|------|------|
| `NodeRef` | `src/types.rs` | 27 |
| `Role` | `src/types.rs` | 66 |
| `SemanticType` | `src/types.rs` | 93 |
| `LinkType` | `src/types.rs` | 123 |
| `PurgeFilter` | `src/types.rs` | 330 |
| `AlayaError` | `src/error.rs` | 4 |

**Step 1: Run the full test suite to establish baseline**

Run: `cargo test`
Expected: All tests pass (this is the baseline — `#[non_exhaustive]` has no effect within the defining crate)

**Step 2: Add #[non_exhaustive] to NodeRef**

In `src/types.rs`, add `#[non_exhaustive]` above line 27:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NodeRef {
```

**Step 3: Add #[non_exhaustive] to Role**

In `src/types.rs`, add above line 66:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
```

**Step 4: Add #[non_exhaustive] to SemanticType**

In `src/types.rs`, add above line 93:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum SemanticType {
```

**Step 5: Add #[non_exhaustive] to LinkType**

In `src/types.rs`, add above line 123:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum LinkType {
```

**Step 6: Add #[non_exhaustive] to PurgeFilter**

In `src/types.rs`, add above line 330:

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PurgeFilter {
```

**Step 7: Add #[non_exhaustive] to AlayaError**

In `src/error.rs`, add above line 4:

```rust
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AlayaError {
```

**Step 8: Run full test suite to verify no regressions**

Run: `cargo test`
Expected: All tests pass (within-crate match expressions are unaffected by `#[non_exhaustive]`)

**Step 9: Commit**

```bash
git add src/types.rs src/error.rs
git commit -m "feat: add #[non_exhaustive] to all public enums (GAP-001)

Prevents downstream exhaustive matching on NodeRef, Role, SemanticType,
LinkType, PurgeFilter, and AlayaError. Allows adding variants in minor
releases without breaking consumers."
```

---

### Task 4: BEGIN IMMEDIATE for write transactions (GAP-002)

**Files:**
- Modify: `src/schema.rs` (add `begin_immediate` helper)
- Modify: `src/lib.rs:50-73,152-173,192-219` (wrap write methods)
- Test: `src/schema.rs` (tests module)
- Test: `src/lib.rs` (tests module)

**Step 1: Write the failing test for begin_immediate helper**

Add to `src/schema.rs` inside the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn test_begin_immediate_transaction() {
        let conn = open_memory_db().unwrap();
        let tx = begin_immediate(&conn).unwrap();
        tx.execute(
            "INSERT INTO episodes (content, role, session_id, timestamp) VALUES (?1, ?2, ?3, ?4)",
            ("test", "user", "s1", &1000i64),
        )
        .unwrap();
        tx.commit().unwrap();

        let count: i64 = conn
            .query_row("SELECT count(*) FROM episodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_immediate_transaction_rollback_on_drop() {
        let conn = open_memory_db().unwrap();
        {
            let tx = begin_immediate(&conn).unwrap();
            tx.execute(
                "INSERT INTO episodes (content, role, session_id, timestamp) VALUES (?1, ?2, ?3, ?4)",
                ("test", "user", "s1", &1000i64),
            )
            .unwrap();
            // tx drops here without commit — should rollback
        }

        let count: i64 = conn
            .query_row("SELECT count(*) FROM episodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0, "uncommitted transaction should rollback on drop");
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_begin_immediate -- --nocapture`
Expected: FAIL — `begin_immediate` function does not exist

**Step 3: Implement begin_immediate**

Add to `src/schema.rs`, after the `open_memory_db` function (after line 18), before `init_db`:

```rust
/// Start a write transaction with IMMEDIATE locking.
/// This prevents SQLITE_BUSY errors under concurrent readers by acquiring
/// the write lock at BEGIN rather than at first write statement.
pub(crate) fn begin_immediate(conn: &Connection) -> Result<rusqlite::Transaction<'_>> {
    Ok(conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test test_begin_immediate test_immediate_transaction_rollback -- --nocapture`
Expected: Both PASS

**Step 5: Wrap store_episode in IMMEDIATE transaction**

In `src/lib.rs`, replace the `store_episode` method (lines 50-73) with:

```rust
    /// Store a conversation episode with full context.
    pub fn store_episode(&self, episode: &NewEpisode) -> Result<EpisodeId> {
        let tx = schema::begin_immediate(&self.conn)?;

        let id = store::episodic::store_episode(&tx, episode)?;

        // Store embedding if provided
        if let Some(ref emb) = episode.embedding {
            store::embeddings::store_embedding(&tx, "episode", id.0, emb, "")?;
        }

        // Initialize node strength
        store::strengths::init_strength(&tx, NodeRef::Episode(id))?;

        // Create temporal link to preceding episode
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
```

**Step 6: Wrap lifecycle write methods in IMMEDIATE transactions**

In `src/lib.rs`, replace `consolidate` (lines 152-154):

```rust
    /// Run consolidation: episodic -> semantic (CLS replay).
    pub fn consolidate(&self, provider: &dyn ConsolidationProvider) -> Result<ConsolidationReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let report = lifecycle::consolidation::consolidate(&tx, provider)?;
        tx.commit()?;
        Ok(report)
    }
```

Replace `perfume` (lines 157-163):

```rust
    /// Run perfuming: extract impressions, crystallize preferences (vasana).
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
```

Replace `transform` (lines 166-168):

```rust
    /// Run transformation: dedup, prune, decay (asraya-paravrtti).
    pub fn transform(&self) -> Result<TransformationReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let report = lifecycle::transformation::transform(&tx)?;
        tx.commit()?;
        Ok(report)
    }
```

Replace `forget` (lines 171-173):

```rust
    /// Run forgetting: decay retrieval strengths, archive weak nodes (Bjork).
    pub fn forget(&self) -> Result<ForgettingReport> {
        let tx = schema::begin_immediate(&self.conn)?;
        let report = lifecycle::forgetting::forget(&tx)?;
        tx.commit()?;
        Ok(report)
    }
```

Replace `purge` (lines 192-219):

```rust
    /// Purge data matching the filter.
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
```

**Step 7: Run full test suite**

Run: `cargo test`
Expected: All tests pass (existing tests exercise store_episode and purge paths)

**Step 8: Write an atomicity test for store_episode**

Add to `src/lib.rs` inside `#[cfg(test)] mod tests`:

```rust
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
        // Episode, embedding, and strength all committed together
        assert_eq!(status.episode_count, 1);
        assert_eq!(status.embedding_count, 1);
        assert!(id.0 > 0);
    }
```

**Step 9: Run test to verify**

Run: `cargo test test_store_episode_with_embedding_is_atomic -- --nocapture`
Expected: PASS

**Step 10: Commit**

```bash
git add src/schema.rs src/lib.rs
git commit -m "feat: wrap write methods in BEGIN IMMEDIATE transactions (GAP-002)

Adds schema::begin_immediate() helper. All write-path methods on
AlayaStore (store_episode, consolidate, perfume, transform, forget,
purge) now acquire the write lock at BEGIN rather than at first write,
preventing SQLITE_BUSY under concurrent readers. Uncommitted
transactions roll back on drop via rusqlite's RAII Transaction."
```

---

### Task 5: Add input validation at API boundary (GAP-003)

**Files:**
- Modify: `src/lib.rs:50,80` (store_episode and query methods)
- Test: `src/lib.rs` (tests module)

**Step 1: Write the failing tests**

Add to `src/lib.rs` inside `#[cfg(test)] mod tests`:

```rust
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test test_store_episode_rejects test_query_rejects -- --nocapture`
Expected: FAIL — `store_episode` currently accepts empty content and `query` accepts empty text

**Step 3: Add validation to store_episode**

In `src/lib.rs`, at the start of the `store_episode` method body (right after the opening brace, before the `let tx = ...` line), add:

```rust
        if episode.content.trim().is_empty() {
            return Err(AlayaError::InvalidInput("episode content must not be empty".into()));
        }
        if episode.session_id.trim().is_empty() {
            return Err(AlayaError::InvalidInput("session_id must not be empty".into()));
        }
```

**Step 4: Run store_episode validation tests**

Run: `cargo test test_store_episode_rejects -- --nocapture`
Expected: Both PASS

**Step 5: Add validation to query**

In `src/lib.rs`, at the start of the `query` method body (right after the opening brace, before the `retrieval::pipeline::execute_query` call), add:

```rust
        if q.text.trim().is_empty() {
            return Err(AlayaError::InvalidInput("query text must not be empty".into()));
        }
        if q.max_results == 0 {
            return Err(AlayaError::InvalidInput("max_results must be greater than 0".into()));
        }
```

**Step 6: Run query validation tests**

Run: `cargo test test_query_rejects -- --nocapture`
Expected: Both PASS

**Step 7: Run full test suite**

Run: `cargo test`
Expected: All tests pass

**Step 8: Commit**

```bash
git add src/lib.rs
git commit -m "feat: add input validation at API boundary (GAP-003)

store_episode rejects empty content and empty session_id.
query rejects empty text and zero max_results.
All return AlayaError::InvalidInput with descriptive messages."
```

---

### Task 6: Change internal modules to pub(crate) (GAP-004)

**Files:**
- Modify: `src/lib.rs:10-17` (module declarations)

**Step 1: Run full test suite to establish baseline**

Run: `cargo test`
Expected: All tests pass

**Step 2: Change module visibility**

In `src/lib.rs`, replace the module declarations (lines 10-17):

From:
```rust
pub mod error;
pub mod types;
pub mod schema;
pub mod store;
pub mod graph;
pub mod retrieval;
pub mod lifecycle;
pub mod provider;
```

To:
```rust
pub(crate) mod error;
pub(crate) mod types;
pub(crate) mod schema;
pub(crate) mod store;
pub(crate) mod graph;
pub(crate) mod retrieval;
pub(crate) mod lifecycle;
pub(crate) mod provider;
```

The existing `pub use` re-exports on lines 22-24 continue to work because they re-export specific items at the crate root:

```rust
pub use error::{AlayaError, Result};           // consumers use alaya::AlayaError
pub use provider::{ConsolidationProvider, NoOpProvider};  // consumers use alaya::ConsolidationProvider
pub use types::*;                              // consumers use alaya::NewEpisode, etc.
```

Consumers access types through the crate root (e.g., `alaya::NewEpisode`) rather than module paths (e.g., `alaya::types::NewEpisode`). Internal implementation modules (`schema`, `store`, `graph`, `retrieval`, `lifecycle`) are no longer directly accessible from outside the crate.

**Step 3: Run full test suite to verify no regressions**

Run: `cargo test`
Expected: All tests pass — unit tests are within the crate and unaffected by `pub(crate)`

**Step 4: Verify with cargo doc**

Run: `cargo doc --no-deps 2>&1 | head -20`
Expected: No errors. Public items still appear in docs via re-exports.

**Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "feat: restrict internal modules to pub(crate) (GAP-004)

All 8 modules changed from pub to pub(crate). Public API surface is
maintained through explicit re-exports: AlayaError, Result,
ConsolidationProvider, NoOpProvider, and all types via pub use types::*.
Internal modules (schema, store, graph, retrieval, lifecycle) are no
longer directly accessible from outside the crate."
```

---

### Task 7: Add compilable doctests on pub API surface (GAP-005)

**Files:**
- Modify: `src/lib.rs:26-220` (AlayaStore struct and methods)
- Modify: `src/types.rs:1-10,27,66,93,123,158,172,282,301,330`
- Modify: `src/error.rs:3-4`
- Modify: `src/provider.rs:4-6`

This task adds `/// # Examples` doc blocks with compilable code to the public API.
Doctests run as external code, so they use `alaya::TypeName` imports.

**Step 1: Run existing doctest baseline**

Run: `cargo test --doc`
Expected: `running 0 tests` (no doctests exist yet)

**Step 2: Add doctest to AlayaStore struct**

In `src/lib.rs`, replace the doc comment on `AlayaStore` (lines 26-27):

```rust
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
```

**Step 3: Add doctest to open**

Replace the doc comment on `open` (line 33):

```rust
    /// Open (or create) a persistent database at `path`.
    ///
    /// # Examples
    ///
    /// ```
    /// let dir = tempfile::tempdir().unwrap();
    /// let store = alaya::AlayaStore::open(dir.path().join("test.db")).unwrap();
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
```

**Step 4: Add doctest to open_in_memory**

Replace the doc comment on `open_in_memory` (line 39):

```rust
    /// Open an ephemeral in-memory database (useful for tests).
    ///
    /// # Examples
    ///
    /// ```
    /// let store = alaya::AlayaStore::open_in_memory().unwrap();
    /// ```
    pub fn open_in_memory() -> Result<Self> {
```

**Step 5: Add doctest to store_episode**

Replace the doc comment on `store_episode` (line 49):

```rust
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
```

**Step 6: Add doctest to query**

Replace the doc comment on `query` (line 79):

```rust
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
```

**Step 7: Add doctest to status**

Replace the doc comment on `status` (line 179):

```rust
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
```

**Step 8: Add doctest to purge**

Replace the doc comment on `purge` (line 191):

```rust
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
```

**Step 9: Add doctest to preferences**

Replace the doc comment on `preferences` (line 84):

```rust
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
```

**Step 10: Add doctest to knowledge**

Replace the doc comment on `knowledge` (line 89):

```rust
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
```

**Step 11: Add doctest to consolidate**

Replace the doc comment on `consolidate` (line 151):

```rust
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
```

**Step 12: Add doctest to transform**

Replace the doc comment on `transform` (line 165):

```rust
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
```

**Step 13: Add doctest to forget**

Replace the doc comment on `forget` (line 170):

```rust
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
```

**Step 14: Add doctest to Query::simple**

In `src/types.rs`, replace the `Query::simple` method (around line 291):

```rust
    /// Create a simple text query with default settings.
    ///
    /// # Examples
    ///
    /// ```
    /// let q = alaya::Query::simple("What is Rust?");
    /// assert_eq!(q.max_results, 5);
    /// ```
    pub fn simple(text: impl Into<String>) -> Self {
```

**Step 15: Add crate-level doctest**

In `src/lib.rs`, expand the module-level doc comment (lines 1-8) to include an example:

```rust
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
```

**Step 16: Run all doctests**

Run: `cargo test --doc`
Expected: All doctests pass (should be ~14 doc tests)

**Step 17: Run full test suite including doctests**

Run: `cargo test`
Expected: All unit tests and doc tests pass

**Step 18: Commit**

```bash
git add src/lib.rs src/types.rs
git commit -m "docs: add compilable doctests to pub API surface (GAP-005)

Adds runnable examples to AlayaStore (struct + 12 methods), Query::simple,
and the crate-level module doc. All examples use open_in_memory() for
zero-setup. The crate doc Quick Start shows the store-then-query flow."
```

---

## Summary

| Task | GAP | What | Files Changed |
|------|-----|------|---------------|
| 1 | — | tempfile dev-dep | `Cargo.toml`, `src/lib.rs` |
| 2 | — | Schema versioning | `src/schema.rs` |
| 3 | GAP-001 | `#[non_exhaustive]` | `src/types.rs`, `src/error.rs` |
| 4 | GAP-002 | BEGIN IMMEDIATE | `src/schema.rs`, `src/lib.rs` |
| 5 | GAP-003 | Input validation | `src/lib.rs` |
| 6 | GAP-004 | `pub(crate)` modules | `src/lib.rs` |
| 7 | GAP-005 | Doctests | `src/lib.rs`, `src/types.rs` |

**Total commits:** 7 (one per task)

**Verification after all tasks:**
```bash
cargo test           # all unit + doc tests pass
cargo test --doc     # ~14 doctests compile and pass
cargo doc --no-deps  # documentation builds cleanly
```
