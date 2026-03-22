# Radical Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `AlayaStore` god object with a coordinator + sub-manager architecture, extract DRY helpers, enrich errors, and add optional tracing.

**Architecture:** New `Alaya` coordinator owns connection and providers, hands out zero-cost sub-manager structs (`Episodes`, `Knowledge`, `Lifecycle`, `Graph`, `Admin`) via accessor methods. Shared `db.rs` provides timestamps, transaction wrappers, and JSON helpers. Error context via `ResultExt` trait. Optional `tracing` feature for structured observability.

**Tech Stack:** Rust 1.85, rusqlite 0.32, serde/serde_json, thiserror 2, tracing 0.1 (optional)

**Spec:** `docs/superpowers/specs/2026-03-23-radical-refactor-design.md`

---

## File Structure

### New Files
- `alaya/src/db.rs` — Shared DB helpers: `now()`, `transact()`, `to_json()`, `from_json_or_default()`, `ResultExt`
- `alaya/src/managers/mod.rs` — Re-exports for sub-managers
- `alaya/src/managers/episodes.rs` — Episodes sub-manager
- `alaya/src/managers/knowledge.rs` — Knowledge sub-manager
- `alaya/src/managers/lifecycle.rs` — Lifecycle sub-manager
- `alaya/src/managers/graph.rs` — Graph sub-manager
- `alaya/src/managers/admin.rs` — Admin sub-manager
- `alaya/src/mcp/handler.rs` — Shared MCP handler helpers
- `alaya/src/testutil.rs` — Shared test fixtures (`#[cfg(test)]`)

### Modified Files
- `alaya/src/lib.rs` — Rewrite: `AlayaStore` → `Alaya` coordinator with sub-manager accessors (~200 LOC, down from 2333)
- `alaya/src/error.rs` — `Db(rusqlite::Error)` → `Db { source, context }`
- `alaya/src/store/episodic.rs` — Replace `SystemTime::now()` with `db::now()`
- `alaya/src/store/semantic.rs` — Replace timestamps, use `db::from_json_or_default()`
- `alaya/src/store/implicit.rs` — Replace 7 timestamp calls
- `alaya/src/store/categories.rs` — Remove local `now()`, use `db::now()`
- `alaya/src/store/strengths.rs` — Replace 2 timestamp calls
- `alaya/src/store/embeddings.rs` — Replace 1 timestamp call
- `alaya/src/graph/links.rs` — Replace 2 timestamp calls
- `alaya/src/lifecycle/reconciliation.rs` — Replace 2 timestamp calls
- `alaya/src/lifecycle/transformation.rs` — Replace 1 timestamp call
- `alaya/src/retrieval/pipeline.rs` — Replace 1 timestamp call
- `alaya/src/mcp/mod.rs` — `AlayaStore` → `Alaya`
- `alaya/src/mcp/memory.rs` — Use handler helpers, new API
- `alaya/src/mcp/query.rs` — Use handler helpers, new API
- `alaya/src/mcp/preferences.rs` — Use handler helpers, new API
- `alaya/src/mcp/lifecycle.rs` — Use handler helpers, new API
- `alaya/src/mcp/status.rs` — Use handler helpers, new API
- `alaya/src/mcp/import.rs` — Use handler helpers, new API
- `alaya/src/mcp/validation.rs` — Remove `now_timestamp()`, use `db::now()`
- `alaya/src/async_store.rs` — `AsyncAlayaStore` → `AsyncAlaya`, updated Request enum
- `alaya/src/bin/alaya-mcp.rs` — Update to use `Alaya`
- `alaya/Cargo.toml` — Verify tracing feature flags
- `alaya/tests/reconciliation.rs` — Update to use `Alaya`
- `alaya/tests/async_coverage.rs` — Update to use `AsyncAlaya`
- `alaya/tests/coverage_gaps.rs` — Update to use `Alaya`

---

## Task 1: Create `db.rs` — Shared DB Helpers

**Files:**
- Create: `alaya/src/db.rs`
- Modify: `alaya/src/lib.rs` (add `pub(crate) mod db;`)

- [ ] **Step 1: Write tests for db helpers**

```rust
// At the bottom of alaya/src/db.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_returns_reasonable_timestamp() {
        let ts = now();
        // Should be after 2024-01-01 and before 2100-01-01
        assert!(ts > 1_704_067_200, "timestamp too old: {ts}");
        assert!(ts < 4_102_444_800, "timestamp too far in future: {ts}");
    }

    #[test]
    fn transact_commits_on_success() {
        let conn = crate::schema::open_memory_db().unwrap();
        let result = transact(&conn, |tx| {
            tx.execute("INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('test', 'user', 's1', 1000)", [])?;
            Ok(42)
        });
        assert_eq!(result.unwrap(), 42);
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn transact_rolls_back_on_error() {
        let conn = crate::schema::open_memory_db().unwrap();
        let result: crate::Result<()> = transact(&conn, |tx| {
            tx.execute("INSERT INTO episodes (content, role, session_id, timestamp) VALUES ('test', 'user', 's1', 1000)", [])?;
            Err(crate::AlayaError::InvalidInput("intentional".into()))
        });
        assert!(result.is_err());
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn to_json_serializes() {
        let v = vec!["a", "b"];
        let json = to_json(&v).unwrap();
        assert_eq!(json, r#"["a","b"]"#);
    }

    #[test]
    fn from_json_or_default_parses() {
        let v: Vec<String> = from_json_or_default(r#"["a","b"]"#);
        assert_eq!(v, vec!["a", "b"]);
    }

    #[test]
    fn from_json_or_default_returns_default_on_bad_input() {
        let v: Vec<String> = from_json_or_default("not json");
        assert!(v.is_empty());
    }

    #[test]
    fn result_ext_adds_context_to_rusqlite_error() {
        let err: std::result::Result<(), rusqlite::Error> = Err(rusqlite::Error::QueryReturnedNoRows);
        let result: crate::Result<()> = err.with_context("test_operation");
        match result.unwrap_err() {
            crate::AlayaError::Db { context, .. } => assert_eq!(context, "test_operation"),
            other => panic!("expected Db error, got: {other}"),
        }
    }

    #[test]
    fn result_ext_passes_through_ok() {
        let ok: std::result::Result<i32, rusqlite::Error> = Ok(42);
        assert_eq!(ok.with_context("ctx").unwrap(), 42);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p alaya db::tests -- --test-threads=1`
Expected: FAIL — module `db` does not exist

- [ ] **Step 3: Implement db.rs**

```rust
//! Shared database helpers — timestamps, transactions, JSON, error context.

use crate::error::{AlayaError, Result};
use crate::schema;
use rusqlite::Connection;
use std::time::{SystemTime, UNIX_EPOCH};

/// Current Unix timestamp in seconds.
pub(crate) fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Run `f` inside a BEGIN IMMEDIATE transaction, committing on success.
pub(crate) fn transact<F, T>(conn: &Connection, f: F) -> Result<T>
where
    F: FnOnce(&rusqlite::Transaction) -> Result<T>,
{
    let tx = schema::begin_immediate(conn)?;
    let result = f(&tx)?;
    tx.commit()?;
    Ok(result)
}

/// Serialize a value to JSON string.
pub(crate) fn to_json<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(Into::into)
}

/// Deserialize JSON, returning `T::default()` on parse failure.
/// Logs a warning via tracing when the tracing feature is enabled.
pub(crate) fn from_json_or_default<T: serde::de::DeserializeOwned + Default>(s: &str) -> T {
    match serde_json::from_str(s) {
        Ok(v) => v,
        Err(_e) => {
            #[cfg(feature = "tracing")]
            tracing::warn!(input = s, error = %_e, "JSON parse failed, using default");
            T::default()
        }
    }
}

/// Extension trait for adding operation context to rusqlite errors.
pub(crate) trait ResultExt<T> {
    fn with_context(self, ctx: &str) -> Result<T>;
}

impl<T> ResultExt<T> for std::result::Result<T, rusqlite::Error> {
    fn with_context(self, ctx: &str) -> Result<T> {
        self.map_err(|e| AlayaError::Db {
            source: e,
            context: ctx.to_string(),
        })
    }
}

impl<T> ResultExt<T> for Result<T> {
    fn with_context(self, ctx: &str) -> Result<T> {
        self.map_err(|e| match e {
            AlayaError::Db { source, .. } => AlayaError::Db {
                source,
                context: ctx.to_string(),
            },
            other => other,
        })
    }
}
```

Add module declaration to `alaya/src/lib.rs` — add `pub(crate) mod db;` after the existing module declarations (after line 32):

```rust
pub(crate) mod db;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p alaya db::tests -- --test-threads=1`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add alaya/src/db.rs alaya/src/lib.rs
git commit --no-gpg-sign -m "feat: add db.rs shared helpers (now, transact, JSON, ResultExt)"
```

---

## Task 2: Enrich `error.rs` — Db Context

**Files:**
- Modify: `alaya/src/error.rs`
- Modify: All files that pattern-match on `AlayaError::Db` (search with `grep -rn "AlayaError::Db\|Db(" alaya/src/`)

- [ ] **Step 1: Update error.rs tests**

Update existing tests in `error.rs` to expect the new struct variant:

```rust
#[test]
fn test_from_rusqlite_error() {
    let sqlite_err = rusqlite::Error::QueryReturnedNoRows;
    let e: AlayaError = AlayaError::Db {
        source: sqlite_err,
        context: String::new(),
    };
    assert!(matches!(e, AlayaError::Db { .. }));
    assert!(e.to_string().contains("database error"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p alaya error::tests -- --test-threads=1`
Expected: FAIL — current `Db` is a tuple variant

- [ ] **Step 3: Change AlayaError::Db to struct variant**

Replace the `Db` variant in `alaya/src/error.rs`:

```rust
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AlayaError {
    #[error("database error in {context}: {source}")]
    Db {
        #[source]
        source: rusqlite::Error,
        context: String,
    },

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("actor dead: message channel closed")]
    ActorDead,
}
```

Remove the `#[from] rusqlite::Error` auto-conversion since we now need context. Add a manual `From` impl:

```rust
impl From<rusqlite::Error> for AlayaError {
    fn from(e: rusqlite::Error) -> Self {
        AlayaError::Db {
            source: e,
            context: String::new(),
        }
    }
}
```

- [ ] **Step 4: Fix all pattern matches on AlayaError::Db across the codebase**

Search for all matches on `Db`:

```bash
grep -rn "AlayaError::Db\b" alaya/src/
```

Each `AlayaError::Db(_)` pattern becomes `AlayaError::Db { .. }`. Key locations:
- `alaya/src/lib.rs` — `not_found_to_none` function
- `alaya/src/db.rs` — `ResultExt` impl (already uses struct variant)
- Any test files matching on `Db`

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add alaya/src/error.rs alaya/src/lib.rs alaya/src/db.rs
git commit --no-gpg-sign -m "feat(error): enrich AlayaError::Db with operation context"
```

---

## Task 3: Replace Timestamps with `db::now()`

**Files:**
- Modify: 12 files (see list below)

This is a mechanical search-and-replace. In each file:

1. Remove the local `use std::time::{SystemTime, UNIX_EPOCH};` import (if no longer used)
2. Replace `let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64;` with `let now = crate::db::now();`
3. For `store/categories.rs`: delete the local `fn now()` function and replace calls with `crate::db::now()`
4. For `mcp/validation.rs`: change `now_timestamp()` body to call `crate::db::now()` (keep the public name for MCP layer)

- [ ] **Step 1: Replace timestamps in store modules**

Files to modify:
- `store/episodic.rs` (line ~6 in `store_episode`)
- `store/semantic.rs` (lines ~6, ~53)
- `store/implicit.rs` (lines ~6, ~55, ~117, ~152, ~256, ~276, ~477)
- `store/categories.rs` (delete local `fn now()` at line 6-10, replace all `now()` calls with `crate::db::now()`)
- `store/strengths.rs` (lines ~6, ~46)
- `store/embeddings.rs` (line ~43)

Pattern: replace `std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64` with `crate::db::now()`

- [ ] **Step 2: Replace timestamps in non-store modules**

Files to modify:
- `graph/links.rs` (lines ~12, ~63)
- `lifecycle/reconciliation.rs` (lines ~59, ~98)
- `lifecycle/transformation.rs` (line ~55)
- `retrieval/pipeline.rs` (line ~17)
- `lib.rs` (line ~667 in `resolve_conflict`)
- `mcp/validation.rs` (line ~19, change body to `crate::db::now()`)

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add -u alaya/src/
git commit --no-gpg-sign -m "refactor: replace 23 scattered timestamp calls with db::now()"
```

---

## Task 4: Create `testutil.rs` — Shared Test Fixtures

**Files:**
- Create: `alaya/src/testutil.rs`
- Modify: `alaya/src/lib.rs` (add module declaration)

- [ ] **Step 1: Create testutil.rs**

```rust
//! Shared test utilities — fixtures, factories, builders.
//! Only compiled in test builds.

#[cfg(test)]
pub(crate) mod fixtures {
    use crate::types::*;
    use rusqlite::Connection;

    /// Open an in-memory DB with schema initialized.
    pub fn test_db() -> Connection {
        crate::schema::open_memory_db().unwrap()
    }

    /// Episode factory with sensible defaults.
    pub fn episode(content: &str) -> NewEpisode {
        NewEpisode {
            content: content.to_string(),
            role: Role::User,
            session_id: "test-session".to_string(),
            timestamp: 1000,
            context: EpisodeContext::default(),
            embedding: None,
        }
    }

    /// Episode with custom role and timestamp.
    pub fn episode_at(content: &str, role: Role, ts: i64) -> NewEpisode {
        NewEpisode {
            timestamp: ts,
            role,
            ..episode(content)
        }
    }

    /// Insert a semantic node directly in the DB, returning its NodeId.
    pub fn insert_semantic_node(conn: &Connection, content: &str, confidence: f32) -> NodeId {
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
             VALUES (?1, 'fact', ?2, 1000, 1000, 1)",
            rusqlite::params![content, confidence],
        )
        .unwrap();
        NodeId(conn.last_insert_rowid())
    }

    /// Insert a semantic node with a specific type.
    pub fn insert_typed_node(
        conn: &Connection,
        content: &str,
        node_type: &str,
        confidence: f32,
    ) -> NodeId {
        conn.execute(
            "INSERT INTO semantic_nodes (content, node_type, confidence, created_at, last_corroborated, corroboration_count)
             VALUES (?1, ?2, ?3, 1000, 1000, 1)",
            rusqlite::params![content, node_type, confidence],
        )
        .unwrap();
        NodeId(conn.last_insert_rowid())
    }

    /// Store an episode via the store module, returning its EpisodeId.
    pub fn store_test_episode(conn: &Connection, content: &str) -> EpisodeId {
        crate::store::episodic::store_episode(conn, &episode(content)).unwrap()
    }
}
```

- [ ] **Step 2: Add module declaration to lib.rs**

Add after the other module declarations:

```rust
#[cfg(test)]
pub(crate) mod testutil;
```

- [ ] **Step 3: Run tests to verify compilation**

Run: `cargo test -p alaya --lib -- testutil --test-threads=1`
Expected: PASS (no tests to run, but compiles)

- [ ] **Step 4: Commit**

```bash
git add alaya/src/testutil.rs alaya/src/lib.rs
git commit --no-gpg-sign -m "test: add shared testutil module with fixtures and factories"
```

---

## Task 5: Create Managers Module Scaffold

**Files:**
- Create: `alaya/src/managers/mod.rs`
- Modify: `alaya/src/lib.rs` (add module declaration)

- [ ] **Step 1: Create managers/mod.rs with sub-manager struct definitions**

```rust
//! Sub-manager types for the Alaya coordinator.
//!
//! Each sub-manager borrows from the parent `Alaya` instance
//! and provides a focused API for a specific domain.

pub(crate) mod admin;
pub(crate) mod episodes;
pub(crate) mod graph;
pub(crate) mod knowledge;
pub(crate) mod lifecycle;

use rusqlite::Connection;

use crate::provider::{EmbeddingProvider, ExtractionProvider};
use crate::types::ConflictStrategy;

/// Episodes sub-manager: store and query conversation episodes.
#[non_exhaustive]
pub struct Episodes<'a> {
    pub(crate) conn: &'a Connection,
    pub(crate) embedding_provider: Option<&'a dyn EmbeddingProvider>,
}

/// Knowledge sub-manager: query and manage semantic knowledge.
#[non_exhaustive]
pub struct Knowledge<'a> {
    pub(crate) conn: &'a Connection,
    pub(crate) embedding_provider: Option<&'a dyn EmbeddingProvider>,
}

/// Lifecycle sub-manager: consolidation, transformation, forgetting, reconciliation.
///
/// **Provider note:** Only `extraction_provider` is stored (needed by `auto_consolidate`).
/// `consolidate`, `dream`, and `perfume` take `&dyn ConsolidationProvider` as a
/// parameter — no `embedding_provider` needed. `auto_consolidate` calls
/// `store::episodic::unconsolidated_episodes` and `lifecycle::consolidation::learn_direct`
/// directly via its borrowed `conn`.
#[non_exhaustive]
pub struct Lifecycle<'a> {
    pub(crate) conn: &'a Connection,
    pub(crate) extraction_provider: Option<&'a dyn ExtractionProvider>,
    pub(crate) conflict_strategy: ConflictStrategy,
}

/// Graph sub-manager: neighbors, link queries.
#[non_exhaustive]
pub struct Graph<'a> {
    pub(crate) conn: &'a Connection,
}

/// Admin sub-manager: status, purge, categories, preferences.
#[non_exhaustive]
pub struct Admin<'a> {
    pub(crate) conn: &'a Connection,
}
```

- [ ] **Step 2: Create empty sub-manager files**

Create each file with just the module imports and an empty impl block:

`alaya/src/managers/episodes.rs`:
```rust
use super::Episodes;
use crate::error::Result;

impl Episodes<'_> {
    // Methods will be moved here from lib.rs in Task 6
}
```

Same pattern for `knowledge.rs`, `lifecycle.rs`, `graph.rs`, `admin.rs` — each imports its own struct from `super`.

- [ ] **Step 3: Add module declaration to lib.rs**

```rust
pub mod managers;
```

- [ ] **Step 4: Run tests to verify compilation**

Run: `cargo test -p alaya --lib -- --test-threads=1 2>&1 | tail -5`
Expected: compiles and all existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add alaya/src/managers/
git commit --no-gpg-sign -m "feat: add managers module scaffold with sub-manager structs"
```

---

## Task 6: Implement `Episodes` Sub-Manager

**Files:**
- Modify: `alaya/src/managers/episodes.rs`
- Modify: `alaya/src/lib.rs` (remove `store_episode`, `episodes_by_session`, `unconsolidated_episodes`)

- [ ] **Step 1: Write tests for Episodes sub-manager**

Add tests to `managers/episodes.rs`:

```rust
#[cfg(test)]
mod tests {
    use crate::testutil::fixtures::*;
    use crate::Alaya;

    #[test]
    fn store_and_retrieve_episode() {
        let alaya = Alaya::open_in_memory().unwrap();
        let id = alaya.episodes().store(&episode("test content")).unwrap();
        assert!(id.0 > 0);

        let eps = alaya.episodes().by_session("test-session").unwrap();
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].content, "test content");
    }

    #[test]
    fn store_rejects_empty_content() {
        let alaya = Alaya::open_in_memory().unwrap();
        let result = alaya.episodes().store(&episode(""));
        assert!(result.is_err());
    }

    #[test]
    fn store_rejects_empty_session_id() {
        let alaya = Alaya::open_in_memory().unwrap();
        let mut ep = episode("content");
        ep.session_id = "".to_string();
        assert!(alaya.episodes().store(&ep).is_err());
    }

    #[test]
    fn unconsolidated_returns_new_episodes() {
        let alaya = Alaya::open_in_memory().unwrap();
        alaya.episodes().store(&episode("msg 1")).unwrap();
        alaya.episodes().store(&episode("msg 2")).unwrap();
        let uncons = alaya.episodes().unconsolidated(100).unwrap();
        assert_eq!(uncons.len(), 2);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p alaya managers::episodes -- --test-threads=1`
Expected: FAIL — `Alaya` type doesn't exist yet, methods not implemented

- [ ] **Step 3: Implement Episodes methods**

Move the method bodies from `AlayaStore` in `lib.rs` into `managers/episodes.rs`:

```rust
use super::Episodes;
use crate::db;
use crate::error::{AlayaError, Result};
use crate::types::*;
use crate::{graph, schema, store};

impl Episodes<'_> {
    /// Store a conversation episode with full context.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn store(&self, episode: &NewEpisode) -> Result<EpisodeId> {
        if episode.content.trim().is_empty() {
            return Err(AlayaError::InvalidInput(
                "episode content must not be empty".into(),
            ));
        }
        if episode.session_id.trim().is_empty() {
            return Err(AlayaError::InvalidInput(
                "session_id must not be empty".into(),
            ));
        }

        db::transact(self.conn, |tx| {
            let id = store::episodic::store_episode(tx, episode)?;

            let effective_embedding = match &episode.embedding {
                Some(emb) => Some(emb.clone()),
                None => self
                    .embedding_provider
                    .and_then(|p| p.embed(&episode.content).ok()),
            };
            if let Some(ref emb) = effective_embedding {
                store::embeddings::store_embedding(tx, "episode", id.0, emb, "")?;
            }

            store::strengths::init_strength(tx, NodeRef::Episode(id))?;

            if let Some(prev) = episode.context.preceding_episode {
                graph::links::create_link(
                    tx,
                    NodeRef::Episode(prev),
                    NodeRef::Episode(id),
                    LinkType::Temporal,
                    0.5,
                )?;
            }

            Ok(id)
        })
    }

    /// Return all episodes belonging to the given session.
    pub fn by_session(&self, session_id: &str) -> Result<Vec<Episode>> {
        store::episodic::get_episodes_by_session(self.conn, session_id)
    }

    /// Return unconsolidated episodes (not yet linked to any semantic node).
    pub fn unconsolidated(&self, limit: u32) -> Result<Vec<Episode>> {
        store::episodic::get_unconsolidated_episodes(self.conn, limit)
    }
}
```

- [ ] **Step 4: Create Alaya coordinator in lib.rs**

This step creates the new `Alaya` struct alongside the existing `AlayaStore`. After all managers are done (Tasks 6-10), we'll remove `AlayaStore`.

**Note:** This step defines the coordinator's configuration methods — `set_embedding_provider`, `set_extraction_provider`, `set_conflict_strategy`, and `rekey` — which stay on the `Alaya` struct directly (they require `&mut self`).

Add to `lib.rs` below the existing `AlayaStore` struct:

```rust
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p alaya managers::episodes -- --test-threads=1`
Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add alaya/src/managers/episodes.rs alaya/src/managers/mod.rs alaya/src/lib.rs
git commit --no-gpg-sign -m "feat(managers): implement Episodes sub-manager with store, by_session, unconsolidated"
```

---

## Task 7: Implement `Knowledge` Sub-Manager

**Files:**
- Modify: `alaya/src/managers/knowledge.rs`

- [ ] **Step 1: Write tests**

Tests covering `query`, `learn`, `filter`, `breakdown`.

- [ ] **Step 2: Implement Knowledge methods**

Move from `AlayaStore`: `query()`, `learn()`, `knowledge()` (rename to `filter()`), `knowledge_breakdown()` (rename to `breakdown()`).

```rust
use super::Knowledge;
use crate::db;
use crate::error::{AlayaError, Result};
use crate::types::*;
use crate::{lifecycle, retrieval, store};

impl Knowledge<'_> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn query(&self, q: &Query) -> Result<Vec<ScoredMemory>> {
        // validation + auto-embed + pipeline::execute_query
        // (move body from AlayaStore::query)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn learn(&self, nodes: Vec<NewSemanticNode>) -> Result<ConsolidationReport> {
        db::transact(self.conn, |tx| lifecycle::consolidation::learn_direct(tx, nodes))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn filter(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>> {
        // (move body from AlayaStore::knowledge)
    }

    pub fn breakdown(&self) -> Result<std::collections::HashMap<SemanticType, u64>> {
        store::semantic::count_nodes_by_type(self.conn)
    }
}
```

- [ ] **Step 3: Run tests, commit**

```bash
git commit --no-gpg-sign -m "feat(managers): implement Knowledge sub-manager with query, learn, filter, breakdown"
```

---

## Task 8: Implement `Lifecycle` Sub-Manager

**Files:**
- Modify: `alaya/src/managers/lifecycle.rs`

- [ ] **Step 1: Write tests**

Tests covering `consolidate`, `auto_consolidate`, `transform`, `forget`, `perfume`, `dream`, `reconcile`, `conflicts`, `resolve_conflict`.

- [ ] **Step 2: Implement Lifecycle methods**

Most methods are one-liners using `db::transact`:

```rust
use super::Lifecycle;
use crate::db;
use crate::error::{AlayaError, Result};
use crate::types::*;
use crate::{graph, lifecycle, store};

impl Lifecycle<'_> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, provider)))]
    pub fn consolidate(&self, provider: &dyn crate::ConsolidationProvider) -> Result<ConsolidationReport> {
        db::transact(self.conn, |tx| lifecycle::consolidation::consolidate(tx, provider))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn auto_consolidate(&self) -> Result<ConsolidationReport> {
        // (move body from AlayaStore::auto_consolidate — needs extraction_provider)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn transform(&self) -> Result<TransformationReport> {
        db::transact(self.conn, lifecycle::transformation::transform)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn forget(&self) -> Result<ForgettingReport> {
        db::transact(self.conn, lifecycle::forgetting::forget)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, provider)))]
    pub fn perfume(&self, interaction: &Interaction, provider: &dyn crate::ConsolidationProvider) -> Result<PerfumingReport> {
        db::transact(self.conn, |tx| lifecycle::perfuming::perfume(tx, interaction, provider))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, provider)))]
    pub fn dream(&self, provider: &dyn crate::ConsolidationProvider, interaction: Option<&Interaction>) -> Result<DreamReport> {
        // (move body from AlayaStore::dream — calls consolidate, perfume, transform, forget)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn reconcile(&self) -> Result<ReconcileReport> {
        db::transact(self.conn, |tx| lifecycle::reconciliation::reconcile(tx, self.conflict_strategy))
    }

    pub fn conflicts(&self) -> Result<Vec<Conflict>> {
        store::conflicts::get_unresolved_conflicts(self.conn)
    }

    pub fn resolve_conflict(&self, conflict_id: ConflictId, winner_id: NodeId) -> Result<()> {
        // (move body from AlayaStore::resolve_conflict)
    }
}
```

- [ ] **Step 3: Run tests, commit**

```bash
git commit --no-gpg-sign -m "feat(managers): implement Lifecycle sub-manager"
```

---

## Task 9: Implement `Graph` Sub-Manager

**Files:**
- Modify: `alaya/src/managers/graph.rs`

- [ ] **Step 1: Write tests**

- [ ] **Step 2: Implement Graph methods**

```rust
use super::Graph;
use crate::error::Result;
use crate::types::*;
use crate::graph;

impl Graph<'_> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>> {
        let result = graph::activation::spread_activation(self.conn, &[node], depth, 0.05, 0.6)?;
        let mut pairs: Vec<(NodeRef, f32)> =
            result.into_iter().filter(|(nr, _)| *nr != node).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(pairs)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn strongest_link(&self) -> Result<Option<(NodeRef, NodeRef, f32)>> {
        graph::links::strongest_link(self.conn)
    }
}
```

- [ ] **Step 3: Run tests, commit**

```bash
git commit --no-gpg-sign -m "feat(managers): implement Graph sub-manager"
```

---

## Task 10: Implement `Admin` Sub-Manager

**Files:**
- Modify: `alaya/src/managers/admin.rs`

- [ ] **Step 1: Write tests**

- [ ] **Step 2: Implement Admin methods**

```rust
use super::Admin;
use crate::db;
use crate::error::Result;
use crate::types::*;
use crate::{graph, store};

impl Admin<'_> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn status(&self) -> Result<MemoryStatus> {
        // (move body from AlayaStore::status)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport> {
        // (move body from AlayaStore::purge, use db::transact)
    }

    pub fn preferences(&self, domain: Option<&str>) -> Result<Vec<Preference>> {
        store::implicit::get_preferences(self.conn, domain)
    }

    pub fn categories(&self, min_stability: Option<f32>) -> Result<Vec<Category>> {
        store::categories::list_categories(self.conn, min_stability)
    }

    pub fn subcategories(&self, parent_id: CategoryId) -> Result<Vec<Category>> {
        store::categories::get_subcategories(self.conn, parent_id)
    }

    pub fn node_category(&self, node_id: NodeId) -> Result<Option<Category>> {
        store::categories::get_node_category(self.conn, node_id)
    }

    pub fn node_content(&self, node: NodeRef) -> Result<Option<String>> {
        // (move body from AlayaStore::node_content)
    }
}
```

- [ ] **Step 3: Run tests, commit**

```bash
git commit --no-gpg-sign -m "feat(managers): implement Admin sub-manager"
```

---

## Task 11: Remove `AlayaStore`, Finalize `Alaya`

**Files:**
- Modify: `alaya/src/lib.rs` — Delete old `AlayaStore` struct and entire impl block
- Modify: All test files referencing `AlayaStore`

- [ ] **Step 1: Delete `AlayaStore` from lib.rs**

Remove the `pub struct AlayaStore` definition and its entire `impl AlayaStore { ... }` block. Keep only the `Alaya` struct, the `truncate_label` helper (move to `managers/admin.rs`), and the module declarations.

- [ ] **Step 2: Add type alias for backwards compat during transition (optional)**

```rust
#[deprecated(note = "use Alaya instead")]
pub type AlayaStore = Alaya;
```

This is temporary — remove after updating all downstream code.

- [ ] **Step 3: Update lib.rs doc examples**

Update the module-level doc example and all `///` examples to use `Alaya`:

```rust
//! ```
//! use alaya::{Alaya, NewEpisode, Role, EpisodeContext, Query};
//!
//! let alaya = Alaya::open_in_memory().unwrap();
//! alaya.episodes().store(&NewEpisode { ... }).unwrap();
//! let results = alaya.knowledge().query(&Query::simple("Rust")).unwrap();
//! ```
```

- [ ] **Step 4: Update all internal tests referencing AlayaStore**

Search: `grep -rn "AlayaStore" alaya/src/ alaya/tests/`

Replace `AlayaStore` with `Alaya` and update method calls to use sub-manager syntax.

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add -u alaya/
git commit --no-gpg-sign -m "refactor: remove AlayaStore, Alaya is now the sole entry point"
```

---

## Task 12: Create MCP Handler Helpers

**Files:**
- Create: `alaya/src/mcp/handler.rs`
- Modify: `alaya/src/mcp/mod.rs` (add module, update McpServer to use Alaya)

- [ ] **Step 1: Create handler.rs**

```rust
//! Shared MCP handler helpers — parameter extraction and error formatting.

use serde_json::Value;

pub(crate) fn require_str<'a>(params: &'a Value, key: &str) -> std::result::Result<&'a str, String> {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing required parameter '{key}'"))
}

pub(crate) fn require_u64(params: &Value, key: &str) -> std::result::Result<u64, String> {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("missing required parameter '{key}'"))
}

pub(crate) fn optional_str<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(|v| v.as_str())
}

pub(crate) fn optional_u64(params: &Value, key: &str) -> Option<u64> {
    params.get(key).and_then(|v| v.as_u64())
}

pub(crate) fn optional_f64(params: &Value, key: &str) -> Option<f64> {
    params.get(key).and_then(|v| v.as_f64())
}

pub(crate) fn optional_bool(params: &Value, key: &str) -> Option<bool> {
    params.get(key).and_then(|v| v.as_bool())
}

/// Run a closure against the Alaya store and serialize the result.
/// Returns a JSON-formatted success string or an error response string.
pub(crate) fn run<F, T>(server: &super::AlayaMcp, f: F) -> String
where
    F: FnOnce(&crate::Alaya) -> crate::Result<T>,
    T: serde::Serialize,
{
    match server.with_store(f) {
        Ok(result) => serde_json::to_string_pretty(&result).unwrap_or_default(),
        Err(e) => super::error_response(&e.to_string()),
    }
}

/// Early-return macro for MCP handlers: converts a `Result<T, String>` from
/// `require_str`/`require_u64` into either the success value or an error
/// response string. Used in handler functions that return `String`.
///
/// Usage: `let val = try_or_err!(handler::require_str(&params, "key"));`
macro_rules! try_or_err {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(msg) => return $crate::mcp::error_response(&msg),
        }
    };
}
pub(crate) use try_or_err;
```

- [ ] **Step 2: Update mcp/mod.rs — change AlayaStore to Alaya in McpServer**

Replace `AlayaStore` with `Alaya` in the `McpServer` struct and its `with_store` method.

- [ ] **Step 3: Run MCP tests, commit**

```bash
git commit --no-gpg-sign -m "feat(mcp): add handler.rs helpers, update McpServer to use Alaya"
```

---

## Task 13: Refactor MCP Handlers

**Files:**
- Modify: `alaya/src/mcp/memory.rs`
- Modify: `alaya/src/mcp/query.rs`
- Modify: `alaya/src/mcp/preferences.rs`
- Modify: `alaya/src/mcp/lifecycle.rs`
- Modify: `alaya/src/mcp/status.rs`
- Modify: `alaya/src/mcp/import.rs`

- [ ] **Step 1: Refactor handlers to use handler.rs helpers and sub-manager API**

For each handler file:
1. Replace inline parameter extraction with `handler::require_str`, `handler::optional_str`, etc.
2. Replace `store.method()` calls with `store.episodes().method()`, `store.knowledge().method()`, etc.

Example — `mcp/memory.rs` handle_store_episode before and after:

Before:
```rust
pub fn handle_store_episode(server: &AlayaMcp, params: Value) -> String {
    let content = match params.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return error_response("missing 'content'"),
    };
    // ... more parameter extraction ...
    server.with_store(|store| store.store_episode(&episode))
}
```

After:
```rust
pub fn handle_store_episode(server: &AlayaMcp, params: Value) -> String {
    let content = try_or_err!(handler::require_str(&params, "content"));
    let role_str = try_or_err!(handler::require_str(&params, "role"));
    let session_id = try_or_err!(handler::require_str(&params, "session_id"));
    // ...
    server.with_store(|store| store.episodes().store(&episode))
}
```

- [ ] **Step 2: Run full MCP test suite**

Run: `cargo test -p alaya --features mcp mcp:: -- --test-threads=1`
Expected: all MCP tests PASS

- [ ] **Step 3: Commit**

```bash
git add -u alaya/src/mcp/
git commit --no-gpg-sign -m "refactor(mcp): use handler helpers and sub-manager API across all handlers"
```

---

## Task 14: Update Async Store and MCP Binary

**Files:**
- Modify: `alaya/src/async_store.rs`
- Modify: `alaya/src/bin/alaya-mcp.rs`

- [ ] **Step 1: Rename AsyncAlayaStore to AsyncAlaya**

Replace `AsyncAlayaStore` with `AsyncAlaya` throughout `async_store.rs`. Update the `run_actor` function to take `Alaya` instead of `AlayaStore`. Update constructors (`open`, `open_in_memory`, `open_encrypted`, `spawn`) to use `Alaya`.

- [ ] **Step 2: Update run_actor match arms to use sub-manager API**

The `Request` enum stays structurally the same (24 variants). Update each match arm in `run_actor` to call sub-manager methods. Full mapping:

```rust
fn run_actor(mut store: Alaya, rx: mpsc::Receiver<Request>) {
    let mut consolidation_provider: Box<dyn ConsolidationProvider + Send> = Box::new(NoOpProvider);

    let mut rx = rx;
    while let Some(req) = rx.blocking_recv() {
        match req {
            // --- Episodes ---
            Request::StoreEpisode { episode, reply } => {
                let _ = reply.send(store.episodes().store(&episode));
            }
            Request::EpisodesBySession { session_id, reply } => {
                let _ = reply.send(store.episodes().by_session(&session_id));
            }
            Request::UnconsolidatedEpisodes { limit, reply } => {
                let _ = reply.send(store.episodes().unconsolidated(limit));
            }

            // --- Knowledge ---
            Request::Query { query, reply } => {
                let _ = reply.send(store.knowledge().query(&query));
            }
            Request::Learn { nodes, reply } => {
                let _ = reply.send(store.knowledge().learn(nodes));
            }
            Request::Knowledge { filter, reply } => {
                let _ = reply.send(store.knowledge().filter(filter));
            }
            Request::KnowledgeBreakdown { reply } => {
                let _ = reply.send(store.knowledge().breakdown());
            }

            // --- Lifecycle ---
            Request::Consolidate { reply } => {
                let _ = reply.send(store.lifecycle().consolidate(consolidation_provider.as_ref()));
            }
            Request::AutoConsolidate { reply } => {
                let _ = reply.send(store.lifecycle().auto_consolidate());
            }
            Request::Perfume { interaction, reply } => {
                let _ = reply.send(store.lifecycle().perfume(&interaction, consolidation_provider.as_ref()));
            }
            Request::Transform { reply } => {
                let _ = reply.send(store.lifecycle().transform());
            }
            Request::Forget { reply } => {
                let _ = reply.send(store.lifecycle().forget());
            }
            Request::Dream { interaction, reply } => {
                let inter_ref = interaction.as_ref();
                let _ = reply.send(store.lifecycle().dream(consolidation_provider.as_ref(), inter_ref));
            }
            Request::Reconcile { reply } => {
                let _ = reply.send(store.lifecycle().reconcile());
            }
            Request::Conflicts { reply } => {
                let _ = reply.send(store.lifecycle().conflicts());
            }
            Request::ResolveConflict { conflict_id, winner_id, reply } => {
                let _ = reply.send(store.lifecycle().resolve_conflict(conflict_id, winner_id));
            }

            // --- Graph ---
            Request::Neighbors { node, depth, reply } => {
                let _ = reply.send(store.graph().neighbors(node, depth));
            }
            Request::StrongestLink { reply } => {
                let _ = reply.send(store.graph().strongest_link());
            }

            // --- Admin ---
            Request::Status { reply } => {
                let _ = reply.send(store.admin().status());
            }
            Request::Purge { filter, reply } => {
                let _ = reply.send(store.admin().purge(filter));
            }
            Request::Preferences { domain, reply } => {
                let _ = reply.send(store.admin().preferences(domain.as_deref()));
            }
            Request::Categories { min_stability, reply } => {
                let _ = reply.send(store.admin().categories(min_stability));
            }
            Request::Subcategories { parent_id, reply } => {
                let _ = reply.send(store.admin().subcategories(parent_id));
            }
            Request::NodeCategory { node_id, reply } => {
                let _ = reply.send(store.admin().node_category(node_id));
            }
            Request::NodeContent { node, reply } => {
                let _ = reply.send(store.admin().node_content(node));
            }

            // --- Configuration (mutate coordinator directly) ---
            Request::SetConsolidationProvider { provider } => {
                consolidation_provider = provider;
            }
            Request::SetEmbeddingProvider { provider } => {
                store.set_embedding_provider(provider);
            }
            Request::SetExtractionProvider { provider } => {
                store.set_extraction_provider(provider);
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
```

- [ ] **Step 3: Update alaya-mcp.rs binary**

In `alaya/src/bin/alaya-mcp.rs`:
1. Replace `use alaya::AlayaStore;` with `use alaya::Alaya;`
2. Replace `AlayaStore::open` with `Alaya::open`
3. Update `configure_extraction` signature: `fn configure_extraction(store: &mut Alaya)`

```rust
// Before
use alaya::AlayaStore;
let mut store = AlayaStore::open(&db_path)?;

// After
use alaya::Alaya;
let mut store = Alaya::open(&db_path)?;
```

- [ ] **Step 4: Run async tests**

Run: `cargo test -p alaya --features async async_store -- --test-threads=1`
Expected: all async tests PASS

- [ ] **Step 5: Verify MCP binary compiles**

Run: `cargo build -p alaya --features "mcp async" --bin alaya-mcp`
Expected: compiles without errors

- [ ] **Step 6: Commit**

```bash
git add alaya/src/async_store.rs alaya/src/bin/alaya-mcp.rs
git commit --no-gpg-sign -m "refactor(async): rename AsyncAlayaStore to AsyncAlaya, use sub-manager API, update MCP binary"
```

---

## Task 15: Replace JSON Patterns in Store Modules

**Files:**
- Modify: `store/episodic.rs`, `store/semantic.rs`, `store/implicit.rs`

- [ ] **Step 1: Replace serde_json calls with db helpers**

In each store module, replace:
- `serde_json::to_string(&ctx).unwrap_or_default()` → `crate::db::to_json(&ctx).unwrap_or_default()`
- `serde_json::from_str(&s).unwrap_or_default()` → `crate::db::from_json_or_default(&s)`

This is a mechanical replacement — search each file for `serde_json::to_string` and `serde_json::from_str` and replace.

- [ ] **Step 2: Run tests, commit**

```bash
git commit --no-gpg-sign -m "refactor(store): use db::to_json and db::from_json_or_default"
```

---

## Task 16: Add Tracing Instrumentation

**Files:**
- Modify: `alaya/Cargo.toml` (verify feature flags)
- Modify: `alaya/src/db.rs` (add span to transact)
- Modify: `alaya/src/retrieval/pipeline.rs` (add spans to pipeline stages)

- [ ] **Step 1: Verify tracing feature in Cargo.toml**

Already present: `tracing = ["dep:tracing", "dep:tracing-subscriber"]`. No changes needed.

- [ ] **Step 2: Add tracing span to db::transact**

```rust
#[cfg_attr(feature = "tracing", tracing::instrument(skip(conn, f)))]
pub(crate) fn transact<F, T>(conn: &Connection, f: F) -> Result<T>
```

- [ ] **Step 3: Add tracing spans to retrieval pipeline stages**

In `retrieval/pipeline.rs`, wrap each pipeline stage with conditional spans:

```rust
#[cfg(feature = "tracing")]
let _span = tracing::info_span!("bm25_search").entered();
```

Add spans for: `bm25_search`, `vector_search`, `graph_activation`, `rrf_fusion`, `rerank`.

- [ ] **Step 4: Run tests with tracing feature**

Run: `cargo test -p alaya --features "tracing mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add alaya/src/db.rs alaya/src/retrieval/pipeline.rs alaya/Cargo.toml
git commit --no-gpg-sign -m "feat(tracing): add spans to transactions and retrieval pipeline"
```

---

## Task 17: Update Integration Tests and Doc Examples

**Files:**
- Modify: `alaya/tests/reconciliation.rs`
- Modify: `alaya/tests/async_coverage.rs`
- Modify: `alaya/tests/coverage_gaps.rs`
- Modify: All doc examples in `lib.rs`, `types.rs`

- [ ] **Step 1: Update integration tests**

Replace `AlayaStore` with `Alaya` and update method calls to use sub-manager syntax in all test files under `alaya/tests/`.

- [ ] **Step 2: Update doc examples**

Update all `/// ```rust` examples in `lib.rs` to use the new API:
```rust
/// use alaya::{Alaya, NewEpisode, Role, EpisodeContext, Query};
/// let alaya = Alaya::open_in_memory().unwrap();
/// alaya.episodes().store(&NewEpisode { ... }).unwrap();
```

- [ ] **Step 3: Run full test suite including doc tests**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS including doc tests

- [ ] **Step 4: Commit**

```bash
git add -u alaya/
git commit --no-gpg-sign -m "test: update all tests and doc examples for Alaya sub-manager API"
```

---

## Task 18: Update Python Bindings

**Files:**
- Modify: `alaya-py/src/lib.rs`

- [ ] **Step 1: Update PyAlaya to use Alaya instead of AlayaStore**

Replace `AlayaStore` with `Alaya` in the Python bindings. Update method calls to use sub-manager syntax internally — the Python API surface can stay flat since Python doesn't have the sub-manager pattern.

- [ ] **Step 2: Run Python tests**

Run: `cd alaya-py && cargo test`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add -u alaya-py/
git commit --no-gpg-sign -m "refactor(python): update bindings for Alaya sub-manager API"
```

---

## Task 19: Final Verification — Coverage and Cleanup

**Files:**
- Modify: Any files with remaining issues

- [ ] **Step 1: Run full test suite with all features**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 2: Run tarpaulin coverage**

Run: `cargo tarpaulin -p alaya --features "mcp async" --out Stdout -- --test-threads=1`
Expected: coverage >= 95% (some new manager code may need additional tests)

- [ ] **Step 3: Run clippy**

Run: `cargo clippy -p alaya --features "mcp async" -- -D warnings`
Expected: no warnings

- [ ] **Step 4: Remove deprecated AlayaStore alias if present**

If Task 11 added a `type AlayaStore = Alaya` alias, remove it now.

- [ ] **Step 5: Remove old test helpers that are now in testutil**

Search for duplicated test helpers across modules and replace with `crate::testutil::fixtures::*` imports.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit --no-gpg-sign -m "chore: final cleanup, remove deprecated aliases, deduplicate test helpers"
```
