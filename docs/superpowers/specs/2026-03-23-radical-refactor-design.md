# Radical Refactor for Maintainability and Debuggability

## Goal

Replace the `AlayaStore` god object with a coordinator + sub-manager architecture, extract DRY helpers, enrich error context, add optional tracing, refactor MCP handler boilerplate, and unify test fixtures.

## Principles

- **DRY** — every repeated pattern gets one canonical location
- **Single responsibility** — each module/struct has one reason to change
- **Debuggability** — errors carry context, tracing spans bracket operations
- **Zero-cost when unused** — tracing is behind a feature flag
- **No algorithm changes** — this is structural; all lifecycle, retrieval, and graph logic stays intact

## Architecture

### Coordinator + Sub-Manager Pattern

`AlayaStore` (32 methods, 900-line impl block) is replaced by `Alaya`, a coordinator that owns the SQLite connection and hands out typed sub-manager references:

```rust
let alaya = Alaya::open_in_memory()?;

alaya.episodes().store(&episode)?;
alaya.episodes().by_session("s1")?;

alaya.knowledge().query(&q)?;
alaya.knowledge().learn(nodes)?;
alaya.knowledge().filter(KnowledgeFilter { .. })?;

alaya.lifecycle().consolidate(&provider)?;
alaya.lifecycle().transform()?;
alaya.lifecycle().forget()?;
alaya.lifecycle().dream(&provider, None)?;
alaya.lifecycle().reconcile()?;

alaya.graph().neighbors(node, 2)?;
alaya.graph().strongest_link()?;

alaya.admin().status()?;
alaya.admin().purge(PurgeFilter::All)?;
alaya.admin().categories(None)?;
```

Each sub-manager is a zero-cost wrapper borrowing `&Connection`:

```rust
pub struct Episodes<'a> {
    conn: &'a Connection,
    embedding_provider: Option<&'a dyn EmbeddingProvider>,
}
```

The coordinator:

```rust
pub struct Alaya {
    conn: Connection,
    embedding_provider: Option<Box<dyn EmbeddingProvider>>,
    extraction_provider: Option<Box<dyn ExtractionProvider>>,
    conflict_strategy: ConflictStrategy,
}

impl Alaya {
    pub fn episodes(&self) -> Episodes<'_> { ... }
    pub fn knowledge(&self) -> Knowledge<'_> { ... }
    pub fn lifecycle(&self) -> Lifecycle<'_> { ... }
    pub fn graph(&self) -> Graph<'_> { ... }
    pub fn admin(&self) -> Admin<'_> { ... }
}
```

### Sub-Manager Responsibilities

| Sub-Manager | Methods | Delegates To |
|-------------|---------|-------------|
| `Episodes` | `store`, `by_session`, `unconsolidated` | `store::episodic`, `store::embeddings`, `graph::links` |
| `Knowledge` | `query`, `learn`, `filter`, `breakdown`, `auto_consolidate` | `retrieval::pipeline`, `lifecycle::consolidation`, `store::semantic` |
| `Lifecycle` | `consolidate`, `transform`, `forget`, `perfume`, `dream`, `reconcile`, `conflicts`, `resolve_conflict` | `lifecycle::*` |
| `Graph` | `neighbors`, `strongest_link` | `graph::activation`, `graph::links` |
| `Admin` | `status`, `purge`, `categories`, `subcategories`, `node_category`, `node_content`, `knowledge_breakdown` | `store::*` |

## Module Structure

```
alaya/src/
├── lib.rs              # Alaya coordinator (~150 LOC)
├── db.rs               # Shared DB helpers: now(), transact(), JSON, ResultExt
├── error.rs            # AlayaError with context enrichment
├── schema.rs           # Migrations (unchanged)
├── types.rs            # Public types (unchanged)
├── managers/
│   ├── mod.rs          # Re-exports
│   ├── episodes.rs     # Episodes sub-manager
│   ├── knowledge.rs    # Knowledge sub-manager
│   ├── lifecycle.rs    # Lifecycle sub-manager
│   ├── graph.rs        # Graph sub-manager
│   └── admin.rs        # Admin sub-manager
├── store/              # Low-level SQL CRUD (uses db.rs helpers)
├── lifecycle/          # Core algorithms (unchanged internally)
├── retrieval/          # Pipeline, BM25, vector, fusion, rerank (unchanged)
├── graph/              # Links, activation (unchanged)
├── mcp/                # MCP handlers (refactored with handler helpers)
│   └── handler.rs      # Shared parameter extraction and error formatting
├── provider.rs         # Traits (unchanged)
├── decay.rs            # Decay math (unchanged)
├── testutil.rs         # #[cfg(test)] shared fixtures
├── async_store.rs      # Async wrapper (updated for new API)
└── extraction.rs       # LLM extraction (unchanged)
```

## DRY Extractions — `db.rs`

### Timestamp Helper

Replaces 23 scattered `SystemTime::now()` calls:

```rust
pub(crate) fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
```

### Transaction Wrapper

Replaces 9 `begin_immediate` + `commit` sites in the current lib.rs:

```rust
pub(crate) fn transact<F, T>(conn: &Connection, f: F) -> Result<T>
where
    F: FnOnce(&Transaction) -> Result<T>,
{
    let tx = schema::begin_immediate(conn)?;
    let result = f(&tx)?;
    tx.commit()?;
    Ok(result)
}
```

Sub-manager lifecycle methods become one-liners:

```rust
pub fn transform(&self) -> Result<TransformationReport> {
    db::transact(self.conn, lifecycle::transformation::transform)
}
```

### JSON Field Helpers

Replaces 8+ scattered serde calls in store modules:

```rust
pub(crate) fn to_json<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(Into::into)
}

pub(crate) fn from_json_or_default<T: serde::de::DeserializeOwned + Default>(s: &str) -> T {
    serde_json::from_str(s).unwrap_or_default()
}
```

### Row Mapping

Stays as-is in individual store modules. Each struct has different field types, enum parsing, and JSON patterns. Generic row mapper abstraction would add complexity without meaningful DRY benefit.

## Error Context Enrichment

`AlayaError::Db` changes from a tuple variant to a struct variant with context:

```rust
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AlayaError {
    #[error("database error in {context}: {source}")]
    Db {
        source: rusqlite::Error,
        context: String,
    },
    // ... other variants unchanged
}
```

A `ResultExt` trait adds context at sub-manager boundaries:

```rust
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
```

Context is added at the sub-manager layer (meaningful operation names), not at every SQL call (too noisy).

## Tracing Integration

Optional dependency behind `tracing` feature flag:

```toml
[features]
tracing = ["dep:tracing"]

[dependencies]
tracing = { version = "0.1", optional = true }
```

Instrumented:
- All sub-manager public methods — `#[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]`
- `db::transact` — span wrapping every transaction
- Retrieval pipeline stages — BM25, vector, activation, fusion, rerank

Not instrumented:
- Individual SQL calls (too noisy)
- Internal helpers
- Test code

Zero cost when feature is disabled.

## MCP Handler Refactoring

New `mcp/handler.rs` extracts shared parameter validation and error formatting:

```rust
pub(crate) fn require_str<'a>(params: &'a Value, key: &str) -> std::result::Result<&'a str, String> {
    params.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing required parameter '{key}'"))
}

pub(crate) fn require_u64(params: &Value, key: &str) -> std::result::Result<u64, String> {
    params.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("missing required parameter '{key}'"))
}

pub(crate) fn optional_str<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(|v| v.as_str())
}

pub(crate) fn run<F, T>(server: &McpServer, f: F) -> String
where
    F: FnOnce(&Alaya) -> crate::Result<T>,
    T: serde::Serialize,
{
    match server.with_store(f) {
        Ok(result) => serde_json::to_string_pretty(&result).unwrap_or_default(),
        Err(e) => error_response(&e.to_string()),
    }
}
```

Handlers reduce from ~20 lines to ~8 lines each. `McpServer` updates to use `Alaya` instead of `AlayaStore`.

## Test Utilities — `testutil.rs`

Shared `#[cfg(test)]` module replacing duplicated setup across 12+ test modules:

```rust
pub(crate) mod testutil {
    /// In-memory DB with schema — replaces 30+ open_memory_db().unwrap() calls
    pub fn test_db() -> Connection;

    /// Alaya with mock providers pre-configured
    pub fn test_alaya() -> Alaya;

    /// Episode factory with sensible defaults
    pub fn episode(content: &str) -> NewEpisode;

    /// Episode with custom role and timestamp
    pub fn episode_at(content: &str, role: Role, ts: i64) -> NewEpisode;

    /// Store N episodes, return IDs
    pub fn seed_episodes(alaya: &Alaya, n: usize) -> Vec<EpisodeId>;

    /// Insert semantic node directly into DB
    pub fn insert_semantic_node(conn: &Connection, content: &str, confidence: f32) -> NodeId;
}
```

## What Does NOT Change

- `store/` SQL queries and row mapping — working code
- `lifecycle/` algorithms (consolidation, transformation, reconciliation, forgetting, perfuming) — 100% coverage
- `retrieval/` pipeline stages — working, tested
- `graph/` activation and links logic — working, tested
- `types.rs` — types are well-designed
- `schema.rs` — migrations are stable
- `provider.rs` — traits are clean

## Breaking Changes

- `AlayaStore` renamed to `Alaya`
- Flat methods become sub-manager calls: `store.query()` becomes `alaya.knowledge().query()`
- `AlayaError::Db` changes from `Db(rusqlite::Error)` to `Db { source, context }`
- `AsyncAlayaStore` renamed to `AsyncAlaya`, updated for new API
- Python bindings (`alaya-py`) need updating
- Doc examples all need updating

## Unchanged Files

- `store/episodic.rs`, `store/semantic.rs`, `store/implicit.rs`, `store/categories.rs`, `store/embeddings.rs`, `store/strengths.rs`, `store/conflicts.rs` — internal SQL logic unchanged, only import `db::now()` instead of inline timestamps
- `lifecycle/consolidation.rs`, `lifecycle/transformation.rs`, `lifecycle/reconciliation.rs`, `lifecycle/forgetting.rs`, `lifecycle/perfuming.rs` — algorithms unchanged
- `retrieval/pipeline.rs`, `retrieval/bm25.rs`, `retrieval/vector.rs`, `retrieval/fusion.rs`, `retrieval/rerank.rs` — unchanged
- `graph/links.rs`, `graph/activation.rs` — unchanged
- `decay.rs` — unchanged
- `provider.rs` — unchanged

## Estimated Scope

~25 files touched, ~2000 lines moved/refactored. Net line count roughly neutral — code moves between files and boilerplate is eliminated. No algorithm changes.
