# Conflict Resolution for Alaya

**Date:** 2026-03-22
**Status:** Approved
**Approach:** Conflict Table + Provider Trait

## Overview

Add a conflict resolution system to Alaya's memory engine that detects contradictions between semantic nodes, resolves them using configurable strategies, and maintains an audit trail via soft-archiving. This is the first "unsolved problem" identified in the architecture docs.

## Goals

- Detect contradictory semantic memories (e.g., "user prefers dark mode" vs. "user prefers light mode")
- Resolve conflicts using pluggable strategies: recency-based, confidence-weighted, or manual (agent-decides)
- Preserve superseded memories for audit/history via soft-archiving with Supersedes links
- Integrate as a new `reconcile` lifecycle stage alongside existing `transform`, `forget`, `consolidate`

## Schema & Data Model

### New table: `conflicts`

```sql
CREATE TABLE conflicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    node_a_id       INTEGER NOT NULL REFERENCES semantic_nodes(id),
    node_b_id       INTEGER NOT NULL REFERENCES semantic_nodes(id),
    similarity      REAL    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'detected',
    resolution      TEXT,
    winner_id       INTEGER REFERENCES semantic_nodes(id),
    detected_at     INTEGER NOT NULL,
    resolved_at     INTEGER,
    UNIQUE(node_a_id, node_b_id)
);
```

**Status values:** `detected` | `verified` | `resolved` | `dismissed`

**Resolution values:** `recency` | `confidence` | `manual` | NULL (if unresolved)

### New column on `semantic_nodes`

```sql
ALTER TABLE semantic_nodes ADD COLUMN superseded_by INTEGER REFERENCES semantic_nodes(id);
```

Nodes with `superseded_by IS NOT NULL` are filtered from retrieval and knowledge queries.

### New link type: `Supersedes`

Created in the existing `links` table when a conflict is resolved. Winner -> loser, weight 1.0.

### ReconcileReport

```rust
pub struct ReconcileReport {
    pub conflicts_detected: u32,
    pub conflicts_resolved: u32,
    pub conflicts_pending: u32,
    pub nodes_superseded: u32,
}
```

## Detection Pipeline

Two-phase detection runs during `reconcile()`:

### Phase 1: Heuristic candidate scan

- Query semantic nodes with embeddings that are not superseded
- Compare within same category (or both uncategorized) to avoid O(n^2) over the full graph
- Pairs with cosine similarity above `CONFLICT_SIMILARITY_THRESHOLD` (0.85) become candidates
- Skip pairs already in the `conflicts` table
- Insert candidates with status `detected`

### Phase 2: Optional LLM verification

- If `ExtractionProvider` is configured, send `detected` conflicts to LLM for batch verification
- Verified contradictions -> status `verified`
- Non-contradictions -> status `dismissed`
- Without a provider, all `detected` conflicts are treated as `verified` (heuristic-only mode)

### Constants

```rust
const CONFLICT_SIMILARITY_THRESHOLD: f32 = 0.85;
const LLM_VERIFICATION_BATCH_SIZE: usize = 20;
```

### Efficiency

Category-scoped comparison: O(N^2/K) where N = nodes, K = categories. For 100 nodes in 10 categories, ~50 comparisons per category.

## Resolution Strategies

### ConflictStrategy enum

```rust
pub enum ConflictStrategy {
    /// Most recent node wins. Default.
    Recency,
    /// Higher confidence wins. Ties broken by recency.
    Confidence,
    /// Don't auto-resolve. Surface via conflicts() API.
    Manual,
}
```

### Strategy behavior

**Recency:** Compare `created_at` timestamps of source episodes. Most recent wins. Falls back to node row ID if no source episodes.

**Confidence:** Compare `confidence` fields. Within 0.01 epsilon, fall back to recency.

**Manual:** Skip resolution. Conflicts stay `verified`. Returned by `conflicts()` query API.

### Resolution execution (shared)

1. Set `conflicts.status = 'resolved'`, `winner_id`, `resolution`, `resolved_at`
2. Set `semantic_nodes.superseded_by = winner_id` on loser
3. Set loser `confidence = 0.0`
4. Create `Supersedes` link: winner -> loser, weight 1.0

### Configuration

```rust
impl AlayaStore {
    pub fn set_conflict_strategy(&mut self, strategy: ConflictStrategy) { ... }
}
```

Default: `ConflictStrategy::Recency`. Runtime config (not persisted to DB).

## Public API

### New methods on AlayaStore

```rust
/// Run conflict detection and resolution.
pub fn reconcile(&self) -> Result<ReconcileReport>

/// Query unresolved conflicts (for Manual strategy).
pub fn conflicts(&self) -> Result<Vec<Conflict>>

/// Manually resolve a specific conflict.
pub fn resolve_conflict(&self, conflict_id: ConflictId, winner_id: NodeId) -> Result<()>

/// Configure the resolution strategy (default: Recency).
pub fn set_conflict_strategy(&mut self, strategy: ConflictStrategy)
```

### New public types

```rust
pub struct Conflict {
    pub id: ConflictId,
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub similarity: f32,
    pub status: ConflictStatus,
    pub detected_at: i64,
}

pub enum ConflictStatus {
    Detected,
    Verified,
    Resolved,
    Dismissed,
}
```

### Retrieval pipeline changes

Single WHERE clause addition in `pipeline.rs`: skip nodes where `superseded_by IS NOT NULL`.

### MCP tools

- `reconcile` - runs the reconcile lifecycle stage, returns formatted report
- `conflicts` - lists unresolved conflicts for agent review

No `resolve_conflict` MCP tool. Manual resolution via MCP uses the agent calling `learn` with the corrected fact; supersession happens on next `reconcile` pass.

### AsyncAlayaStore

Standard async wrappers for all new methods via the existing actor pattern.

### Integration with existing lifecycle

`reconcile()` is independent and opt-in. Not called by `maintain` MCP tool or auto-maintenance in `remember`. Users invoke explicitly.

## File Layout

### New files

| File | Purpose |
|------|---------|
| `src/lifecycle/reconciliation.rs` | Core detection + resolution logic, constants, unit tests |
| `src/store/conflicts.rs` | CRUD for conflicts table |

### Modified files

| File | Change |
|------|--------|
| `src/schema.rs` | Add conflicts table, superseded_by column migration |
| `src/lib.rs` | Add public methods, ConflictStrategy field |
| `src/types.rs` | Add Conflict, ConflictId, ConflictStatus, ConflictStrategy, ReconcileReport |
| `src/async_store.rs` | Async wrappers + actor Request/Reply variants |
| `src/store/semantic.rs` | Add superseded_by filter to knowledge queries |
| `src/retrieval/pipeline.rs` | Filter superseded nodes from candidates |
| `src/mcp/mod.rs` | Add reconcile and conflicts tool handlers |
| `src/mcp/lifecycle.rs` | Add handle_reconcile and handle_conflicts |
| `src/lifecycle/mod.rs` | Add pub mod reconciliation |
| `src/store/mod.rs` | Add pub mod conflicts |

### Untouched files

extraction.rs, graph/, store/episodic.rs, store/implicit.rs, store/embeddings.rs, store/strengths.rs, retrieval/bm25.rs, retrieval/vector.rs, retrieval/fusion.rs, retrieval/rerank.rs, mcp/memory.rs, mcp/preferences.rs, mcp/query.rs, mcp/import.rs, mcp/status.rs, mcp/serialization.rs, mcp/validation.rs.

## Testing Strategy

### Unit tests (lifecycle/reconciliation.rs)

- Detection with two contradictory nodes -> conflict created
- Detection skips existing pairs
- Category-scoped detection ignores cross-category pairs
- Each resolution strategy: recency wins, confidence wins, manual leaves unresolved
- Confidence tie-breaking falls back to recency
- Resolution creates Supersedes link and sets superseded_by
- Superseded node has confidence 0.0
- Empty store -> empty report
- No conflicts when all nodes are dissimilar

### Integration tests (tests/)

- Full lifecycle: store -> learn -> reconcile -> superseded excluded from queries
- Manual strategy: reconcile -> conflicts() -> resolve_conflict() -> empty
- Idempotent: second reconcile with no changes -> 0 new detections
- Interaction with transform: reconcile after transform preserves categories

### MCP handler tests (mcp/)

- reconcile tool returns formatted report
- conflicts tool returns formatted list
- Both handle DB errors gracefully

### LLM verification

Tested via MockExtractionProvider. HTTP implementation excluded from coverage (existing pattern).

**Coverage target:** 99%+ (matching existing codebase standard).
