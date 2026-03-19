# 100% Test Coverage & Radical Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve 100% line and branch test coverage, then refactor the codebase for maximum maintainability, debuggability, and DRY compliance — all while preserving the public API exactly.

**Architecture:** Four sequential phases: (1) coverage infrastructure, (2) test writing against current code, (3) refactoring with test safety net, (4) filling new coverage gaps. The refactoring extracts a `Decay` trait hierarchy, decomposes the 3,415-line `mcp.rs` monolith into 9 domain/layer modules, adds optional `tracing` instrumentation, and consolidates all DRY violations.

**Tech Stack:** Rust 1.85+, cargo-tarpaulin (coverage), tracing (instrumentation), proptest (property tests), rusqlite, rmcp, thiserror

**Spec:** `docs/superpowers/specs/2026-03-18-100pct-coverage-and-refactor-design.md`

---

## File Structure

### New files to create:
```
tarpaulin.toml                          — Coverage configuration
src/decay.rs                            — Decay/SqlDecay traits + ExponentialDecay, MultiplicativeDecay
src/mcp/mod.rs                          — MCP server struct, re-exports, dispatcher
src/mcp/memory.rs                       — remember, recall tool handlers
src/mcp/lifecycle.rs                    — maintain, purge tool handlers
src/mcp/preferences.rs                 — learn, preferences tool handlers
src/mcp/query.rs                        — knowledge, categories, neighbors, node_category handlers
src/mcp/import.rs                       — import_claude_mem, import_claude_code handlers
src/mcp/status.rs                       — status tool handler
src/mcp/validation.rs                   — Shared param extraction helpers
src/mcp/serialization.rs               — Response formatting, error-to-JSON
tests/common/mod.rs                     — Shared test fixtures and builders
tests/memory.rs                         — Memory integration tests (from integration.rs)
tests/lifecycle.rs                      — Lifecycle integration tests (from integration.rs)
tests/retrieval.rs                      — Retrieval integration tests (from integration.rs)
```

### Files to modify:
```
Cargo.toml                              — Add tracing dep, tracing feature
.github/workflows/ci.yml               — Add coverage job
src/lib.rs                              — Add decay module, test_helpers, tracing cfg_attr
src/graph/links.rs                      — Replace decay_links with Decay trait call
src/store/implicit.rs                   — Replace decay_preferences with Decay trait call
src/store/strengths.rs                  — Replace decay_all_retrieval with Decay trait call
src/retrieval/rerank.rs                 — Replace recency_decay with Decay trait call
src/lifecycle/transformation.rs         — Use Decay trait at call site
src/lifecycle/forgetting.rs             — Use Decay trait at call site
src/bin/alaya-mcp.rs                    — Add tracing-subscriber setup
```

### Files to delete (after migration):
```
src/mcp.rs                              — Replaced by src/mcp/ directory
tests/integration.rs                    — Split into tests/memory.rs, tests/lifecycle.rs, tests/retrieval.rs
tests/mcp_tools.rs                      — Renamed to tests/mcp.rs
```

---

## Phase 1: Coverage Infrastructure

### Task 1: Add tarpaulin configuration

**Files:**
- Create: `tarpaulin.toml`

- [ ] **Step 1: Create tarpaulin.toml**

```toml
[default]
# Measure both line and branch coverage
branch = true
# All features to cover all code paths
features = "mcp,llm"
# Output HTML for local dev, JSON for CI
out = ["html", "json"]
# Exclude test code from coverage measurement
exclude-files = ["tests/*", "src/bin/*"]
# Timeout per test (seconds)
test-timeout = 120
# Run all test types
run-types = ["tests", "doctests"]
```

- [ ] **Step 2: Verify tarpaulin runs locally**

Run: `cargo install cargo-tarpaulin && cargo tarpaulin --config tarpaulin.toml --skip-clean 2>&1 | tail -5`
Expected: Coverage report with percentages (likely 60-80% baseline)

- [ ] **Step 3: Commit**

```bash
git add tarpaulin.toml
git commit -m "chore: add tarpaulin coverage configuration"
```

### Task 2: Add coverage CI job

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add coverage job to ci.yml**

Add this job after the existing `deny` job:

```yaml
  coverage:
    name: Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4.3.1

      - uses: dtolnay/rust-toolchain@efa25f7f19611383d5b0ccf2d1c8914531636bf9 # master
        with:
          toolchain: stable

      - uses: Swatinem/rust-cache@779680da715d629ac1d338a641029a2f4372abb5 # v2.8.2

      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Run coverage
        run: cargo tarpaulin --config tarpaulin.toml --skip-clean

      - name: Check coverage threshold
        run: |
          # Parse coverage from the JSON already produced above
          COVERAGE=$(python3 -c "import json; d=json.load(open('tarpaulin-report.json')); print(f\"{d.get('coverage',0):.2f}\")" 2>/dev/null || echo "0")
          echo "Coverage: ${COVERAGE}%"
```

Note: Initially report-only. After Phase 2, add threshold enforcement.

- [ ] **Step 2: Verify CI syntax is valid**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "Valid YAML"`
Expected: "Valid YAML"

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add tarpaulin coverage reporting job"
```

### Task 3: Create shared test fixtures

**Files:**
- Create: `tests/common/mod.rs`

- [ ] **Step 1: Write shared test fixtures**

```rust
use alaya::{AlayaStore, EpisodeContext, NewEpisode, NewSemanticNode, Role, SemanticType};

/// Create an empty in-memory store for testing.
pub fn empty_store() -> AlayaStore {
    AlayaStore::open_in_memory().unwrap()
}

/// Create a store populated with sample episodes.
pub fn populated_store() -> AlayaStore {
    let store = empty_store();
    for i in 0..5 {
        store
            .store_episode(&make_episode(
                &format!("Test message {i}"),
                "user",
                "session-1",
                1700000000 + i * 60,
            ))
            .unwrap();
    }
    store
}

/// Create a store with a specific number of episodes.
pub fn store_with_episodes(n: i64) -> AlayaStore {
    let store = empty_store();
    for i in 0..n {
        store
            .store_episode(&make_episode(
                &format!("Episode {i}"),
                "user",
                "session-1",
                1700000000 + i * 60,
            ))
            .unwrap();
    }
    store
}

/// Build a NewEpisode with sensible defaults.
pub fn make_episode(content: &str, role: &str, session_id: &str, timestamp: i64) -> NewEpisode {
    let role = match role {
        "assistant" => Role::Assistant,
        "system" => Role::System,
        _ => Role::User,
    };
    NewEpisode {
        content: content.to_string(),
        role,
        session_id: session_id.to_string(),
        timestamp,
        context: EpisodeContext::default(),
        embedding: None,
    }
}

/// Build a NewSemanticNode with sensible defaults.
pub fn make_semantic_node(content: &str, node_type: SemanticType) -> NewSemanticNode {
    NewSemanticNode {
        content: content.to_string(),
        node_type,
        source_episodes: vec![],
        embedding: None,
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo test --no-run --features "mcp llm" 2>&1 | tail -3`
Expected: Compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add tests/common/mod.rs
git commit -m "test: add shared test fixtures module"
```

### Task 4: Migrate integration test files

**Files:**
- Modify: `tests/integration.rs` → split into `tests/memory.rs`, `tests/lifecycle.rs`, `tests/retrieval.rs`
- Modify: `tests/mcp_tools.rs` → rename to `tests/mcp.rs`

- [ ] **Step 1: Read tests/integration.rs to understand test groupings**

Read the full file and categorize each test function by domain (memory, lifecycle, retrieval).

- [ ] **Step 2: Create tests/memory.rs with memory-related tests**

Move all tests related to `store_episode`, `query`, `store_semantic_node`, `node`, `import_*` into `tests/memory.rs`. Add `mod common;` at the top.

- [ ] **Step 3: Create tests/lifecycle.rs with lifecycle-related tests**

Move all tests related to `consolidate`, `learn`, `perfume`, `transform`, `forget`, `purge` into `tests/lifecycle.rs`. Add `mod common;` at the top.

- [ ] **Step 4: Create tests/retrieval.rs with retrieval-related tests**

Move all tests related to `query` pipeline behavior, result ordering, fusion scoring into `tests/retrieval.rs`. Add `mod common;` at the top.

- [ ] **Step 5: Rename tests/mcp_tools.rs to tests/mcp.rs**

```bash
git mv tests/mcp_tools.rs tests/mcp.rs
```

- [ ] **Step 6: Delete tests/integration.rs**

```bash
git rm tests/integration.rs
```

- [ ] **Step 7: Run full test suite**

Run: `cargo test --features "mcp llm" 2>&1 | tail -10`
Expected: All tests pass (same count as before migration)

- [ ] **Step 8: Commit**

```bash
git add tests/
git commit -m "test: reorganize integration tests by domain"
```

---

## Phase 2: Test Writing (Against Current Code)

### Task 5: Add unit tests for retrieval/vector.rs (100% coverage)

**Files:**
- Modify: `src/retrieval/vector.rs`
- Test: `src/retrieval/vector.rs` (inline `#[cfg(test)]`)

The file is 30 lines with 1 test. Current code: `search_vector` wraps `embeddings::search_by_vector` and casts `f32` to `f64`.

- [ ] **Step 1: Write test for vector search with stored embeddings**

Add to existing `mod tests` in `src/retrieval/vector.rs`:

```rust
#[test]
fn test_vector_search_with_results() {
    let conn = open_memory_db().unwrap();
    // Store an embedding for a semantic node
    crate::store::embeddings::store_embedding(
        &conn,
        &NodeRef::Semantic(crate::types::NodeId(1)),
        &[1.0, 0.0, 0.0],
    )
    .unwrap();
    // First, need a semantic node in the DB
    crate::store::semantic::store_node(
        &conn,
        &crate::types::NewSemanticNode {
            content: "test".to_string(),
            node_type: crate::types::SemanticType::Fact,
            source_episodes: vec![],
            embedding: None,
        },
    )
    .unwrap();
    crate::store::embeddings::store_embedding(
        &conn,
        &NodeRef::Semantic(crate::types::NodeId(1)),
        &[1.0, 0.0, 0.0],
    )
    .unwrap();

    let results = search_vector(&conn, &[1.0, 0.0, 0.0], 10).unwrap();
    assert!(!results.is_empty());
    // Verify f32->f64 cast works (similarity should be ~1.0 for identical vectors)
    assert!(results[0].1 > 0.99);
}

#[test]
fn test_vector_search_limit() {
    let conn = open_memory_db().unwrap();
    let results = search_vector(&conn, &[1.0, 0.0], 0).unwrap();
    assert!(results.is_empty());
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test --lib retrieval::vector -- --nocapture 2>&1`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/vector.rs
git commit -m "test: expand vector search coverage"
```

### Task 6: Add unit tests for store/embeddings.rs

**Files:**
- Modify: `src/store/embeddings.rs`

- [ ] **Step 1: Read src/store/embeddings.rs fully to identify all untested branches**

- [ ] **Step 2: Write tests for serialization/deserialization edge cases**

Add tests for:
- Empty embedding vector
- Single-element vector
- Large vector (1000+ dimensions)
- Cosine similarity: identical vectors (→ 1.0), orthogonal (→ 0.0), opposite (→ -1.0), zero vector handling
- `store_embedding` then `search_by_vector` round-trip
- `search_by_vector` with `namespace` filter (Some and None)
- Dimension mismatch between query and stored vector

- [ ] **Step 3: Run tests to verify pass**

Run: `cargo test --lib store::embeddings -- --nocapture 2>&1`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/store/embeddings.rs
git commit -m "test: full coverage for embeddings store"
```

### Task 7: Add unit tests for retrieval/pipeline.rs

**Files:**
- Modify: `src/retrieval/pipeline.rs`

- [ ] **Step 1: Read src/retrieval/pipeline.rs fully to identify all branches**

- [ ] **Step 2: Write tests for each pipeline path**

Test cases needed:
- Query with no results (empty store)
- Query with BM25-only results (no embedding provider)
- Query that triggers vector search (when embeddings exist)
- Query with multiple results — verify RRF fusion ordering
- Query with `max_results` limit
- Query with context (topics, entities, sentiment) — verify reranking adjusts scores
- Query that exercises graph boost (co-retrieval links exist)
- Error propagation from BM25 failure
- Error propagation from vector search failure

- [ ] **Step 3: Run tests**

Run: `cargo test --lib retrieval::pipeline -- --nocapture 2>&1`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/retrieval/pipeline.rs
git commit -m "test: full coverage for retrieval pipeline"
```

### Task 8: Add unit tests for mcp.rs formatting helpers

**Files:**
- Modify: `src/mcp.rs`

The formatting helper functions (`format_preferences`, `format_categories`, `format_neighbors`, etc.) are standalone functions that can be unit-tested directly.

- [ ] **Step 1: Read all formatting helpers in src/mcp.rs (lines ~260-520)**

- [ ] **Step 2: Write unit tests for each formatting helper**

For each `format_*` function, test:
- Empty input slice
- Single item
- Multiple items
- Edge cases (long strings, special characters, zero values)

- [ ] **Step 3: Run tests**

Run: `cargo test --features mcp --lib mcp -- --nocapture 2>&1`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/mcp.rs
git commit -m "test: full coverage for MCP formatting helpers"
```

### Task 9: Add unit tests for mcp.rs tool handlers

**Files:**
- Modify: `src/mcp.rs`

- [ ] **Step 1: Read each tool handler method in src/mcp.rs**

All 13 handlers: remember, recall, status, preferences, knowledge, maintain, categories, neighbors, node_category, learn, purge, import_claude_mem, import_claude_code

- [ ] **Step 2: Identify untested branches per handler**

For each handler, identify:
- Success path
- Error paths (invalid params, store errors)
- Edge cases (empty results, boundary values)
- `remember`: invalid role string, auto-consolidation threshold (10 episodes), extraction provider paths
- `recall`: zero max_results, empty query
- `learn`: empty nodes array, malformed node entries, all semantic types
- `purge`: each PurgeFilter variant

- [ ] **Step 3: Write tests for all untested branches**

Add tests to the existing `#[cfg(test)] mod tests` in `src/mcp.rs`. Use the existing `make_server()` helper pattern.

- [ ] **Step 4: Run tests**

Run: `cargo test --features mcp --lib mcp -- --nocapture 2>&1`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/mcp.rs
git commit -m "test: full branch coverage for MCP tool handlers"
```

### Task 10: Add unit tests for remaining modules

**Files:**
- Modify: All modules with coverage gaps (identified by tarpaulin)

- [ ] **Step 1: Run tarpaulin to identify remaining gaps**

Run: `cargo tarpaulin --config tarpaulin.toml --skip-clean --out html 2>&1 | tail -20`

- [ ] **Step 2: For each module with < 100% coverage, add tests for uncovered lines/branches**

Work through modules in order of coverage gap size. Key modules to check:
- `src/schema.rs` (488 LOC) — schema creation, migration paths
- `src/types.rs` (572 LOC) — type constructors, Display impls, edge cases
- `src/provider.rs` (280 LOC) — NoOpProvider, MockProviders, provider trait methods
- `src/lifecycle/transformation.rs` (1227 LOC) — deduplication, merging, decay paths
- `src/lifecycle/consolidation.rs` (577 LOC) — consolidation with/without provider
- `src/store/categories.rs` (600 LOC) — category formation, stability, hierarchy
- `src/store/semantic.rs` (387 LOC) — semantic node CRUD
- `src/store/episodic.rs` (255 LOC) — episode storage, retrieval
- `src/store/implicit.rs` (381 LOC) — preference learning, decay
- `src/graph/links.rs` (416 LOC) — link creation, decay, pruning
- `src/graph/activation.rs` (166 LOC) — spreading activation
- `src/retrieval/bm25.rs` (150 LOC) — BM25 scoring
- `src/retrieval/fusion.rs` (116 LOC) — RRF fusion
- `src/retrieval/rerank.rs` (268 LOC) — contextual reranking
- `src/lifecycle/forgetting.rs` (192 LOC) — forgetting thresholds, archival
- `src/lifecycle/perfuming.rs` (154 LOC) — preference extraction
- `src/extraction.rs` (433 LOC, llm feature) — LLM extraction provider
- `src/error.rs` (62 LOC) — likely already well-covered

- [ ] **Step 3: Run tarpaulin and verify 100%**

Run: `cargo tarpaulin --config tarpaulin.toml --skip-clean 2>&1 | grep -E "^[0-9]"`
Expected: 100.00% line coverage, 100.00% branch coverage

- [ ] **Step 4: Commit**

```bash
git add src/
git commit -m "test: achieve 100% line and branch coverage"
```

### Task 11: Add property-based tests for critical invariants

**Files:**
- Modify: `src/graph/links.rs`, `src/retrieval/rerank.rs`, `src/retrieval/fusion.rs`

- [ ] **Step 1: Add proptest for decay invariants**

Expand existing `prop_decay_links_weight_bounded` pattern:

```rust
proptest! {
    #[test]
    fn prop_recency_decay_bounded(ts in 0i64..2_000_000_000, now in 0i64..2_000_000_000) {
        let result = recency_decay(ts, now);
        prop_assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn prop_recency_decay_monotonic(age1 in 0i64..1_000_000, age2 in 0i64..1_000_000) {
        let now = 2_000_000_000i64;
        let d1 = recency_decay(now - age1, now);
        let d2 = recency_decay(now - age2, now);
        // More recent should have higher decay score
        if age1 < age2 { prop_assert!(d1 >= d2); }
    }
}
```

- [ ] **Step 2: Add proptest for fusion score bounds**

```rust
proptest! {
    #[test]
    fn prop_rrf_scores_bounded(k in 1u32..100, rank in 0usize..1000) {
        let score = 1.0 / (k as f64 + rank as f64);
        prop_assert!(score > 0.0 && score <= 1.0);
    }
}
```

- [ ] **Step 3: Run all proptests**

Run: `cargo test --features "mcp llm" -- prop_ --nocapture 2>&1`
Expected: All property tests pass

- [ ] **Step 4: Commit**

```bash
git add src/
git commit -m "test: add property-based tests for decay and fusion invariants"
```

---

## Phase 3: Refactoring

### Task 12: Create Decay trait module

**Files:**
- Create: `src/decay.rs`
- Modify: `src/lib.rs` (add `pub(crate) mod decay;`)

- [ ] **Step 1: Write failing tests for Decay trait**

Create `src/decay.rs` with tests first:

```rust
use crate::error::Result;
use rusqlite::Connection;

/// Strategy for computing temporal decay factors.
pub trait Decay {
    /// Compute a multiplicative decay factor (0.0..=1.0) for the given elapsed time.
    fn factor(&self, elapsed_secs: i64) -> f64;
}

/// Multiplicative decay: multiplies by a fixed factor each sweep.
/// Used by link decay and retrieval strength decay.
pub struct MultiplicativeDecay {
    pub factor: f64,
}

/// Exponential decay: exp(-0.693 * elapsed / half_life).
/// For Rust contexts, uses exact f64 exponential.
/// Note: decay_preferences uses its own SQL with a time-conditional WHERE clause
/// that cannot be expressed generically, so it calls Decay::factor() only.
pub struct ExponentialDecay {
    pub half_life_secs: i64,
}

/// Apply multiplicative decay to a single column via SQL UPDATE.
/// Only used where the SQL pattern is simple: `SET col = col * factor WHERE col > 0.01`.
/// decay_preferences has a complex WHERE clause and is NOT a candidate for this helper.
pub fn apply_multiplicative_sql(
    conn: &Connection,
    table: &str,
    column: &str,
    factor: f64,
) -> Result<u64> {
    let sql = format!(
        "UPDATE {table} SET {column} = {column} * ?1 WHERE {column} > 0.01"
    );
    let changed = conn.execute(&sql, [factor])?;
    Ok(changed as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    // --- MultiplicativeDecay ---

    #[test]
    fn multiplicative_factor_returns_stored_factor() {
        let d = MultiplicativeDecay { factor: 0.9 };
        assert!((d.factor(1000) - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn multiplicative_factor_ignores_elapsed() {
        let d = MultiplicativeDecay { factor: 0.85 };
        assert_eq!(d.factor(0), d.factor(999_999));
    }

    #[test]
    fn apply_multiplicative_sql_updates_rows() {
        let conn = open_memory_db().unwrap();
        // Insert test data into node_strengths table
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength)
             VALUES ('semantic', 1, 1.0, 0.8)",
            [],
        ).unwrap();
        let changed = apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.9).unwrap();
        assert!(changed > 0);
    }

    #[test]
    fn apply_multiplicative_sql_skips_below_threshold() {
        let conn = open_memory_db().unwrap();
        conn.execute(
            "INSERT INTO node_strengths (node_type, node_id, storage_strength, retrieval_strength)
             VALUES ('semantic', 1, 1.0, 0.005)",
            [],
        ).unwrap();
        let changed = apply_multiplicative_sql(&conn, "node_strengths", "retrieval_strength", 0.9).unwrap();
        assert_eq!(changed, 0);
    }

    // --- ExponentialDecay ---

    #[test]
    fn exponential_factor_zero_elapsed() {
        let d = ExponentialDecay { half_life_secs: 86400 };
        assert!((d.factor(0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn exponential_factor_at_half_life() {
        let d = ExponentialDecay { half_life_secs: 86400 };
        let f = d.factor(86400);
        assert!((f - 0.5).abs() < 0.01);
    }

    #[test]
    fn exponential_factor_large_elapsed() {
        let d = ExponentialDecay { half_life_secs: 86400 };
        let f = d.factor(86400 * 100);
        assert!(f < 0.001);
        assert!(f >= 0.0);
    }

    #[test]
    fn exponential_factor_negative_elapsed() {
        let d = ExponentialDecay { half_life_secs: 86400 };
        let f = d.factor(-1000);
        // Negative elapsed should clamp to 0 elapsed → factor = 1.0
        assert!((f - 1.0).abs() < f64::EPSILON);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib decay -- --nocapture 2>&1`
Expected: FAIL — `Decay` and `SqlDecay` not implemented yet

- [ ] **Step 3: Implement Decay and SqlDecay traits**

Add implementations below the struct definitions in `src/decay.rs`:

```rust
impl Decay for MultiplicativeDecay {
    fn factor(&self, _elapsed_secs: i64) -> f64 {
        self.factor
    }
}

impl Decay for ExponentialDecay {
    fn factor(&self, elapsed_secs: i64) -> f64 {
        let elapsed = elapsed_secs.max(0) as f64;
        (-0.693 * elapsed / self.half_life_secs as f64).exp()
    }
}
```

- [ ] **Step 4: Add `pub(crate) mod decay;` to src/lib.rs**

Add after `pub(crate) mod types;` (line 39) in `src/lib.rs`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib decay -- --nocapture 2>&1`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/decay.rs src/lib.rs
git commit -m "feat: add Decay/SqlDecay trait hierarchy"
```

### Task 13: Migrate decay call sites to use Decay trait

**Files:**
- Modify: `src/graph/links.rs:93` — `decay_links()`
- Modify: `src/store/implicit.rs:135` — `decay_preferences()`
- Modify: `src/store/strengths.rs:88` — `decay_all_retrieval()`
- Modify: `src/retrieval/rerank.rs:38` — `recency_decay()`

- [ ] **Step 1: Run full test suite before changes (baseline)**

Run: `cargo test --features "mcp llm" 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 2: Refactor decay_links in graph/links.rs**

`decay_links` updates two columns in one UPDATE — it cannot use the single-column `apply_multiplicative_sql` helper. Keep its SQL but document the relationship:

```rust
pub fn decay_links(conn: &Connection, decay_factor: f32) -> Result<u64> {
    // MultiplicativeDecay pattern applied to two columns simultaneously.
    // Cannot use apply_multiplicative_sql (single-column helper).
    let changed = conn.execute(
        "UPDATE links SET
            forward_weight = forward_weight * ?1,
            backward_weight = backward_weight * ?1
         WHERE forward_weight > 0.01 OR backward_weight > 0.01",
        [decay_factor],
    )?;
    Ok(changed as u64)
}
```

No functional change — just documenting the decay pattern relationship.

- [ ] **Step 3: Refactor decay_preferences in store/implicit.rs**

`decay_preferences` has a time-conditional WHERE clause (`WHERE (?1 - last_reinforced) > ?2`) that cannot be expressed generically. Keep its SQL but document the decay pattern:

```rust
pub fn decay_preferences(conn: &Connection, now: i64, half_life_secs: i64) -> Result<u64> {
    // ExponentialDecay pattern, but with time-conditional WHERE clause.
    // SQLite lacks exp(), so uses 0.95 linear approximation per sweep.
    // Cannot use apply_multiplicative_sql (needs custom WHERE).
    let changed = conn.execute(
        "UPDATE preferences SET confidence = confidence * 0.95
         WHERE (?1 - last_reinforced) > ?2 AND confidence > 0.01",
        rusqlite::params![now, half_life_secs],
    )?;
    Ok(changed as u64)
}
```

- [ ] **Step 4: Refactor decay_all_retrieval in store/strengths.rs**

This is a clean candidate for the shared helper:

```rust
pub fn decay_all_retrieval(conn: &Connection, decay_factor: f32) -> Result<u64> {
    crate::decay::apply_multiplicative_sql(
        conn, "node_strengths", "retrieval_strength", decay_factor as f64
    )
}
```

- [ ] **Step 5: Refactor recency_decay in retrieval/rerank.rs**

```rust
fn recency_decay(timestamp: i64, now: i64) -> f64 {
    use crate::decay::Decay;
    // 30-day half-life for recency scoring
    let decay = crate::decay::ExponentialDecay {
        half_life_secs: 30 * 86400,
    };
    let elapsed = (now - timestamp).max(0);
    decay.factor(elapsed)
}
```

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `cargo test --features "mcp llm" 2>&1 | tail -5`
Expected: All tests pass with same count as baseline

- [ ] **Step 7: Commit**

```bash
git add src/graph/links.rs src/store/implicit.rs src/store/strengths.rs src/retrieval/rerank.rs
git commit -m "refactor: migrate decay call sites to Decay trait"
```

### Task 14: Decompose mcp.rs — Extract validation and serialization

**Files:**
- Create: `src/mcp/validation.rs`
- Create: `src/mcp/serialization.rs`

- [ ] **Step 1: Create src/mcp/ directory**

```bash
mkdir -p src/mcp
```

- [ ] **Step 2: Read src/mcp.rs to identify all param extraction patterns**

Look for repeated patterns like:
- `params.field.as_deref()`, `params.field.unwrap_or(default)`
- Role string → `Role` enum parsing
- Timestamp generation (`SystemTime::now()...`)

- [ ] **Step 2: Create src/mcp/validation.rs**

Extract common validation helpers:

```rust
use crate::Role;

/// Parse a role string into the Role enum.
pub fn parse_role(role: &str) -> Result<Role, String> {
    match role.to_lowercase().as_str() {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "system" => Ok(Role::System),
        _ => Err(format!(
            "invalid role '{}'. Use: user, assistant, system",
            role
        )),
    }
}

/// Get the current Unix timestamp.
pub fn now_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
```

- [ ] **Step 3: Create src/mcp/serialization.rs**

Move all `format_*` functions from `mcp.rs` into this file:

```rust
use crate::types::{Category, MemoryStatus, NodeRef, Preference};

pub fn format_preferences(prefs: &[Preference]) -> String { /* ... */ }
pub fn format_categories(cats: &[Category]) -> String { /* ... */ }
pub fn format_neighbors(neighbors: &[(NodeRef, f32)]) -> String { /* ... */ }
// ... all format_* functions
```

- [ ] **Step 4: Write tests for validation.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_role_user() { assert!(matches!(parse_role("user"), Ok(Role::User))); }
    #[test]
    fn parse_role_case_insensitive() { assert!(matches!(parse_role("USER"), Ok(Role::User))); }
    #[test]
    fn parse_role_invalid() { assert!(parse_role("invalid").is_err()); }
    #[test]
    fn now_timestamp_reasonable() {
        let ts = now_timestamp();
        assert!(ts > 1_700_000_000); // After 2023
    }
}
```

- [ ] **Step 5: Verify compilation**

Run: `cargo test --features mcp --no-run 2>&1 | tail -3`
Expected: Compiles successfully

- [ ] **Step 6: Commit**

```bash
git add src/mcp/validation.rs src/mcp/serialization.rs
git commit -m "refactor: extract MCP validation and serialization modules"
```

### Task 15: Decompose mcp.rs — Extract domain handler modules

**Files:**
- Create: `src/mcp/mod.rs`
- Create: `src/mcp/memory.rs`
- Create: `src/mcp/lifecycle.rs`
- Create: `src/mcp/preferences.rs`
- Create: `src/mcp/query.rs`
- Create: `src/mcp/import.rs`
- Create: `src/mcp/status.rs`
- Delete: `src/mcp.rs`

This is the largest refactoring task. The key constraint is that `rmcp`'s `#[tool(tool_box)]` macro expects all `#[tool]` methods on a single `impl` block. We may need to keep the tool method signatures in `mod.rs` as thin wrappers that delegate to domain modules, or use a different decomposition strategy based on how `rmcp` works.

- [ ] **Step 1: Investigate rmcp tool_box macro constraints**

Read the rmcp docs/source to understand:
- Can `#[tool]` methods be split across multiple `impl` blocks?
- Can tool methods delegate to functions in other modules?
- What is the minimum that must stay in the `#[tool(tool_box)] impl` block?

- [ ] **Step 2: Design the decomposition based on rmcp constraints**

If rmcp requires a single `#[tool(tool_box)] impl`:
- `mod.rs` keeps the `AlayaMcp` struct, `#[tool(tool_box)] impl`, and thin wrapper methods
- Each wrapper calls into domain modules: `memory::handle_remember(...)`, `lifecycle::handle_maintain(...)`, etc.
- All param types stay in `mod.rs` (rmcp needs them in scope)

If rmcp supports splitting:
- Each domain module gets its own `#[tool]` methods
- `mod.rs` just re-exports and wires them together

- [ ] **Step 3: Create src/mcp/ directory and mod.rs**

```bash
mkdir -p src/mcp
```

Move the `AlayaMcp` struct definition, `with_store` helper, param types, and `#[tool(tool_box)] impl` to `src/mcp/mod.rs`.

- [ ] **Step 4: Create domain handler modules**

For each domain module, move the handler logic:

`src/mcp/memory.rs`:
```rust
// Handler logic for remember and recall
// Called from mod.rs #[tool] methods
```

`src/mcp/lifecycle.rs`:
```rust
// Handler logic for maintain and purge
```

`src/mcp/preferences.rs`:
```rust
// Handler logic for learn and preferences
```

`src/mcp/query.rs`:
```rust
// Handler logic for knowledge, categories, neighbors, node_category
```

`src/mcp/import.rs`:
```rust
// Handler logic for import_claude_mem, import_claude_code
// Including parse_claude_code_jsonl helper
```

`src/mcp/status.rs`:
```rust
// Handler logic for status
```

- [ ] **Step 5: Update src/lib.rs module declaration**

Change `pub mod mcp;` — this now resolves to `src/mcp/mod.rs` instead of `src/mcp.rs`.

- [ ] **Step 6: Delete src/mcp.rs**

```bash
git rm src/mcp.rs
```

- [ ] **Step 7: Run full test suite**

Run: `cargo test --features "mcp llm" 2>&1 | tail -10`
Expected: All tests pass with same count

- [ ] **Step 8: Verify no public API changes**

Run: `cargo doc --features "mcp llm" --no-deps 2>&1 | tail -5`
Expected: Doc generation succeeds, same public API surface

- [ ] **Step 9: Commit**

```bash
git add src/mcp/ src/lib.rs
git rm src/mcp.rs
git commit -m "refactor: decompose mcp.rs monolith into 9 domain/layer modules"
```

### Task 16: Add tracing instrumentation

**Files:**
- Modify: `Cargo.toml` — add `tracing` optional dep and feature
- Modify: `src/lib.rs` — add `#[cfg_attr]` instrumentation on AlayaStore methods
- Modify: `src/retrieval/pipeline.rs` — add span hierarchy for pipeline stages
- Modify: `src/lifecycle/*.rs` — add spans for lifecycle operations
- Modify: `src/bin/alaya-mcp.rs` — add tracing-subscriber setup

- [ ] **Step 1: Add tracing dependencies to Cargo.toml**

```toml
# In [features]
tracing = ["dep:tracing"]

# In [dependencies]
tracing = { version = "0.1", optional = true }

# In [dev-dependencies] (for tests)
tracing-subscriber = { version = "0.3", features = ["fmt"] }
```

The `tracing` feature is independent — consumers opt in separately. The `alaya-mcp` binary enables it via `required-features`:

```toml
# In [[bin]] section, update:
required-features = ["mcp", "tracing"]
```

- [ ] **Step 2: Add conditional tracing macro helper to lib.rs**

```rust
/// Conditional tracing: no-op when tracing feature is disabled.
#[cfg(feature = "tracing")]
macro_rules! trace_span {
    ($name:expr) => { tracing::info_span!($name) };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_span {
    ($name:expr) => { /* no-op */ };
}
```

- [ ] **Step 3: Instrument AlayaStore public methods**

Add `#[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]` to all public methods on `AlayaStore`:
- `store_episode`, `query`, `store_semantic_node`, `node`, `knowledge`, `categories`, `preferences`, `node_category`, `neighbors`, `strongest_link`, `consolidate`, `learn`, `perfume`, `transform`, `forget`, `status`, `purge`, `import_claude_mem`, `import_claude_code`

- [ ] **Step 4: Add pipeline spans in retrieval/pipeline.rs**

```rust
#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn};

// In execute_query:
#[cfg(feature = "tracing")]
debug!(query = %query.text, max_results = query.max_results, "executing retrieval pipeline");

// After BM25 stage:
#[cfg(feature = "tracing")]
trace!(bm25_results = bm25_results.len(), "BM25 search complete");

// After vector stage:
#[cfg(feature = "tracing")]
trace!(vector_results = vector_results.len(), "vector search complete");

// After fusion:
#[cfg(feature = "tracing")]
trace!(fused_results = fused.len(), "RRF fusion complete");
```

- [ ] **Step 5: Update alaya-mcp binary with tracing-subscriber**

Replace `eprintln!` calls with tracing and add subscriber setup:

```rust
// In main():
#[cfg(feature = "tracing")]
{
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("alaya=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();
}
```

- [ ] **Step 6: Run tests with and without tracing feature**

Run: `cargo test --features "mcp llm" 2>&1 | tail -5`
Run: `cargo test 2>&1 | tail -5`
Expected: Both pass — tracing is zero-cost when disabled

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml src/lib.rs src/retrieval/pipeline.rs src/lifecycle/ src/bin/alaya-mcp.rs
git commit -m "feat: add optional tracing instrumentation"
```

### Task 17: DRY cleanup — Debug/Display derives and remaining patterns

**Files:**
- Modify: `src/types.rs` — ensure all types derive Debug
- Modify: Various modules — add Display impls for key types

- [ ] **Step 1: Audit all structs for missing Debug derives**

Run: Search for `pub struct` without `#[derive(Debug` in the same block.

- [ ] **Step 2: Add missing Debug derives**

For each struct missing `Debug`, add it. Exception: `AlayaStore` (contains `Connection`).

- [ ] **Step 3: Add Display impl for MemoryStatus**

```rust
impl std::fmt::Display for MemoryStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "episodes: {}, semantic_nodes: {}, preferences: {}, links: {}",
            self.episode_count, self.semantic_node_count,
            self.preference_count, self.link_count
        )
    }
}
```

- [ ] **Step 4: Add Display impls for lifecycle report types**

```rust
impl std::fmt::Display for ConsolidationReport { /* ... */ }
impl std::fmt::Display for TransformationReport { /* ... */ }
impl std::fmt::Display for ForgettingReport { /* ... */ }
```

- [ ] **Step 5: Run full test suite**

Run: `cargo test --features "mcp llm" 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/
git commit -m "refactor: add Debug/Display derives for debuggability"
```

---

## Phase 4: Fill New Coverage Gaps

### Task 18: Add tests for new decay.rs module

**Files:**
- Modify: `src/decay.rs`

- [ ] **Step 1: Run tarpaulin on decay.rs specifically**

Check for any uncovered branches in the new `Decay`/`SqlDecay` implementations.

- [ ] **Step 2: Add tests for any uncovered branches**

Additional tests to consider:
- `ExponentialDecay.apply_sql` with empty table
- `MultiplicativeDecay.apply_sql` with no matching rows (all values below 0.01)
- Property test: `MultiplicativeDecay.factor` always returns the stored factor
- Property test: `ExponentialDecay.factor` is monotonically decreasing with elapsed time

- [ ] **Step 3: Run tests and verify 100%**

Run: `cargo tarpaulin --config tarpaulin.toml --skip-clean -- decay 2>&1 | tail -10`
Expected: 100% coverage for `src/decay.rs`

- [ ] **Step 4: Commit**

```bash
git add src/decay.rs
git commit -m "test: 100% coverage for decay module"
```

### Task 19: Add tests for new MCP submodules

**Files:**
- Modify: `src/mcp/validation.rs`, `src/mcp/serialization.rs`
- Modify: `src/mcp/memory.rs`, `src/mcp/lifecycle.rs`, `src/mcp/preferences.rs`, `src/mcp/query.rs`, `src/mcp/import.rs`, `src/mcp/status.rs`

- [ ] **Step 1: Run tarpaulin on mcp/ directory**

Identify all uncovered lines/branches in the decomposed modules.

- [ ] **Step 2: Add unit tests for validation.rs**

Cover: every validation function, error paths, edge cases.

- [ ] **Step 3: Add unit tests for serialization.rs**

Cover: every format function with empty, single, and multi-item inputs.

- [ ] **Step 4: Add unit tests for each domain handler module**

Cover: success paths, error paths, edge cases per handler.

- [ ] **Step 5: Run tarpaulin and verify 100%**

Run: `cargo tarpaulin --config tarpaulin.toml --skip-clean 2>&1 | grep -E "coverage"`
Expected: 100% for all MCP submodules

- [ ] **Step 6: Commit**

```bash
git add src/mcp/
git commit -m "test: 100% coverage for MCP submodules"
```

### Task 20: Add doc-tests for public API

**Files:**
- Modify: `src/lib.rs` — add `/// # Examples` to all public methods missing them

- [ ] **Step 1: Identify public methods missing doc-tests**

Check each public method on `AlayaStore` for `/// # Examples` section.

- [ ] **Step 2: Add doc-tests**

For each public method, add a minimal working example:

```rust
/// Query memories using hybrid retrieval.
///
/// # Examples
///
/// ```
/// use alaya::{AlayaStore, Query};
/// let store = AlayaStore::open_in_memory().unwrap();
/// let results = store.query(&Query::simple("test")).unwrap();
/// assert!(results.is_empty()); // no episodes stored yet
/// ```
pub fn query(&self, query: &Query) -> Result<Vec<RetrievalResult>> {
```

- [ ] **Step 3: Run doc-tests**

Run: `cargo test --doc 2>&1 | tail -10`
Expected: All doc-tests pass

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs
git commit -m "docs: add doc-tests for all public API methods"
```

### Task 21: Final coverage verification and CI enforcement

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Run full tarpaulin coverage check**

Run: `cargo tarpaulin --config tarpaulin.toml --skip-clean 2>&1 | tail -20`
Expected: 100% line coverage, 100% branch coverage

- [ ] **Step 2: If any gaps remain, add exclusion comments or tests**

For genuinely unreachable code, add:
```rust
#[cfg(not(tarpaulin_include))]  // Unreachable: <justification>
```

For reachable uncovered code, write the missing test.

- [ ] **Step 3: Update CI to enforce 100% threshold**

Update the coverage job in `.github/workflows/ci.yml`:

```yaml
      - name: Check coverage threshold
        run: |
          COVERAGE=$(cargo tarpaulin --config tarpaulin.toml --skip-clean --out json 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('coverage',0):.2f}\")" 2>/dev/null || echo "0")
          echo "Coverage: ${COVERAGE}%"
          python3 -c "import sys; sys.exit(0 if float('${COVERAGE}') >= 100.0 else 1)"
```

- [ ] **Step 4: Run full test suite one final time**

Run: `cargo test --features "mcp llm" 2>&1 | tail -10`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: enforce 100% coverage threshold"
```

### Task 22: Final verification

- [ ] **Step 1: Run all test suites**

```bash
cargo test --verbose
cargo test --features mcp --verbose
cargo test --features llm --verbose
cargo test --features "mcp llm" --verbose
cargo test --doc
```

Expected: All pass across all feature combinations.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets --features "mcp llm" -- -D warnings
```

Expected: Zero warnings.

- [ ] **Step 3: Run formatting check**

```bash
cargo fmt -- --check
```

Expected: Already formatted.

- [ ] **Step 4: Run coverage one final time**

```bash
cargo tarpaulin --config tarpaulin.toml --skip-clean
```

Expected: 100.00% line, 100.00% branch.

- [ ] **Step 5: Final commit if any formatting fixes needed**

```bash
git add -A
git commit -m "chore: final formatting and cleanup"
```
