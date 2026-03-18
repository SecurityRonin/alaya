# 100% Test Coverage & Radical Refactoring Design

**Date:** 2026-03-18
**Status:** Draft
**Approach:** Tests First, Then Refactor (Approach A)

## Goals

1. Achieve 100% line and branch test coverage
2. Radically refactor for maximum maintainability and debuggability
3. Strictly observe the DRY principle
4. Preserve the public API exactly (no breaking changes)

## Current State

- **13,162 LOC** across 29 source files
- **285 unit tests** + 2 integration test files (1,687 LOC)
- **No coverage reporting** (no tarpaulin, no codecov)
- **No logging framework** (only `eprintln!` in MCP binary)
- **Monolithic `mcp.rs`**: 3,415 LOC mixing protocol, validation, handlers, serialization
- **6 scattered decay implementations** across graph, retrieval, store, and lifecycle modules
- **Under-tested modules**: `mcp.rs` (integration-only), `retrieval/pipeline.rs` (3 tests), `retrieval/vector.rs` (1 test), `store/embeddings.rs` (minimal)

## Constraints

- Public API surface of `AlayaStore` and MCP tool names must remain identical
- Refactoring is internal only — no semver-breaking changes
- Library must remain zero-cost when `tracing` subscriber is not attached
- Coverage CI job runs on `ubuntu-latest` only (tarpaulin is Linux-only)
- Coverage measured with `--features "mcp llm"` (all features enabled)
- Genuinely unreachable code (defensive `unreachable!()` arms) may use `#[cfg(not(tarpaulin_include))]` exclusions — each exclusion must have a justifying comment

---

## Phase 1: Coverage Infrastructure & Baseline

### Coverage Tooling

Add `cargo-tarpaulin` with configuration for both line and branch coverage.

**`tarpaulin.toml` at project root:**
- Target: 100% line and branch coverage
- Branch coverage enabled (`--branch` flag)
- Output formats: HTML (local dev) + JSON (CI parsing)
- Exclude test code and build scripts from coverage measurement

**CI Workflow:**
- Add a coverage job to the existing GitHub Actions workflow
- Initially: report-only (no failure threshold)
- After Phase 2: enforce 100% threshold (fail on regression)

### Baseline Measurement

- Run tarpaulin against current 285 tests
- Generate HTML report identifying every uncovered line and branch
- This report drives Phase 2 priorities

### Test Organization

- **Unit tests**: `#[cfg(test)] mod tests` within each source file (Rust convention)
- **Integration/E2E tests**: `tests/` directory, organized by domain:
  - `tests/memory.rs` — episode storage, recall, semantic nodes
  - `tests/lifecycle.rs` — consolidation, perfuming, transformation, forgetting
  - `tests/mcp.rs` — MCP tool handler end-to-end tests
  - `tests/retrieval.rs` — pipeline, fusion, reranking
- **Shared fixtures**: `tests/common/mod.rs` for test helpers and builders

**Migration from existing test files:**
- Split `tests/integration.rs` into `tests/memory.rs`, `tests/lifecycle.rs`, `tests/retrieval.rs` by domain
- Rename `tests/mcp_tools.rs` to `tests/mcp.rs`
- Delete old files after migration is verified

---

## Phase 2: Test Writing (Against Current Code)

### Priority Order (Risk x Coverage Gap)

1. **`mcp.rs` (3,415 LOC, integration-only)**
   - Test each tool handler in isolation by mocking `AlayaStore`
   - Cover all param validation paths, error paths, edge cases
   - Empty inputs, malformed JSON, missing required fields

2. **`retrieval/pipeline.rs` (418 LOC, 3 tests)**
   - Test each pipeline stage independently
   - End-to-end pipeline tests: empty store, single entry, many entries, no embeddings

3. **`store/embeddings.rs` (347 LOC, minimal tests)**
   - Graceful degradation path (no embedding provider)
   - Storage/retrieval of vectors
   - Dimension mismatch handling

4. **`retrieval/vector.rs` (30 LOC, 1 test)**
   - Cosine similarity edge cases: zero vectors, identical, orthogonal, single-dimension

5. **All remaining modules**
   - Systematic pass targeting uncovered branches from tarpaulin report

### Branch Coverage Strategy

- Every `match` arm gets a test
- Every `if/else` path gets a test
- Every `Result` gets both `Ok` and `Err` tested
- Every `Option` gets both `Some` and `None` tested
- Error propagation boundaries: verify errors from inner modules surface correctly at outer boundaries

### Test Helpers (DRY in Tests)

- `TestFixtures` struct with factory methods: `empty_store()`, `populated_store()`, `store_with_episodes(n)`, `store_with_embeddings()`
- Builder pattern for constructing test episodes, semantic nodes, preferences
- Assertion helpers: `assert_recall_contains`, `assert_episode_stored`
- `#[cfg(test)] mod test_helpers` in `lib.rs` available to all unit test modules

### Coverage Verification

- Run tarpaulin with `--branch` flag
- Verify 100% line coverage
- Verify 100% branch coverage
- All 285 existing tests + new tests pass

---

## Phase 3: Refactoring

### 3a: Decay Trait Consolidation

**New module:** `src/decay.rs`

The current decay functions operate at two different levels:
1. **SQL-level decay** — `decay_links(conn, factor: f32)`, `decay_all_retrieval(conn, factor: f32)`, `decay_preferences(conn, now: i64, half_life: i64)` apply decay directly in SQL UPDATE statements
2. **Rust-level decay** — `recency_decay(timestamp: i64, now: i64) -> f64` computes a score in Rust

**Important constraint:** SQLite lacks `exp()`, so `decay_preferences` uses a linear approximation of exponential decay. This must be preserved.

The trait therefore has two facets:

```rust
/// Decay strategy — computes the factor to apply.
pub trait Decay {
    /// Compute a multiplicative decay factor (0.0..=1.0) for the given elapsed time.
    fn factor(&self, elapsed_secs: i64) -> f64;
}

/// Apply decay in a SQL UPDATE statement using a computed factor.
pub trait SqlDecay: Decay {
    /// Generate the SQL SET clause fragment and bind the factor.
    fn apply_sql(&self, conn: &Connection, table: &str, column: &str, elapsed_secs: i64) -> Result<u64>;
}

pub struct ExponentialDecay { pub half_life_secs: i64 }
pub struct MultiplicativeDecay { pub factor: f64 }
```

**Migration map:**

| Current Function | Signature | Becomes |
|---|---|---|
| `decay_links(conn, factor: f32)` | `graph/links.rs:93` | `MultiplicativeDecay.apply_sql()` |
| `decay_preferences(conn, now: i64, half_life: i64)` | `store/implicit.rs:135` | `ExponentialDecay.apply_sql()` (preserves linear approx) |
| `decay_all_retrieval(conn, factor: f32)` | `store/strengths.rs:88` | `MultiplicativeDecay.apply_sql()` |
| `recency_decay(timestamp: i64, now: i64) -> f64` | `retrieval/rerank.rs:38` | `ExponentialDecay.factor()` (Rust-level, no SQL) |
| forgetting decay | `lifecycle/forgetting.rs` | Uses configured `Decay` impl via existing call sites |
| transformation decay | `lifecycle/transformation.rs` | Uses configured `Decay` impl via existing call sites |

**Note:** `LinearDecay` is not included — no current code uses linear decay. The `decay_preferences` approximation is an implementation detail of `ExponentialDecay.apply_sql()` for SQLite, not a separate strategy. Adding `LinearDecay` later is trivial if needed (YAGNI).

**Exclusion:** `spread_activation`'s `decay_per_hop: f32` in `graph/activation.rs` is graph signal attenuation (spatial), not temporal decay. Intentionally excluded from this consolidation.

**Constraint:** Public API unchanged. The `Decay`/`SqlDecay` traits are internal. Call sites like `store.transform()` and `store.forget()` keep the same signatures.

**Testability gain:** One thorough test suite for each `Decay` impl (edge cases: zero elapsed, very large elapsed, negative values, overflow, the SQLite linear approximation accuracy). Consumer modules test correct delegation, not math.

### 3b: MCP Module Decomposition

**From:** `src/mcp.rs` (3,415 LOC monolith)
**To:** `src/mcp/` directory with 9 files

**Complete tool handler inventory (13 tools):**
`remember`, `recall`, `status`, `preferences`, `knowledge`, `maintain`, `categories`, `neighbors`, `node_category`, `learn`, `purge`, `import_claude_mem`, `import_claude_code`

**Domain modules (handlers):**
```
src/mcp/
  mod.rs              — Re-exports, MCP server struct, thin dispatcher
  memory.rs           — remember, recall (core memory operations)
  lifecycle.rs        — maintain, purge (lifecycle management)
  preferences.rs      — learn, preferences (preference learning)
  query.rs            — knowledge, categories, neighbors, node_category (read-only queries)
  import.rs           — import_claude_mem, import_claude_code (data import)
  status.rs           — status (system status)
```

**Cross-cutting layer modules:**
```
src/mcp/
  validation.rs       — Shared param extraction & validation logic
  serialization.rs    — Response formatting, error-to-JSON conversion
```

**Note on `protocol.rs`:** The `rmcp` crate already handles JSON-RPC framing and deserialization. A separate `protocol.rs` is unnecessary — `rmcp` owns that layer. The `mod.rs` dispatcher delegates directly to domain handlers after `rmcp` dispatches.

**Request flow:**
1. `rmcp` receives JSON-RPC request, deserializes, dispatches to `#[tool]` method
2. Domain handler calls `validation.rs` for param extraction/validation
3. Domain handler calls `AlayaStore` methods (business logic)
4. Domain handler calls `serialization.rs` for response formatting

**DRY gain:** Repeated param validation across 13 tool handlers extracted into `validation.rs` as reusable extractors (`extract_required_string(params, "key")`, `extract_optional_i64(params, "key", default)`). Repeated error-to-JSON conversion becomes a single function in `serialization.rs`.

**Public API preserved:** The `#[cfg(feature = "mcp")]` boundary and MCP server external interface remain identical.

### 3c: Tracing Instrumentation

**Dependency:** Add `tracing` as an optional feature (`tracing = ["dep:tracing"]`). Use `#[cfg_attr(feature = "tracing", tracing::instrument)]` macros so the library compiles without the tracing dependency when consumers don't want it. The `alaya-mcp` binary enables the `tracing` feature by default.

**Instrumentation levels:**
- `#[instrument]` on all public `AlayaStore` methods
- `#[instrument(skip(self))]` where `self` isn't Debug-printable
- `trace!` for internal pipeline steps (retrieval stages, decay applications, fusion scoring)
- `debug!` for state transitions (episode stored, node consolidated, preference learned)
- `warn!` for graceful degradation (no embedding provider, BM25-only fallback)
- `error!` for actual failures (database errors, corrupt data)

**Span hierarchy:**
- MCP handlers: `#[instrument(name = "mcp::remember", skip(self))]`
- Lifecycle operations: `#[instrument(name = "lifecycle::consolidate")]`
- Retrieval pipeline: nested spans `pipeline -> bm25 -> vector -> fusion -> rerank`

**Library vs binary boundary:** Library emits tracing events only. The `alaya-mcp` binary sets up the subscriber (`tracing-subscriber` with `fmt` layer to stderr). Library consumers choose their own subscriber.

### 3d: Additional DRY & Debuggability

1. **SQL query patterns** — If repeated query fragments exist across store modules, extract into constants or builder methods on a shared query module.

2. **Error context enrichment** — `#[instrument]` context ensures errors carry span information. No new error variants needed.

3. **`Debug` derives everywhere** — All internal structs derive `Debug` for `#[instrument]` capture. Use `#[instrument(skip(field))]` for large/sensitive fields. **Exception:** `AlayaStore` cannot derive `Debug` because `rusqlite::Connection` does not implement `Debug` — use `#[instrument(skip(self))]` on all `AlayaStore` methods.

4. **`Display` on key types** — `AlayaStatus`, retrieval results, lifecycle outcomes get meaningful `Display` impls for human-readable log output.

5. **Pipeline introspection** — Retrieval pipeline exposes which stages contributed to a result via tracing spans (BM25 score, vector score, fusion rank).

### Refactoring Verification

After each sub-step (3a, 3b, 3c, 3d):
- Run full test suite (all 285+ existing tests plus new Phase 2 tests)
- Verify zero regressions
- Verify public API unchanged

---

## Phase 4: Fill New Coverage Gaps

Refactored code creates new module boundaries and branches that need testing.

### New Test Targets

- **`src/decay.rs`** — Unit tests for `ExponentialDecay`, `MultiplicativeDecay` trait impls. Edge cases: zero elapsed, max elapsed, overflow, negative input, SQLite linear approximation accuracy.
- **`src/mcp/validation.rs`** — Unit tests for each extractor function. Missing params, wrong types, empty strings, boundary values.
- **`src/mcp/serialization.rs`** — Unit tests for response formatting. Error serialization, success formatting, edge case payloads.
- **Each domain handler module** (`memory.rs`, `lifecycle.rs`, `preferences.rs`, `query.rs`, `import.rs`, `status.rs`) — Unit tests that the handler correctly delegates to validation, store, and serialization.
### Property-Based Tests

Expand existing `proptest` usage (e.g., `prop_decay_links_weight_bounded` in `graph/links.rs`) to new decay trait impls and retrieval pipeline. Property tests verify invariants (decay always reduces, scores bounded 0..1) across random inputs.

### Doc-Tests

Add doc-tests to all public API items with `# Examples` sections. These count toward coverage and serve as living documentation.

### Coverage Enforcement

- Re-run tarpaulin with `--branch`
- Verify 100% line + branch coverage on refactored code
- Update CI threshold to 100% (hard gate — builds fail on regression)

---

## Deliverables

1. 100% line and branch coverage enforced in CI (Linux-only, all features enabled)
2. `mcp.rs` monolith decomposed into 9 focused modules (6 domain + 2 cross-cutting + mod.rs)
3. 4 decay functions consolidated into `Decay`/`SqlDecay` traits with 2 strategy impls; 2 lifecycle call sites updated
4. Optional `tracing` feature with `#[instrument]` span hierarchy
5. Shared test infrastructure (no duplicated fixtures)
6. All `Debug`/`Display` derives for debuggability (`AlayaStore` excluded, uses `skip(self)`)
7. CI coverage gate preventing regressions
8. Existing test files (`tests/integration.rs`, `tests/mcp_tools.rs`) migrated to domain-organized files
9. Expanded property-based tests and doc-tests on public API
