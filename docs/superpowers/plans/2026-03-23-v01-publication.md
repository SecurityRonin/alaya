# v0.1 Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close P0 hardening gaps and publish Alaya to crates.io with working README examples, key doctests, input validation, and semver safety.

**Architecture:** No structural changes. Add `#[non_exhaustive]` to 2 enums, add input validation guards at sub-manager boundaries using existing `AlayaError::InvalidInput`, update README code examples to current API, add doctests to key public methods, verify publish readiness.

**Tech Stack:** Rust, rusqlite, cargo publish

---

## Task 1: Add `#[non_exhaustive]` to Missing Enums

**Files:**
- Modify: `alaya/src/types.rs`

- [ ] **Step 1: Write tests verifying the enums exist and can be matched**

Add to the existing tests in `alaya/src/types.rs` (or at the bottom of the file in a test module):

```rust
#[test]
fn conflict_status_is_non_exhaustive() {
    // This test documents that ConflictStatus has #[non_exhaustive].
    // If the attribute is removed, this test still compiles but the
    // attribute's presence is verified by the compiler at use sites.
    let s = ConflictStatus::Detected;
    assert_eq!(s.as_str(), "detected");
}

#[test]
fn conflict_strategy_is_non_exhaustive() {
    let s = ConflictStrategy::default();
    assert!(matches!(s, ConflictStrategy::Recency));
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p alaya --features "mcp async" -- conflict_status_is_non_exhaustive conflict_strategy_is_non_exhaustive --test-threads=1`
Expected: PASS

- [ ] **Step 3: Add `#[non_exhaustive]` to both enums**

In `alaya/src/types.rs`, add `#[non_exhaustive]` before `ConflictStatus` (line ~179):

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConflictStatus {
```

And before `ConflictStrategy` (line ~208):

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictStrategy {
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add alaya/src/types.rs
git commit --no-gpg-sign -m "fix(types): add #[non_exhaustive] to ConflictStatus and ConflictStrategy"
```

---

## Task 2: Add Input Validation to Episodes

**Files:**
- Modify: `alaya/src/managers/episodes.rs`

- [ ] **Step 1: Write failing tests for content length and embedding validation**

Add to the test module in `alaya/src/managers/episodes.rs`:

```rust
#[test]
fn store_rejects_oversized_content() {
    let alaya = Alaya::open_in_memory().unwrap();
    let mut ep = episode("x");
    ep.content = "x".repeat(100 * 1024 + 1); // 100KB + 1 byte
    let result = alaya.episodes().store(&ep);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("100KB"), "error should mention limit: {err}");
}

#[test]
fn store_rejects_nan_embedding() {
    let alaya = Alaya::open_in_memory().unwrap();
    let mut ep = episode("valid content");
    ep.embedding = Some(vec![1.0, f32::NAN, 3.0]);
    let result = alaya.episodes().store(&ep);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("NaN") || err.contains("finite"), "error should mention NaN: {err}");
}

#[test]
fn store_rejects_infinity_embedding() {
    let alaya = Alaya::open_in_memory().unwrap();
    let mut ep = episode("valid content");
    ep.embedding = Some(vec![1.0, f32::INFINITY, 3.0]);
    let result = alaya.episodes().store(&ep);
    assert!(result.is_err());
}

#[test]
fn store_accepts_valid_embedding() {
    let alaya = Alaya::open_in_memory().unwrap();
    let mut ep = episode("valid content");
    ep.embedding = Some(vec![0.1, 0.2, 0.3]);
    let result = alaya.episodes().store(&ep);
    assert!(result.is_ok());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p alaya -- store_rejects_oversized store_rejects_nan store_rejects_infinity --test-threads=1`
Expected: FAIL (no validation yet)

- [ ] **Step 3: Add validation to `store()`**

In `alaya/src/managers/episodes.rs`, add these checks after the existing `session_id` validation (after line 19), before the `db::transact` call:

```rust
        const MAX_CONTENT_BYTES: usize = 100 * 1024; // 100KB
        if episode.content.len() > MAX_CONTENT_BYTES {
            return Err(AlayaError::InvalidInput(format!(
                "episode content exceeds 100KB limit ({} bytes)",
                episode.content.len()
            )));
        }
        if let Some(ref emb) = episode.embedding {
            if emb.iter().any(|v| !v.is_finite()) {
                return Err(AlayaError::InvalidInput(
                    "embedding contains NaN or infinity values".into(),
                ));
            }
        }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add alaya/src/managers/episodes.rs
git commit --no-gpg-sign -m "fix(episodes): add content length and embedding validation"
```

---

## Task 3: Add Input Validation to Knowledge

**Files:**
- Modify: `alaya/src/managers/knowledge.rs`

- [ ] **Step 1: Write failing tests for content length and query length**

Add to the test module in `alaya/src/managers/knowledge.rs`:

```rust
#[test]
fn learn_rejects_oversized_content() {
    let alaya = Alaya::open_in_memory().unwrap();
    let node = crate::NewSemanticNode {
        content: "x".repeat(100 * 1024 + 1),
        node_type: crate::SemanticType::Fact,
        confidence: 0.9,
        source_episodes: vec![],
    };
    let result = alaya.knowledge().learn(vec![node]);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("100KB"), "error should mention limit: {err}");
}

#[test]
fn query_rejects_oversized_text() {
    let alaya = Alaya::open_in_memory().unwrap();
    let q = crate::Query {
        text: "x".repeat(100 * 1024 + 1),
        ..crate::Query::simple("x")
    };
    let result = alaya.knowledge().query(&q);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("100KB"), "error should mention limit: {err}");
}

#[test]
fn learn_accepts_valid_content() {
    let alaya = Alaya::open_in_memory().unwrap();
    let node = crate::NewSemanticNode {
        content: "The sky is blue".into(),
        node_type: crate::SemanticType::Fact,
        confidence: 0.9,
        source_episodes: vec![],
    };
    let result = alaya.knowledge().learn(vec![node]);
    assert!(result.is_ok());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p alaya -- learn_rejects_oversized query_rejects_oversized --test-threads=1`
Expected: FAIL

- [ ] **Step 3: Add validation to `query()` and `learn()`**

In `alaya/src/managers/knowledge.rs`, add to `query()` after the existing `max_results` check (after line 22):

```rust
        const MAX_TEXT_BYTES: usize = 100 * 1024;
        if q.text.len() > MAX_TEXT_BYTES {
            return Err(AlayaError::InvalidInput(format!(
                "query text exceeds 100KB limit ({} bytes)",
                q.text.len()
            )));
        }
```

Add to `learn()` at the start of the method (after line 37, before `db::transact`):

```rust
        const MAX_CONTENT_BYTES: usize = 100 * 1024;
        for node in &nodes {
            if node.content.len() > MAX_CONTENT_BYTES {
                return Err(AlayaError::InvalidInput(format!(
                    "semantic node content exceeds 100KB limit ({} bytes)",
                    node.content.len()
                )));
            }
        }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add alaya/src/managers/knowledge.rs
git commit --no-gpg-sign -m "fix(knowledge): add content and query length validation"
```

---

## Task 4: Update README Code Examples

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update all `AlayaStore` references to `Alaya`**

Replace all 6 occurrences of `AlayaStore` with `Alaya` in `README.md`:

| Line | Old | New |
|------|-----|-----|
| 27 | `let store = AlayaStore::open("memory.db")?;` | `let alaya = Alaya::open("memory.db")?;` |
| 206 | `use alaya::{AlayaStore, NewEpisode, ...}` | `use alaya::{Alaya, NewEpisode, ...}` |
| 209 | `let store = AlayaStore::open("memory.db")?;` | `let mut alaya = Alaya::open("memory.db")?;` |
| 271 | `AlayaStore struct` | `Alaya struct` |
| 379 | `let mut store = AlayaStore::open("memory.db")?;` | `let mut alaya = Alaya::open("memory.db")?;` |
| 411 | `impl AlayaStore {` | `impl Alaya {` |

- [ ] **Step 2: Update flat method calls to sub-manager syntax**

Replace these patterns throughout README.md:

| Old | New |
|-----|-----|
| `store.store_episode(&episode)?;` | `alaya.episodes().store(&episode)?;` |
| `store.query(&query)?;` | `alaya.knowledge().query(&query)?;` |
| `store.consolidate(&provider)?;` | `alaya.lifecycle().consolidate(&provider)?;` |
| `store.transform()?;` | `alaya.lifecycle().transform()?;` |
| `store.forget()?;` | `alaya.lifecycle().forget()?;` |
| `store.categories(None)?;` | `alaya.admin().categories(None)?;` |
| `store.purge(PurgeFilter::Session("s1"))?;` | `alaya.admin().purge(PurgeFilter::Session("s1"))?;` |
| `store.preferences(Some("..."))?;` | `alaya.admin().preferences(Some("..."))?;` |
| `store.set_extraction_provider(...)` | `alaya.set_extraction_provider(...)` |
| variable name `store` | `alaya` |

Also update the import line (line ~206) to include `NoOpProvider` if consolidation example uses it.

- [ ] **Step 3: Verify no `AlayaStore` or `store.` flat method calls remain**

Search README.md for `AlayaStore` and `store.store_episode`, `store.query`, `store.consolidate`, etc. — none should remain.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit --no-gpg-sign -m "docs: update README code examples for Alaya sub-manager API"
```

---

## Task 5: Add Doctests to Key Public Methods

**Files:**
- Modify: `alaya/src/managers/episodes.rs`
- Modify: `alaya/src/managers/knowledge.rs`
- Modify: `alaya/src/managers/lifecycle.rs`
- Modify: `alaya/src/managers/admin.rs`

Doctests go on the sub-manager method doc comments. Each doctest must be self-contained (creates its own `Alaya::open_in_memory()`).

- [ ] **Step 1: Add doctest to `Episodes::store()`**

In `alaya/src/managers/episodes.rs`, replace the `store()` doc comment:

```rust
    /// Store a conversation episode with full context.
    ///
    /// ```
    /// use alaya::{Alaya, NewEpisode, Role, EpisodeContext};
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// let id = alaya.episodes().store(&NewEpisode {
    ///     content: "Rust has zero-cost abstractions.".to_string(),
    ///     role: Role::User,
    ///     session_id: "session-1".to_string(),
    ///     timestamp: 1700000000,
    ///     context: EpisodeContext::default(),
    ///     embedding: None,
    /// }).unwrap();
    /// assert!(id.0 > 0);
    /// ```
```

- [ ] **Step 2: Add doctest to `Knowledge::query()`**

In `alaya/src/managers/knowledge.rs`, replace the `query()` doc comment:

```rust
    /// Hybrid retrieval: BM25 + vector + graph activation -> RRF -> rerank.
    ///
    /// ```
    /// use alaya::{Alaya, NewEpisode, Role, EpisodeContext, Query};
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// alaya.episodes().store(&NewEpisode {
    ///     content: "Rust has zero-cost abstractions.".to_string(),
    ///     role: Role::User,
    ///     session_id: "s1".to_string(),
    ///     timestamp: 1700000000,
    ///     context: EpisodeContext::default(),
    ///     embedding: None,
    /// }).unwrap();
    ///
    /// let results = alaya.knowledge().query(&Query::simple("Rust")).unwrap();
    /// assert!(!results.is_empty());
    /// ```
```

- [ ] **Step 3: Add doctest to `Lifecycle::consolidate()`**

In `alaya/src/managers/lifecycle.rs`, replace the `consolidate()` doc comment:

```rust
    /// Run consolidation: episodic -> semantic (CLS replay).
    ///
    /// ```
    /// use alaya::{Alaya, NoOpProvider};
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// let report = alaya.lifecycle().consolidate(&NoOpProvider).unwrap();
    /// assert_eq!(report.nodes_created, 0); // no episodes to consolidate
    /// ```
```

- [ ] **Step 4: Add doctest to `Admin::status()`**

In `alaya/src/managers/admin.rs`, replace the `status()` doc comment:

```rust
    /// Get a summary of memory system state.
    ///
    /// ```
    /// use alaya::Alaya;
    ///
    /// let alaya = Alaya::open_in_memory().unwrap();
    /// let status = alaya.admin().status().unwrap();
    /// assert_eq!(status.episode_count, 0);
    /// ```
```

- [ ] **Step 5: Run doc tests**

Run: `cargo test -p alaya --doc -- --test-threads=1`
Expected: all doc tests PASS (including the existing lib.rs and Query::simple doctests)

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p alaya --features "mcp async" -- --test-threads=1`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add alaya/src/managers/
git commit --no-gpg-sign -m "docs: add doctests to Episodes::store, Knowledge::query, Lifecycle::consolidate, Admin::status"
```

---

## Task 6: Publish Verification

**Files:**
- Possibly modify: `alaya/Cargo.toml` (only if issues found)

- [ ] **Step 1: Run `cargo doc` and verify clean build**

Run: `cargo doc --no-deps -p alaya --features "mcp async" 2>&1`
Expected: no warnings or errors. Fix any broken doc links if found.

- [ ] **Step 2: Verify Cargo.toml metadata**

Verify these fields are present and correct in `alaya/Cargo.toml`:
- `name = "alaya"`
- `description` — present and meaningful
- `license = "MIT"`
- `repository` — valid GitHub URL
- `keywords` — present (max 5)
- `categories` — present
- `readme` — points to `../README.md`

All should already be correct (version is 0.2.6). No changes expected.

- [ ] **Step 3: Run `cargo publish --dry-run`**

Run: `cargo publish --dry-run -p alaya 2>&1`
Expected: PASS (packaging succeeds, no errors). If it warns about missing fields, fix them.

- [ ] **Step 4: Run full test suite with all feature combos**

Run these sequentially:
```bash
cargo test -p alaya -- --test-threads=1
cargo test -p alaya --features "mcp async" -- --test-threads=1
cargo test -p alaya --features "tracing" -- --test-threads=1
```
Expected: all PASS

- [ ] **Step 5: Run clippy**

Run: `cargo clippy -p alaya --features "mcp async" -- -D warnings`
Expected: zero warnings

- [ ] **Step 6: Commit any fixes**

If any issues were found and fixed:
```bash
git add -u
git commit --no-gpg-sign -m "chore: fix publish verification issues"
```

If no issues: no commit needed.
