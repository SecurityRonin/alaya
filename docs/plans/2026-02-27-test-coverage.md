# 100% Test Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve 100% function coverage with unit tests + E2E integration tests. Maximize maintainability and debuggability.

**Architecture:** Add tests to existing `#[cfg(test)]` modules in each source file. Create `tests/integration.rs` for E2E. Every public and internal function gets at least one happy-path test plus edge case tests for boundary conditions and error paths.

**Tech Stack:** Rust, `#[cfg(test)]`, tempfile (dev-dependency)

**Verification:** `cargo check --tests` (linker blocked by Xcode license; compilation check sufficient)

---

## Current Coverage: 52 tests, ~68% function coverage

## Target: ~110+ tests, 100% function coverage + E2E

---

### Task 1: types.rs + error.rs tests

**Files:**
- Modify: `src/types.rs` (add tests module)
- Modify: `src/error.rs` (add tests module)

Add tests for:
- `Role::as_str()` / `Role::from_str()` — all variants + invalid input
- `SemanticType::as_str()` / `SemanticType::from_str()` — all variants + invalid
- `LinkType::as_str()` / `LinkType::from_str()` — all variants + invalid
- `NodeRef::from_parts()` — all variants + invalid type string
- `NodeRef::type_str()` / `NodeRef::id()` — all variants
- `Query::simple()` — defaults
- `AlayaError` Display — all 5 variants

### Task 2: store/episodic.rs + store/semantic.rs tests

**Files:**
- Modify: `src/store/episodic.rs`
- Modify: `src/store/semantic.rs`

Add tests for:
- `get_recent_episodes()` — ordering, limit, empty table
- `get_unconsolidated_episodes()` — returns unlinked, excludes linked
- `get_episode()` — not found error
- `delete_episodes()` — empty slice returns 0
- `find_by_type()` — single type filter, ordering by confidence
- `delete_node()` — cascade cleanup (embeddings, links, strengths)
- `get_semantic_node()` — not found error
- `count_nodes()` — after store and delete

### Task 3: store/implicit.rs + store/embeddings.rs + store/strengths.rs tests

**Files:**
- Modify: `src/store/implicit.rs`
- Modify: `src/store/embeddings.rs`
- Modify: `src/store/strengths.rs`

Add tests for:
- `count_impressions_by_domain()` — counting, non-existent domain
- `decay_preferences()` — decay old, skip fresh
- `prune_weak_preferences()` — delete below threshold, keep above
- `prune_old_impressions()` — delete old, keep recent
- `get_preferences(None)` — all domains
- `get_embedding()` — found, not found
- `get_unembedded_episodes()` — mixed embedded/unembedded
- `cosine_similarity()` — different lengths, zero vector, negative
- `boost_retrieval()` — clamp at 1.0
- `find_archivable()` — below both thresholds, mixed
- `get_strength()` — default for untracked node

### Task 4: graph/links.rs + retrieval tests

**Files:**
- Modify: `src/graph/links.rs`
- Modify: `src/retrieval/rerank.rs`
- Modify: `src/retrieval/fusion.rs`

Add tests for:
- `decay_links()` — weight reduction, skip already-weak
- `on_co_retrieval()` — creates new link if none exists
- `count_links()` — after create and prune
- `context_similarity()` — full match, no match, partial match
- `rerank()` — empty candidates, ordering, max_results truncation
- `rrf_merge()` — empty sets

### Task 5: lib.rs API-level tests

**Files:**
- Modify: `src/lib.rs`

Add tests for:
- `preferences()` — with domain filter, without filter, after perfuming
- `knowledge()` — with type filter, min_confidence filter, limit
- `neighbors()` — with links, without links, depth=0
- `perfume()` — stores impressions, crystallizes at threshold
- `transform()` — prunes weak links
- `forget()` — decays retrieval strength, archives weak nodes
- `purge(Session)` — deletes only target session
- `purge(OlderThan)` — deletes old episodes

### Task 6: E2E integration tests

**Files:**
- Create: `tests/integration.rs`

Add tests for:
- Multi-session lifecycle (3 sessions, store → query → consolidate → perfume → transform → forget)
- Persistence across open/close cycles (tempfile)
- Full retrieval pipeline with co-retrieval Hebbian strengthening
- Preference crystallization end-to-end
- Memory decay and revival

---

## Summary

| Task | What | Approx New Tests |
|------|------|-----------------|
| 1 | types.rs + error.rs | ~15 |
| 2 | episodic + semantic store | ~10 |
| 3 | implicit + embeddings + strengths | ~14 |
| 4 | graph + retrieval | ~10 |
| 5 | lib.rs API tests | ~12 |
| 6 | E2E integration | ~5 |

**Total new tests:** ~66
**Total after:** ~118
