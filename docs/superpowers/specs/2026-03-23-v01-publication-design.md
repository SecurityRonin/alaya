# Minimum Viable v0.1 Publication

## Goal

Close remaining P0 hardening gaps and publish Alaya v0.1 to crates.io with a credible first impression — working README examples, key doctests, input validation, and semver safety.

## Principles

- **Ship fast, iterate later** — v0.1 is a credible baseline, not a perfect product
- **No behavior changes** — this is hardening and documentation, not new features
- **Executable documentation** — doctests double as tests; README examples must compile

## Scope

### 1. `#[non_exhaustive]` Fix

Add `#[non_exhaustive]` to two public enums in `types.rs` that are missing it:

- `ConflictStatus`
- `ConflictStrategy`

All other public enums (`NodeRef`, `Role`, `SemanticType`, `LinkType`, `PurgeFilter`) already have it. This ensures adding variants later is not a semver-breaking change.

### 2. Input Validation

Add validation at sub-manager boundaries where user data enters the system. Introduce a new `AlayaError::Validation(String)` variant.

**Episodes (`managers/episodes.rs` `store()`):**
- Reject content over 100KB
- If `episode.embedding` is `Some(vec)`, reject NaN/infinity values
- If an embedding provider is configured, validate embedding dimension matches provider's dimension

**Knowledge (`managers/knowledge.rs` `learn()`):**
- Reject `NewSemanticNode.content` over 100KB

**Query (`managers/knowledge.rs` `query()`):**
- Reject query text over 100KB

The 100KB cap is a sensible default — no legitimate conversation message or query approaches this size. It prevents accidental memory bombs from malformed input.

### 3. README Update

Update all code examples in `README.md` to use the current API:

- `AlayaStore` → `Alaya`
- Flat method calls → sub-manager syntax (`alaya.episodes().store()`, `alaya.knowledge().query()`, etc.)
- Verify all code blocks are copy-pasteable and would compile

No structural rewrite — positioning text, architecture description, and feature table stay as-is.

### 4. Doctests

Add compilable doc examples to 5-6 key public entry points:

- `Alaya::open_in_memory()` — constructor
- `Episodes::store()` — storing an episode
- `Knowledge::query()` — querying memories
- `Lifecycle::consolidate()` — running consolidation
- `Admin::status()` — checking system status
- `Query::simple()` — verify existing doctest works

Each doctest creates an in-memory `Alaya`, performs the operation, and asserts something meaningful. They are self-contained and run as part of `cargo test`.

### 5. Publish Verification

Final gate:

- `cargo publish --dry-run -p alaya` passes
- `cargo doc --no-deps -p alaya` builds clean
- Cargo.toml metadata complete: description, license (MIT), repository, keywords, categories
- Full test suite passes with all feature combinations (`mcp async`, `tracing`, `sqlcipher`)

## What Does NOT Change

- No new features
- No retrieval pipeline changes (LTD, RIF are P1)
- No new example programs (existing `demo.rs` is sufficient)
- No CHANGELOG generation (git history suffices for v0.1)
- No CI pipeline changes (already configured)
- Algorithm logic is untouched

## Breaking Changes

- New `AlayaError::Validation(String)` variant — downstream `match` arms on `AlayaError` will need a wildcard. This is safe because `AlayaError` already has `#[non_exhaustive]`.
- `#[non_exhaustive]` on `ConflictStatus` and `ConflictStrategy` — same pattern, forces wildcard arms.

## Estimated Scope

~10 files touched, ~200 lines added. Mostly documentation and validation guards.
