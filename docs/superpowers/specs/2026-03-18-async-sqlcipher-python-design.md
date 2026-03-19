# Async API, SQLCipher, and Python Bindings Design

**Date:** 2026-03-18
**Status:** Draft

## Goals

1. Add an async API via an actor pattern for safe concurrent access
2. Add optional SQLCipher encryption via feature flag
3. Add Python bindings via PyO3/maturin exposing the full AlayaStore API
4. Transition to a Cargo workspace to support the Python crate

## Constraints

- Public API of `AlayaStore` remains unchanged (sync API is preserved)
- Core crate stays zero-network-calls (privacy by architecture)
- Feature flag ceiling: 6 maximum on core crate
- Python bindings sync-only (no pyo3-asyncio); consumers use `asyncio.to_thread()` if needed
- SQLCipher and bundled-sqlite are mutually exclusive at compile time
- Implementation order: async → SQLCipher → workspace migration → Python → CI

---

## Phase 1: Async API (Actor Pattern)

### Feature Flag

```toml
[features]
async = ["dep:tokio"]
```

Adds `tokio` as an optional dependency (only `sync` and `rt` features needed for mpsc/oneshot/spawn).

### Architecture

```
AsyncAlayaStore                          Actor Thread
┌─────────────┐    mpsc::Sender         ┌──────────────┐
│  async fn   │ ───── Request ────────> │  AlayaStore   │
│  query()    │                         │  (owns conn)  │
│  store()    │ <── oneshot::Receiver ── │  loop { }     │
│  dream()    │                         └──────────────┘
└─────────────┘
```

- `AsyncAlayaStore` holds a `tokio::sync::mpsc::Sender<Request>` and a `JoinHandle` for the actor thread
- A dedicated `std::thread::spawn` background thread owns the `AlayaStore` and `tokio::sync::mpsc::Receiver`
- Each public method sends a `Request` variant with a `tokio::sync::oneshot::Sender` for the reply
- The actor thread runs a `while let Some(req) = rx.recv() { ... }` loop, dispatching to `AlayaStore` methods
- `Drop` sends `Request::Shutdown` and joins the thread

### New Module

**`src/async_store.rs`** (behind `#[cfg(feature = "async")]`)

### Request Enum (internal)

```rust
enum Request {
    StoreEpisode { episode: NewEpisode, reply: oneshot::Sender<Result<EpisodeId>> },
    Query { query: Query, reply: oneshot::Sender<Result<Vec<ScoredMemory>>> },
    Dream { interaction: Option<Interaction>, reply: oneshot::Sender<Result<DreamReport>> },
    Consolidate { reply: oneshot::Sender<Result<ConsolidationReport>> },
    // ... one variant per public AlayaStore method
    SetConsolidationProvider { provider: Box<dyn ConsolidationProvider + Send> },
    SetEmbeddingProvider { provider: Box<dyn EmbeddingProvider + Send> },
    Shutdown,
}
```

### Provider Handling

`ConsolidationProvider` and `EmbeddingProvider` are stored inside the actor thread, set via dedicated messages. `dream()` and `consolidate()` on `AsyncAlayaStore` use the stored provider, not a per-call argument.

```rust
impl AsyncAlayaStore {
    pub async fn set_consolidation_provider(&self, provider: Box<dyn ConsolidationProvider + Send>) { ... }
    pub async fn dream(&self, interaction: Option<&Interaction>) -> Result<DreamReport> { ... }
}
```

### Public API

```rust
let store = AsyncAlayaStore::open("memory.db").await?;
store.set_consolidation_provider(Box::new(my_provider)).await;
store.store_episode(&episode).await?;
let results = store.query(&Query::simple("rust")).await?;
let report = store.dream(None).await?;
```

### Error Handling

If the actor thread panics or the channel is closed, methods return `AlayaError::ActorDead` (new error variant, `#[non_exhaustive]` already on `AlayaError`).

### Testing

- Basic open/store/query/close roundtrip
- Concurrent access from multiple tokio tasks (verify serialization)
- Actor shutdown on drop
- Provider set/use lifecycle
- Error propagation from actor thread

---

## Phase 2: SQLCipher Support

### Feature Flag Restructure

```toml
[features]
default = ["bundled-sqlite"]
bundled-sqlite = ["rusqlite/bundled"]
sqlcipher = ["rusqlite/bundled-sqlcipher-vendored"]
```

`bundled-sqlite` becomes the default. `sqlcipher` replaces it. The two are mutually exclusive — enabling both is a compile error (rusqlite enforces this).

### New API

```rust
#[cfg(feature = "sqlcipher")]
impl AlayaStore {
    /// Open an encrypted database. The key is passed to SQLCipher via `PRAGMA key`.
    pub fn open_encrypted(path: impl AsRef<Path>, key: &str) -> Result<Self> { ... }

    /// Re-encrypt the database with a new key via `PRAGMA rekey`.
    pub fn rekey(&self, new_key: &str) -> Result<()> { ... }
}
```

- `open_encrypted` opens the connection, executes `PRAGMA key = '{key}'`, then initializes the schema
- `rekey` executes `PRAGMA rekey = '{new_key}'`
- Key format: SQLCipher accepts raw strings, hex-encoded (`x'...'`), or PBKDF2-derived keys. We pass through as-is — key derivation is the consumer's responsibility
- The existing `open()` works for unencrypted databases even when SQLCipher is bundled

### Async Integration

When both `async` and `sqlcipher` features are enabled:

```rust
#[cfg(all(feature = "async", feature = "sqlcipher"))]
impl AsyncAlayaStore {
    pub async fn open_encrypted(path: impl AsRef<Path>, key: &str) -> Result<Self> { ... }
    pub async fn rekey(&self, new_key: &str) -> Result<()> { ... }
}
```

### Testing

- Open encrypted, write, close, reopen with correct key — data accessible
- Reopen with wrong key — returns error
- `rekey`, close, reopen with new key — data accessible
- `open()` (no key) still works for unencrypted databases
- Async variants of the above (when both features enabled)

---

## Phase 3: Workspace Migration

### Structure Change

```
alaya/                      (repo root)
  Cargo.toml                (workspace definition)
  alaya/                    (core library)
    Cargo.toml
    src/
    tests/
  alaya-py/                 (Python bindings)
    Cargo.toml
    pyproject.toml
    src/
    tests/
```

### Root Cargo.toml

```toml
[workspace]
members = ["alaya", "alaya-py"]
resolver = "2"
```

### Migration Steps

1. Create `alaya/` subdirectory
2. Move `src/`, `tests/`, `Cargo.toml`, `tarpaulin.toml` into it
3. Create root `Cargo.toml` with workspace definition
4. Update CI paths (cargo commands now target `--package alaya` or run from workspace root)
5. Update `.github/workflows/` to use workspace-aware commands
6. Verify all existing tests, clippy, fmt pass

### Path-Dependent Files

- `tarpaulin.toml` moves into `alaya/`
- `deny.toml` stays at workspace root
- `CLAUDE.md` stays at repo root
- npm/ and .github/ stay at repo root

---

## Phase 4: Python Bindings

### Crate: `alaya-py`

```toml
[package]
name = "alaya-py"
version = "0.3.0"

[lib]
name = "alaya"
crate-type = ["cdylib"]

[dependencies]
alaya = { path = "../alaya", features = ["sqlcipher"] }
pyo3 = { version = "0.23", features = ["abi3-py39", "extension-module"] }
```

Note: The `alaya` dependency features will be configurable. The default build includes `sqlcipher` so Python consumers get encryption support. If binary size is a concern, it can be feature-gated.

### Build System

**`pyproject.toml`:**
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "alaya"
requires-python = ">=3.9"

[tool.maturin]
features = ["pyo3/abi3-py39"]
```

### Python API

**`src/lib.rs`** — PyO3 module:

```python
from alaya import Alaya, Query, Role

# Constructor (unencrypted)
store = Alaya("memory.db")

# Constructor (encrypted)
store = Alaya.open_encrypted("memory.db", key="secret")

# In-memory
store = Alaya.in_memory()

# Context manager
with Alaya("memory.db") as store:
    store.store_episode(
        content="Rust has zero-cost abstractions",
        role="user",
        session_id="s1",
        timestamp=1700000000,
    )
    results = store.query("Rust")
    report = store.dream()
    status = store.status()
```

### File Structure

```
alaya-py/
  Cargo.toml
  pyproject.toml
  src/
    lib.rs          # PyO3 module, Alaya class, open/close/context-manager
    types.rs        # Python wrappers for Episode, Query, ScoredMemory, reports, etc.
    provider.rs     # PyConsolidationProvider: Python class → Rust trait bridge
  alaya.pyi         # Type stubs for IDE autocompletion
  tests/
    test_basic.py       # open, store, query, close
    test_lifecycle.py   # dream, consolidate, transform, forget
    test_provider.py    # Python-implemented provider
    test_encrypted.py   # SQLCipher via Python
```

### Type Mapping

| Rust | Python |
|------|--------|
| `String` | `str` |
| `i64` | `int` |
| `f64` / `f32` | `float` |
| `Vec<T>` | `list[T]` |
| `Option<T>` | `T | None` |
| `Result<T, AlayaError>` | returns `T`, raises `AlayaError` |
| `EpisodeId(i64)` | `int` |
| `NodeRef` | `tuple[str, int]` e.g. `("episode", 42)` |
| `Role` | `str` (`"user"`, `"assistant"`, `"system"`) |
| `Query` | `Query` class or plain `str` for simple queries |
| Report types | dataclass-like PyO3 classes with named fields |

### Provider Bridge

Python consumers can implement `ConsolidationProvider` in Python:

```python
class MyProvider(alaya.ConsolidationProvider):
    def extract_knowledge(self, episodes):
        return [...]

    def extract_impressions(self, interaction):
        return [...]

    def detect_contradiction(self, a, b):
        return False

store.set_consolidation_provider(MyProvider())
```

Internally, `PyConsolidationProvider` wraps the Python object and implements the Rust `ConsolidationProvider` trait, calling back into Python via `pyo3::Python::with_gil`.

### Error Handling

- Rust `AlayaError` maps to a Python `AlayaError` exception
- The exception message contains the Rust error's `Display` output
- Subclasses: `AlayaError.InvalidInput`, `AlayaError.Database`, `AlayaError.Provider`

---

## Phase 5: CI & Publishing

### New CI Jobs

1. **Async tests:** `cargo test --package alaya --features async`
2. **SQLCipher tests:** `cargo test --package alaya --features sqlcipher --no-default-features`
3. **Python wheel build:** `maturin-action` for linux-x64, darwin-arm64, darwin-x64, windows-x64
4. **Python tests:** `pytest alaya-py/tests/` against built wheel

### Release Workflow Extension

On tag `v*`:
1. Existing: crates.io publish, npm publish, MCP registry, GitHub release
2. New: `maturin build --release` per platform → PyPI publish (trusted publishing OIDC)

### Feature Combination Matrix

Tests run for these feature combinations:
- `alaya` (default — bundled-sqlite)
- `alaya --features mcp`
- `alaya --features llm`
- `alaya --features async`
- `alaya --features "mcp llm async"`
- `alaya --features sqlcipher --no-default-features`
- `alaya --features "sqlcipher async" --no-default-features`
- `alaya-py` (pytest)

---

## Deliverables

1. `AsyncAlayaStore` with actor pattern behind `feature = "async"`
2. `open_encrypted()` and `rekey()` behind `feature = "sqlcipher"`
3. Workspace migration (`alaya/` + `alaya-py/`)
4. Python `alaya` package on PyPI with full API mirror, type stubs, provider bridge
5. CI: async tests, SQLCipher tests, wheel builds, PyPI publishing
6. Updated release workflow for multi-artifact publishing
