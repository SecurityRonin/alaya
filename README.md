# Alaya

A neuroscience and Buddhist psychology-inspired memory engine for conversational AI agents.

**Alaya** (Sanskrit: *alaya-vijnana* · आलयविज्ञान · Chinese: 阿賴耶識, "storehouse consciousness") is a Rust
library that provides three-tier memory, a Hebbian graph overlay, hybrid
retrieval with spreading activation, and adaptive lifecycle processes. It is
headless and LLM-agnostic — the consuming agent owns identity, embeddings,
and prompt assembly.

## Why Alaya?

Most AI memory systems treat memory as a retrieval problem — store vectors,
fetch the nearest ones. Alaya treats memory as a *process*: memories strengthen
through co-retrieval, weaken through disuse, consolidate from episodes into
knowledge, and crystallize into preferences. The graph reshapes itself through
use, like a biological memory system.

**Key differentiators:**

- **Memory as process** — Hebbian graph reshaping, adaptive forgetting, and preference crystallization make memory a living system, not a static store
- **Principled foundations** — architecture grounded in CLS theory, Bjork forgetting, spreading activation, and Yogacara psychology, not ad-hoc heuristics
- **LLM-agnostic** — no hardcoded provider; the agent supplies embeddings and consolidation logic via traits
- **Graceful degradation** — no embeddings? BM25-only retrieval. No LLM? Episodes accumulate. Every feature works independently
- **Zero infrastructure** — one SQLite file, no external services, no network calls
- **Embeddable** — Rust with C FFI; runs anywhere with no runtime overhead

## Getting Started

### Installation

Add Alaya to your `Cargo.toml`:

```toml
[dependencies]
alaya = { git = "https://github.com/h4x0r/alaya" }
```

### Run the Demo

The included demo walks through all six core capabilities with annotated output
and no external dependencies (uses a rule-based provider instead of an LLM):

```bash
git clone https://github.com/h4x0r/alaya.git
cd alaya
cargo run --example demo
```

The demo covers:

1. **Episodic Memory** — storing and querying conversation episodes
2. **Hebbian Graph** — temporal links, co-retrieval strengthening, spreading activation
3. **Consolidation** — extracting semantic knowledge from episodes (CLS replay)
4. **Perfuming** — accumulating impressions, crystallizing preferences (vasana)
5. **Transformation** — deduplication, pruning, decay
6. **Forgetting** — Bjork dual-strength decay, memory revival

### Quick Start

```rust
use alaya::{AlayaStore, NewEpisode, Role, EpisodeContext, Query, NoOpProvider};

// Open a persistent database (or use open_in_memory() for tests)
let store = AlayaStore::open("memory.db")?;

// Store a conversation episode
store.store_episode(&NewEpisode {
    content: "I've been learning Rust for about six months now".into(),
    role: Role::User,
    session_id: "session-1".into(),
    timestamp: 1740000000,
    context: EpisodeContext::default(),
    embedding: None, // pass Some(vec![...]) if you have embeddings
})?;

// Query with hybrid retrieval (BM25 + vector + graph + RRF)
let results = store.query(&Query::simple("Rust experience"))?;
for mem in &results {
    println!("[{:.2}] {}", mem.score, mem.content);
}

// Get crystallized preferences
let prefs = store.preferences(Some("communication_style"))?;

// Run lifecycle (use NoOpProvider, or implement ConsolidationProvider for LLM-backed extraction)
store.consolidate(&NoOpProvider)?;
store.transform()?;
store.forget()?;
```

## Integration Guide

### The Integration Pattern

Alaya is a library, not a framework. Your agent owns the conversation loop,
the LLM connection, and the embedding model. Alaya owns memory storage,
retrieval, and lifecycle.

```
Your Agent                          Alaya
─────────                           ─────
receive message
  ├── store_episode()           ──▶ episodic store + graph links
  ├── query()                   ──▶ BM25 + vector + graph → RRF → rerank
  ├── preferences()             ──▶ crystallized behavioral patterns
  ├── knowledge()               ──▶ consolidated semantic nodes
  ├── assemble context + prompt
  ├── call LLM
  └── respond

periodic background tasks:
  ├── consolidate(provider)     ──▶ episodes → semantic knowledge
  ├── perfume(interaction, provider) ──▶ impressions → preferences
  ├── transform()               ──▶ dedup, prune, decay
  └── forget()                  ──▶ Bjork strength decay + archival
```

### Implementing ConsolidationProvider

The `ConsolidationProvider` trait is how your agent teaches Alaya to extract
knowledge. You implement three methods backed by your LLM of choice:

```rust
use alaya::*;

struct MyProvider { /* your LLM client */ }

impl ConsolidationProvider for MyProvider {
    fn extract_knowledge(&self, episodes: &[Episode]) -> Result<Vec<NewSemanticNode>> {
        // Ask your LLM: "What facts/relationships can you extract from these episodes?"
        // Return structured NewSemanticNode values
        todo!()
    }

    fn extract_impressions(&self, interaction: &Interaction) -> Result<Vec<NewImpression>> {
        // Ask your LLM: "What behavioral signals does this interaction contain?"
        // Return domain + observation + valence
        todo!()
    }

    fn detect_contradiction(&self, a: &SemanticNode, b: &SemanticNode) -> Result<bool> {
        // Ask your LLM: "Do these two facts contradict each other?"
        todo!()
    }
}
```

Use `NoOpProvider` if you don't have an LLM available — episodes accumulate
and BM25 retrieval works without consolidation.

### Lifecycle Scheduling

Call lifecycle methods on a schedule that suits your application:

| Method | When to call | What it does |
|--------|-------------|--------------|
| `consolidate()` | After accumulating 10+ episodes | Extracts semantic knowledge from episodes |
| `perfume()` | On every user interaction | Extracts behavioral impressions, crystallizes preferences |
| `transform()` | Daily or weekly | Deduplicates, prunes weak links, decays stale preferences |
| `forget()` | Daily or weekly | Decays retrieval strength, archives truly forgotten nodes |

## API Reference

```rust
impl AlayaStore {
    // Open / create
    pub fn open(path: impl AsRef<Path>) -> Result<Self>;
    pub fn open_in_memory() -> Result<Self>;

    // Write
    pub fn store_episode(&self, episode: &NewEpisode) -> Result<EpisodeId>;

    // Read
    pub fn query(&self, q: &Query) -> Result<Vec<ScoredMemory>>;
    pub fn preferences(&self, domain: Option<&str>) -> Result<Vec<Preference>>;
    pub fn knowledge(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>>;
    pub fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>>;

    // Lifecycle
    pub fn consolidate(&self, provider: &dyn ConsolidationProvider) -> Result<ConsolidationReport>;
    pub fn perfume(&self, interaction: &Interaction, provider: &dyn ConsolidationProvider) -> Result<PerfumingReport>;
    pub fn transform(&self) -> Result<TransformationReport>;
    pub fn forget(&self) -> Result<ForgettingReport>;

    // Admin
    pub fn status(&self) -> Result<MemoryStatus>;
    pub fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport>;
}
```

## Architecture

```mermaid
graph LR
    subgraph Alaya["ALAYA (memory crate)"]
        ES[Episodic Store]
        SS[Semantic Store]
        IS[Implicit Store]
        GO[Graph Overlay]
        RE[Retrieval Engine]
        LP[Lifecycle Processes]

        ES --- GO
        SS --- GO
        IS --- GO
        GO --- RE
        RE --- LP
    end

    subgraph Agent["AGENT"]
        SOUL[Identity / SOUL.md]
        CTX[Context Assembly]
        LLM[LLM Provider]
        EMB[Embedding Provider]
    end

    Agent <-->|query · store · lifecycle| Alaya
```

### Three Stores

| Store | Analog | Purpose |
|-------|--------|---------|
| **Episodic** | Hippocampus | Raw conversation events with full context |
| **Semantic** | Neocortex | Distilled knowledge extracted through consolidation |
| **Implicit** | Alaya-vijnana | Preferences and habits that emerge through perfuming |

### Graph Overlay

A Hebbian weighted directed graph spans all three stores. Links strengthen on
co-retrieval (LTP) and weaken through disuse (LTD), naturally developing
small-world topology.

### Retrieval Pipeline

```mermaid
flowchart LR
    Q[Query] --> BM25[BM25 / FTS5]
    Q --> VEC[Vector / Cosine]
    Q --> GR[Graph Neighbors]

    BM25 --> RRF[Reciprocal Rank Fusion]
    VEC --> RRF
    GR --> RRF

    RRF --> RR[Context-Weighted Reranking]
    RR --> SA[Spreading Activation]
    SA --> RIF[Retrieval-Induced Forgetting]
    RIF --> OUT[Top 3-5 Results]
```

### Lifecycle Processes

| Process | Inspiration | What it does |
|---------|-------------|--------------|
| **Consolidation** | CLS theory (McClelland et al.) | Distills episodes into semantic knowledge |
| **Perfuming** | Vasana (Yogacara Buddhist psychology) | Accumulates impressions, crystallizes preferences |
| **Transformation** | Asraya-paravrtti | Deduplicates, resolves contradictions, prunes |
| **Forgetting** | Bjork & Bjork (1992) | Decays retrieval strength, archives weak nodes |

## Design Principles

1. **Memory is a process, not a database.** Every retrieval changes what is
   remembered. The graph reshapes through use.

2. **Forgetting is a feature.** Strategic decay and suppression improve
   retrieval quality over time.

3. **Preferences emerge, they are not declared.** The vasana/perfuming model
   lets behavioral patterns crystallize from accumulated observations.

4. **The agent owns identity.** Alaya stores seeds. The agent decides which
   seeds matter and how to present them.

5. **Graceful degradation.** No embeddings? BM25-only. No LLM for
   consolidation? Episodes accumulate. Every feature works independently.

## Research Foundations

For detailed explanations of how each theory maps to Alaya's implementation,
see [docs/theoretical-foundations.md](docs/theoretical-foundations.md).

### Neuroscience

- **Hebbian LTP/LTD** — synapses strengthen on co-activation (Hebb 1949, Bliss & Lomo 1973)
- **Complementary Learning Systems** — fast hippocampus + slow neocortex (McClelland et al. 1995)
- **Spreading Activation** — associative retrieval beyond embedding similarity (Collins & Loftus 1975)
- **Encoding Specificity** — context-dependent retrieval (Tulving & Thomson 1973)
- **Dual-Strength Forgetting** — storage vs retrieval strength (Bjork & Bjork 1992)
- **Retrieval-Induced Forgetting** — retrieving some memories suppresses competitors (Anderson et al. 1994)
- **Working Memory Limits** — 4 +/- 1 chunks (Cowan 2001)

### Yogacara Buddhist Psychology

- **Alaya-vijnana** — the storehouse consciousness, persistent substrate for all seeds
- **Bija (seeds)** — living potentials that ripen when conditions align
- **Vasana (perfuming)** — gradual accumulation of impressions that shape behavior
- **Asraya-paravrtti** — periodic transformation toward clarity
- **Vijnaptimatrata** — memory is perspective-relative, not objective

### Information Retrieval

- **Reciprocal Rank Fusion** — merging multiple ranked result sets (Cormack et al. 2009)
- **BM25 via FTS5** — keyword matching with relevance scoring
- **Cosine Similarity** — semantic vector search

## Coming from MEMORY.md?

If you're using file-based memory (OpenClaw, Claudesidian, or a
hand-rolled `MEMORY.md`), you already have long-term memory across
sessions. The problem is what happens next: the files keep growing,
context windows fill up, and retrieval degrades to "dump everything and
hope the LLM finds what matters." Alaya solves the scaling problem with
structure underneath.

| What changes | MEMORY.md pattern | Alaya |
|---|---|---|
| **Storage** | Markdown files the agent reads/writes | SQLite with typed stores (episodes, knowledge, preferences) |
| **Retrieval** | `grep`, file scan, or dump everything into context | Ranked hybrid search: BM25 + vector + graph traversal + RRF fusion |
| **What gets remembered** | Whatever the agent decides to write down | Everything is stored; retrieval quality determines what surfaces |
| **Forgetting** | Manual cleanup or unbounded growth | Automatic: weak memories decay, strong ones persist (Bjork model) |
| **Associations** | None — flat files | Hebbian graph links memories that are retrieved together |
| **Preferences** | Agent-authored summary, easily drifts | Emerge from accumulated impressions (vasana), crystallize at threshold |
| **Context window cost** | Grows linearly — eventually you hit the limit | Ranked retrieval returns only the most relevant memories |
| **LLM dependency** | Required for writing and organizing | Optional — works without an LLM, gets better with one |

The tradeoff: MEMORY.md is zero-setup and human-readable. Alaya
requires `cargo add alaya` and a few trait implementations. In return
you get retrieval that improves with use, memories that self-organize,
and a context window that stays clean.

## Comparison with Alternatives

Alaya is compared against 40+ memory systems across six categories
(dedicated engines, framework modules, coding agent memory, file-based,
research architectures, and vector databases). The closest architectural
peers are **Vestige** (Rust, FSRS-6 spaced repetition, spreading
activation) and **SYNAPSE** (unified episodic-semantic graph, lateral
inhibition). Alaya is the only system combining CLS-inspired
consolidation, Bjork dual-strength forgetting, Hebbian graph reshaping,
and emergent preference crystallization.

- [Full comparison tables and system-by-system analysis](docs/related-work.md) — grounded in the CoALA taxonomy (Sumers et al., 2024) and RAG survey literature
- [Interactive landscape visualization](https://h4x0r.github.io/alaya/docs/memory-landscape.html) — D3.js force-directed graph of the memory system ecosystem
- [Theoretical foundations](docs/theoretical-foundations.md) — neuroscience and Buddhist psychology behind Alaya's architecture

## License

MIT
