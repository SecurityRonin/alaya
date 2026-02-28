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

## Comparison with Alternatives

<details>
<summary>Click to expand comparison table</summary>

Grouped by category. Within each category, sorted by feature richness
(graph, forgetting, preferences, hybrid retrieval, multi-store
architecture), then by adoption where richness is comparable. For an
interactive visualization, see [docs/memory-landscape.html](docs/memory-landscape.html).

#### Dedicated Memory Systems

Purpose-built memory engines you integrate into your agent.

| System | Lang | Storage | Infra | LLM | Memory Model | Graph | Retrieval | Forgetting | Preferences |
|--------|:----:|---------|:-----:|:---:|-------------|:-----:|-----------|:----------:|:-----------:|
| **Alaya** | Rust | SQLite single file | None | Optional (traits) | Three-store: episodic, semantic, implicit | Hebbian (reshapes through use) | BM25 + vector + graph + RRF | Bjork dual-strength + RIF | Vasana (emergent) |
| **Vestige** | Rust | SQLCipher SQLite | None | None (local Nomic + Qwen3) | Working + semantic + procedural | Spreading activation graph | 7-stage cognitive (HyDE + BM25 + semantic + rerank + temporal + competition + spreading) | FSRS-6 spaced repetition | Prediction error gating |
| **MemOS** | Python | Graph + vector (Postgres, Redis) | 0-2 | Optional (OpenAI, Ollama, local) | Multi-type: episodic, semantic, procedural, preference, skill | Yes (unified, inspectable) | BM25 + graph recall + mixture search | Version-controlled consolidation | Yes (preference memory type) |
| **EverMemOS** | Python | MemCells / MemScenes | 0-1 | Required | Engram lifecycle: traces → scenes → reconstruction | MemScene hierarchies | MemScene-guided agentic retrieval | Consolidation compression + foresight signals | User profile updating |
| **OpenMemory** | Python | SQLite / Postgres | None | Required | 5-sector: episodic, semantic, procedural, emotional, reflective | Temporal KG + waypoint graph | Hybrid semantic + temporal + sector-weighted | Sector-specific decay rates | Yes (reflective + emotional) |
| **mem0** | Python | Qdrant/Pinecone + Postgres + Neo4j | 2-3 | Required | Tiered + optional graph | Optional (Mem0g) | Vector + graph | Exponential decay | LLM-extracted |
| **Zep / Graphiti** | Python | Neo4j + Lucene | 1-2 | Required | Temporal knowledge graph | Static temporal KG | Cosine + BM25 + graph + RRF | Temporal invalidation | Indirect (graph) |
| **Supermemory** | TS | KG + vector + graph DB | 2-3 | Required | Graph + vector with decay | Yes | Hybrid vector + graph | Decay curves + expiry | LLM-extracted |
| **Neo4j Agent Memory** | Python | Neo4j | 1 | Required | 3-type: short-term, long-term, reasoning (POLE+O entities) | Yes (Neo4j core) | Multi-stage entity extraction + graph | No | Long-term facts/prefs |
| **Hindsight** | Python | Configurable | 1-2 | Required | Four networks: world, experience, opinion, observation | No | Temporal priming + adaptive reasoning | Belief confidence evolution | Opinion memory (novel) |
| **MemoryOS** | Python | Configurable | 0-1 | Required | Three-tier OS hierarchy | No | Hierarchical cross-tier | FIFO + paging | Yes |
| **Memary** | Python | Neo4j | 1 | Required | KG with entity tracking and depth ranking | Yes (Neo4j, multi-hop) | Recursive subgraph + multi-hop reasoning | Compression/summarization | Entity-based personalization |
| **OpenClaw Redis** | TS/Python | Redis Stack (JSON + Search + Vector) | 1-2 | Required | Two-tier: working (session) + long-term (episodic, semantic, message) | No | Vector cosine + recency boosting | Recency scoring (soft) + GDPR deletion | Yes (dedicated extraction strategy) |
| **Cognee** | Python | Neo4j + vectors | 1-2 | Required | Vector + graph knowledge engine | Yes | Hybrid vector + graph | Not documented | Via graph |
| **Redis Memory** | Python | Redis + backends | 1+ | Required | Topics + entities + HNSW | No | HNSW vector + topic filter | No | No |
| **Cortex-Mem** | Rust | Configurable | 0-1 | Required | Extracted facts with dedup | No | Vector similarity | No | No |
| **PageIndex** | Python | JSON tree index | 0 | Required (OpenAI) | Hierarchical ToC tree | Tree (DAG) | LLM reasoning over tree | No | No |
| **Memvid** | Rust | Single `.mv2` file | None | None (local ONNX) | Append-only Smart Frames | No | Tantivy FTS + HNSW | None (immutable) | No |
| **OpenViking** | Python | VikingDB | 1 | Required | Virtual filesystem: L0, L1, L2 | No | Directory + semantic search | Implicit (tiered) | No |

#### Framework Memory Modules

Memory features built into larger agent frameworks.

| System | Lang | Storage | Infra | LLM | Memory Model | Graph | Retrieval | Forgetting | Preferences |
|--------|:----:|---------|:-----:|:---:|-------------|:-----:|-----------|:----------:|:-----------:|
| **Letta (MemGPT)** | Python | Postgres + Chroma/Qdrant | 1-2 | Required (LLM = manager) | OS-inspired: core, recall, archival | No | Agent-driven tool calls | Eviction + summarization | Agent-edited blocks |
| **CrewAI** | Python | SQLite | 0 | Required | Unified: short-term + long-term + entity | No | Adaptive-depth recall with composite scoring | Not explicit | Importance scoring |
| **MS GraphRAG** | Python | KG (various backends) | 1-2 | Required | Hierarchical KG with Leiden community structures | Yes (core design) | Graph + community summaries | Incremental updates | No |
| **LangChain** | Python | In-memory / Redis | 0-1 | Optional | Buffer / summary / entity | No | Direct injection | Window / truncation | Minimal |
| **LlamaIndex** | Python | SQLite / Postgres | 0-1 | Optional | Composable blocks | No | Block-dependent | FIFO eviction | Basic (facts) |
| **LangMem SDK** | Python | LangGraph store | 0-1 | Required | Semantic + procedural (prompt updates) | No | Vector similarity | No | Via extracted facts |

#### Coding Agent Memory

Memory systems targeting IDE agents and coding assistants, typically
exposed via MCP.

| System | Lang | Storage | Infra | LLM | Memory Model | Graph | Retrieval | Forgetting | Preferences |
|--------|:----:|---------|:-----:|:---:|-------------|:-----:|-----------|:----------:|:-----------:|
| **Beads** | Go | Dolt (version-controlled SQL) | 0 | Agent-driven | Dependency-aware issue graph with threading | Yes (4 link types) | Graph traversal along dependency chains | Compaction of old tasks | No |
| **Engram** | Go | SQLite + FTS5 | None | Agent-driven | Agent-curated structured summaries | No | Full-text search (FTS5) | No | Decisions/patterns |
| **Memsearch** | Python | Milvus (vector) | 1 | Required (embeddings) | Markdown-first with vector index overlay | No | Cosine similarity + 3-layer progressive disclosure | Stale chunk auto-deletion | No |

#### File-Based Memory (MEMORY.md Pattern)

Agent reads and writes memory as markdown files. Simple to implement but
no automated lifecycle, retrieval ranking, or emergent structure.

| System | Lang | Storage | Infra | LLM | Memory Model | Graph | Retrieval | Forgetting | Preferences |
|--------|:----:|---------|:-----:|:---:|-------------|:-----:|-----------|:----------:|:-----------:|
| **OpenClaw** | Python | Markdown + SQLite FTS5 | None | Required (agent-authored) | Two-layer: MEMORY.md + daily logs | No | grep + FTS5 | No (append-only) | Agent-authored |
| **Claudesidian** | TS | Obsidian vault (markdown) | Obsidian | Required (Claude Code) | PARA folders + daily/weekly notes | Obsidian links (static) | Obsidian search + file scan | Manual summarization | No |

#### Research Architectures

Academic papers with reference implementations. Often influential
designs but not packaged as production libraries.

| System | Lang | Storage | Infra | LLM | Memory Model | Graph | Retrieval | Forgetting | Preferences |
|--------|:----:|---------|:-----:|:---:|-------------|:-----:|-----------|:----------:|:-----------:|
| **Generative Agents** | Python | In-memory | 0 | Required | Stream + reflections + plans | No | Recency x importance x relevance | Recency decay | Emergent |
| **SYNAPSE** | Python | In-memory | 0 | Required | Unified episodic-semantic graph | Spreading activation + lateral inhibition | Activation-based graph traversal | Temporal decay | No |
| **LightRAG** | Python | KG + vector (Neo4j, Milvus, etc.) | 0-2 | Required | Dual-level KG with incremental updates | Yes (core design) | Dual-level entity + community retrieval | KG regeneration on deletion | No |
| **A-MEM** | Python | Vector + note graph | 0 | Required | Zettelkasten-inspired linked notes | Implicit links | Embedding + note traversal | Evolution-based | No |
| **LightMem** | Python | Configurable | 0-1 | Required | Atkinson-Shiffrin: sensory, STM, LTM | No | Topic-grouped | Sleep-time consolidation | No |
| **Mem-alpha** | Python | Configurable | 0-1 | Required | Core + episodic + semantic (RL-managed) | No | RL-learned | RL-learned | No |
| **HippoRAG** | Python | In-memory KG | 0 | Required | Hippocampal indexing + open KG | Personalized PageRank | PPR on knowledge graph | No | No |

#### Vector Databases (Infrastructure Layer)

These provide storage and retrieval but not memory semantics (lifecycle,
forgetting, preference learning, graph dynamics).

| System | Language | Hybrid Search | Managed Cloud | Open Source |
|--------|----------|:------------:|:-------------:|:----------:|
| **Pinecone** | Cloud-native | No native BM25 | Yes (only option) | No |
| **ChromaDB** | Python | FTS + vector | Chroma Cloud | Yes |
| **Weaviate** | Go | BM25 + vector | Weaviate Cloud | Yes |
| **Milvus** | Go/C++ | Dense + sparse | Zilliz Cloud | Yes |
| **Cloudflare Vectorize** | Cloud-native | Via Workers AI | Yes (only option) | No |

For a comprehensive analysis grounded in the CoALA taxonomy (Sumers et al.,
2024) and RAG survey literature (Gao et al., 2023; Zhang et al., 2024), see
[docs/related-work.md](docs/related-work.md). For the neuroscience and Buddhist
psychology behind Alaya, see
[docs/theoretical-foundations.md](docs/theoretical-foundations.md).

</details>

## License

MIT
