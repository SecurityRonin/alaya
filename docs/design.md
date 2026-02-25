# Alaya: A Neuroscience and Buddhist-Inspired Memory Engine

## Name

**Alaya** (Sanskrit: *alaya-vijnana*, "storehouse consciousness") — the eighth
consciousness in Yogacara Buddhism. The persistent substrate that stores seeds
(bija), accumulates impressions through perfuming (vasana), and lets seeds ripen
when conditions align. No ego, no identity, no interpretation. Just the store.

## What Alaya Is

A Rust library crate that provides a headless, LLM-agnostic memory engine for
conversational AI agents. It stores, links, consolidates, retrieves, and forgets.

## What Alaya Is Not

- Not an agent. No SOUL.md, no personality, no system prompts.
- Not an LLM wrapper. No Ollama/OpenAI calls. The agent provides embeddings and
  consolidation logic via traits.
- Not a context assembler. The agent calls `query()` + `preferences()` and
  builds its own prompt.

## Research Foundations

### Neuroscience

| Principle | Mechanism | Alaya Component |
|-----------|-----------|-----------------|
| Hebbian LTP/LTD | Synapses strengthen on co-activation, weaken on disuse | Weighted graph links |
| Complementary Learning Systems | Fast hippocampus + slow neocortex | Episodic + Semantic stores |
| Spreading Activation (Collins & Loftus) | Activation propagates through associative network | Graph-based retrieval |
| Encoding Specificity (Tulving) | Retrieval matches encoding context | Context-weighted reranking |
| Dual-Strength Forgetting (Bjork) | Storage strength vs retrieval strength | Adaptive forgetting |
| Retrieval-Induced Forgetting (Anderson) | Retrieving some memories suppresses competitors | Post-retrieval suppression |
| Working Memory Limits (Cowan) | 4 +/- 1 chunks | Max 3-5 retrieval results |
| Episodic-to-Semantic Consolidation | Repeated episodes become general knowledge | Consolidation pipeline |

### Yogacara Buddhism

| Concept | Meaning | Alaya Component |
|---------|---------|-----------------|
| Alaya-vijnana (8th consciousness) | storehouse — persistent, always-on | The crate itself |
| Bija (seeds) | Living potentials that ripen under conditions | All stored nodes |
| Vasana (perfuming) | Gradual impression accumulation | Implicit store + perfuming process |
| Asraya-paravrtti (transformation) | Periodic refinement toward clarity | Transformation lifecycle |
| Vijnaptimatrata (consciousness-only) | Memory is perspective-relative | No "objective facts" — traces with confidence |

### QMD (Tobi Lutke)

| Technique | Alaya Adaptation |
|-----------|-----------------|
| Query expansion | Future enhancement (requires LLM, agent-provided) |
| Reciprocal Rank Fusion | Core fusion algorithm for multi-signal retrieval |
| Position-aware blending | Rank-dependent weight mixing in reranking |

## Architecture

```mermaid
graph LR
    subgraph Alaya["ALAYA (memory crate)"]
        ES[Episodic Store]
        SS[Semantic Store]
        IS[Implicit Store<br/>vasana]
        GO[Graph Overlay<br/>Hebbian links]
        RE[Retrieval Engine]
        LP[Lifecycle Processes]

        ES --- GO
        SS --- GO
        IS --- GO
        GO --- RE
        RE --- LP
    end

    subgraph Agent["AGENT (ccchat, OpenClaw, etc.)"]
        SOUL[SOUL.md<br/>agent identity]
        USER[USER.md<br/>user identity]
        CTX[Context window mgmt]
        SYS[System prompt assembly]
        LLM[LLM calls]
        EMB[Embedding provider]
    end

    Agent <-->|query + preferences<br/>store + lifecycle| Alaya
```

### Boundary

The agent calls `query()` + `preferences()` and assembles its own prompts.
Alaya never calls an LLM. The agent provides embeddings on write and
consolidation logic via traits.

### Three Stores

**Episodic** (hippocampus analog): Raw conversation events with full context.
Fast writes, natural decay. Each episode records what was said, who said it,
when, and the surrounding context (topics, sentiment, entities).

**Semantic** (neocortex analog): Distilled knowledge extracted from repeated
episodes. Slow-written through consolidation. Each node represents a fact,
relationship, event, or concept with a confidence score and provenance chain.

**Implicit / Vasana** (alaya proper): Preferences and behavioral patterns that
emerge from accumulated observations. Never explicitly declared. Impressions
accumulate until they crystallize into preferences when a pattern crosses a
confidence threshold.

### Graph Overlay

```mermaid
graph TD
    subgraph Episodic
        E1[Episode: discussed Rust async]
        E2[Episode: asked about tokio]
        E3[Episode: mentioned dog]
    end

    subgraph Semantic
        S1[Fact: user is a Rust developer]
        S2[Fact: user has a dog named Max]
    end

    subgraph Implicit
        P1[Pref: concise answers]
        P2[Pref: prefers code examples]
    end

    E1 -->|temporal| E2
    E1 -->|topical 0.8| S1
    E2 -->|topical 0.9| S1
    E3 -->|entity 0.7| S2
    S1 -->|co-retrieval 0.6| P2
    E1 -->|co-retrieval 0.4| P1

    style E1 fill:#4a9eff,color:#fff
    style E2 fill:#4a9eff,color:#fff
    style E3 fill:#4a9eff,color:#fff
    style S1 fill:#22c55e,color:#fff
    style S2 fill:#22c55e,color:#fff
    style P1 fill:#f59e0b,color:#fff
    style P2 fill:#f59e0b,color:#fff
```

A weighted directed graph spans all three stores. Edges represent associations:
temporal (A before B), topical (shared topic), entity (shared mention), causal
(A led to B), co-retrieval (A and B retrieved together — Hebbian).

Links have asymmetric forward/backward weights. Co-retrieval strengthens weights
(LTP), disuse weakens them (LTD). The graph naturally develops small-world
topology: dense topic clusters with sparse cross-domain shortcuts.

### Retrieval Engine

```mermaid
flowchart LR
    Q[Query] --> BM25[BM25<br/>FTS5]
    Q --> VEC[Vector<br/>cosine sim]
    Q --> GR[Graph<br/>neighbors]

    BM25 --> RRF[Reciprocal<br/>Rank Fusion]
    VEC --> RRF
    GR --> RRF

    RRF --> RR[Context-Weighted<br/>Reranking]
    RR --> SA[Spreading<br/>Activation]
    SA --> RIF[Post-Retrieval<br/>RIF Update]
    RIF --> OUT[Top 3-5<br/>Results]
```

Four-stage pipeline:

1. **Parallel retrieval**: BM25 (FTS5) + vector (cosine) + graph neighbors
2. **Reciprocal Rank Fusion**: Merge the three result sets
3. **Context-weighted reranking**: Boost results matching current context
4. **Spreading activation**: Follow graph links to find associatively related memories

Post-retrieval: boost retrieved nodes, suppress activated-but-not-retrieved
(retrieval-induced forgetting).

### Lifecycle Processes

```mermaid
flowchart TD
    INT[Interaction] --> EP[Store Episode]
    EP --> PF[Perfuming]
    PF --> |impressions accumulate| CR[Crystallize Preferences]

    EP --> |periodic| CON[Consolidation]
    CON --> |episodes grouped| SEM[Create Semantic Nodes]

    SEM --> |periodic| TR[Transformation]
    CR --> |periodic| TR
    TR --> DD[Dedup]
    TR --> CT[Contradiction Resolution]
    TR --> PR[Pruning]

    subgraph Always Running
        FG[Forgetting]
        FG --> |decay| RS[Retrieval Strength]
        FG --> |archive| AR[Low-Strength Nodes]
    end
```

- **Consolidation**: Episodic -> semantic (CLS interleaved replay)
- **Perfuming**: Interactions -> impressions -> crystallized preferences
- **Transformation**: Dedup, contradiction resolution, pattern extraction, pruning
- **Forgetting**: Dual-strength decay (storage vs retrieval strength)

## Public API

```rust
pub trait Alaya {
    // Write
    fn store_episode(&self, episode: NewEpisode) -> Result<EpisodeId>;

    // Read
    fn query(&self, q: Query) -> Result<Vec<ScoredMemory>>;
    fn preferences(&self, domain: Option<&str>) -> Result<Vec<Preference>>;
    fn knowledge(&self, filter: Option<KnowledgeFilter>) -> Result<Vec<SemanticNode>>;
    fn neighbors(&self, node: NodeRef, depth: u32) -> Result<Vec<(NodeRef, f32)>>;

    // Lifecycle (triggered by agent or autonomous schedule)
    async fn consolidate(&self, provider: &dyn ConsolidationProvider) -> Result<Report>;
    async fn perfume(&self, interaction: &Interaction, provider: &dyn ConsolidationProvider) -> Result<Report>;
    async fn transform(&self) -> Result<Report>;
    async fn forget(&self) -> Result<Report>;

    // Admin
    fn status(&self) -> Result<MemoryStatus>;
    fn purge(&self, filter: PurgeFilter) -> Result<PurgeReport>;
    fn export(&self, format: ExportFormat) -> Result<ExportData>;
}
```

The agent provides embeddings via `Option<Vec<f32>>` on write, and LLM logic via
the `ConsolidationProvider` trait. Alaya never calls an LLM directly.

## Design Principles

1. **Memory is a process, not a database.** Every retrieval changes what is
   remembered (reconsolidation, RIF). The graph reshapes through use.

2. **Forgetting is a feature.** Strategic decay and suppression improve
   retrieval quality over time.

3. **Preferences emerge, they are not declared.** The vasana/perfuming model
   lets behavioral patterns crystallize from accumulated observations.

4. **The agent owns identity.** Alaya stores seeds. The agent decides which
   seeds matter and how to present them.

5. **Graceful degradation.** No embeddings? BM25-only retrieval. No LLM for
   consolidation? Episodes accumulate until one is available. Every feature
   is independently useful.
