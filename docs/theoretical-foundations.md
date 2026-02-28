# Theoretical Foundations

Alaya's architecture draws from three domains: cognitive neuroscience,
Yogacara Buddhist psychology, and information retrieval theory. This
document explains each concept and maps it to specific implementation
details.

---

## Cognitive Neuroscience

### Complementary Learning Systems (CLS)

**Source:** McClelland, McNaughton & O'Reilly (1995). "Why there are
complementary learning systems in the hippocampus and neocortex."

**The problem:** Neural networks suffer from *catastrophic forgetting* —
learning new patterns overwrites old ones. How does the brain retain
decades of knowledge while absorbing new experiences daily?

**The theory:** The brain uses two complementary systems:

| System | Brain Region | Learning Speed | Purpose |
|--------|-------------|----------------|---------|
| Fast learner | Hippocampus | One-shot | Records raw experiences (episodes) |
| Slow learner | Neocortex | Gradual | Extracts stable patterns (knowledge) |

The key mechanism is **interleaved replay**: during sleep and rest, the
hippocampus replays recent episodes to the neocortex, which gradually
integrates them without overwriting existing knowledge. The replay
interleaves old and new memories to prevent interference.

**In Alaya:**

| CLS Component | Alaya Component | Implementation |
|---------------|-----------------|----------------|
| Hippocampus | Episodic Store | `store/episodic.rs` — raw conversation events |
| Neocortex | Semantic Store | `store/semantic.rs` — distilled knowledge nodes |
| Replay | `consolidate()` | `lifecycle/consolidation.rs` — the `ConsolidationProvider` examines batches of episodes and extracts semantic nodes |

Consolidation requires 3+ unconsolidated episodes (to ensure
corroboration) and processes them in batches of 10. The original episodes
are preserved — just as the hippocampus retains episodes alongside
neocortical knowledge. The two stores serve different retrieval needs:
episodes for "what exactly happened" and semantic nodes for "what do I
know."

### Hebbian Learning (LTP / LTD)

**Source:** Hebb (1949). "The Organization of Behavior." Bliss & Lomo
(1973), experimental confirmation in hippocampal slices.

**The principle:** "Neurons that fire together wire together." When two
neurons are repeatedly co-activated, the synapse between them
strengthens (Long-Term Potentiation, LTP). Synapses that are not
co-activated weaken over time (Long-Term Depression, LTD).

**In Alaya:**

The graph overlay (`graph/links.rs`) implements Hebbian dynamics:

- **LTP (strengthening):** When two memories are retrieved together in
  the same query, `on_co_retrieval()` strengthens the link between them.
  The weight update uses an asymptotic learning rule:
  `w = w + 0.1 * (1.0 - w)` — rapid early strengthening that slows as
  the link approaches 1.0.

- **LTD (weakening):** `decay_links()` multiplies all link weights by a
  decay factor (default 0.9). Links that are not co-activated gradually
  fade. When links drop below the pruning threshold (0.02),
  `prune_weak_links()` removes them entirely during transformation.

- **Link creation:** If two nodes are co-retrieved but no link exists, a
  new `CoRetrieval` link is created with initial weight 0.3.

Over time, this produces emergent **small-world topology**: frequently
co-retrieved memories form tightly-linked clusters, with weaker
cross-cluster bridges.

### Spreading Activation

**Source:** Collins & Loftus (1975). "A spreading-activation theory of
semantic processing."

**The theory:** When a concept is activated (e.g., by thinking about
"doctor"), activation spreads to associated concepts ("nurse,"
"hospital," "stethoscope") through weighted links in a semantic network.
More strongly associated concepts receive more activation. Activation
decays with graph distance.

**In Alaya:**

The retrieval pipeline (`retrieval/pipeline.rs`) uses spreading
activation as a third retrieval signal alongside BM25 and vector search:

1. Top results from BM25 and vector search become **seed nodes**
2. `spread_activation()` (`graph/activation.rs`) propagates activation
   from seeds through the Hebbian graph
3. Activation decays by `decay_per_hop` (default 0.6) at each hop
4. Signal splits proportionally across outgoing edges using absolute
   weights (matching neuroscience: synaptic strength is absolute, not
   relative)
5. Activation is capped at 2.0 to prevent runaway cascading
6. Nodes below the threshold (default 0.1) are filtered out

This retrieves memories that are *associatively related* but might not
share keywords or embedding similarity — the same way thinking about
"rain" might activate "umbrella" even though the two words are not
semantically similar in embedding space.

### Bjork Dual-Strength Forgetting

**Source:** Bjork & Bjork (1992). "A new theory of disuse and an old
theory of stimulus fluctuation."

**The theory:** Each memory has two independent strengths:

| Strength | Behavior | Analogy |
|----------|----------|---------|
| **Storage strength** | How well-learned; only increases with access | How deeply engraved |
| **Retrieval strength** | How accessible right now; decays over time | How easy to find |

A memory with high storage but low retrieval is *latent* — it exists but
is hard to find without a strong cue. When re-accessed, its retrieval
strength resets immediately. This explains the "tip of the tongue"
phenomenon and why relearning is faster than initial learning.

**In Alaya:**

The `store/strengths.rs` module tracks both strengths for every node:

- **On access** (`on_access()`): storage strength increases
  asymptotically (`s = s + 0.05 * (1.0 - s)`), retrieval strength
  resets to 1.0
- **On forget sweep** (`forget()` in `lifecycle/forgetting.rs`):
  retrieval strength decays by 0.95 per sweep. Storage strength is
  never decreased.
- **Archival:** Nodes with both storage < 0.1 AND retrieval < 0.05
  are archived (deleted). Since storage only increases, this naturally
  targets only memories that were barely accessed in their lifetime.
- **Latent memories:** A memory accessed many times (high storage)
  but not recently (low retrieval) still exists in the store — it just
  ranks lower in retrieval. A query that matches it will revive its
  retrieval strength to 1.0.

### Retrieval-Induced Forgetting (RIF)

**Source:** Anderson, Bjork & Bjork (1994). "Remembering can cause
forgetting."

**The theory:** Retrieving a memory suppresses competing memories that
share the same retrieval cues. If you practice remembering "apple" from
the category "fruits," your ability to recall "orange" from "fruits"
temporarily decreases. This is not decay — it is active suppression
driven by retrieval itself.

**In Alaya:**

After retrieval, the pipeline tracks which nodes were accessed
(`strengths::on_access()`). Nodes that were *not* retrieved but share
graph links with retrieved nodes experience implicit suppression through
the dual-strength model: their retrieval strength continues to decay
while the retrieved nodes get a reset to 1.0. Over many retrieval
cycles, this creates a natural winner-take-all dynamic among competing
memories.

### Encoding Specificity

**Source:** Tulving & Thomson (1973). "Encoding specificity and retrieval
processes in episodic memory."

**The theory:** A memory is most accessible when the retrieval context
matches the encoding context. If you learned something while listening
to music, hearing that same music helps you recall it.

**In Alaya:**

Each episode stores an `EpisodeContext` with topics, sentiment,
mentioned entities, and conversation turn. The reranker
(`retrieval/rerank.rs`) computes context similarity between the query
context and each candidate's encoding context using:

- **Topic overlap** (Jaccard similarity, weight 0.5)
- **Entity overlap** (Jaccard similarity, weight 0.25)
- **Sentiment proximity** (absolute difference, weight 0.25)

The context score modulates the base retrieval score:
`final = base * (1 + 0.3 * context_sim) * (1 + 0.2 * recency)`

### Working Memory Limits

**Source:** Cowan (2001). "The magical number 4 in short-term memory."

**The principle:** Human working memory can hold roughly 4 ± 1 chunks at
a time. Overloading it degrades decision quality.

**In Alaya:**

`Query::simple()` defaults to `max_results: 5`, and the retrieval
pipeline enforces this limit. The system returns the *best* 3-5
memories, not all matching memories. This forces the retrieval pipeline
to be highly selective, which in turn drives the multi-stage
architecture (BM25 → vector → graph → RRF → rerank → truncate).

---

## Information Retrieval

### Reciprocal Rank Fusion (RRF)

**Source:** Cormack, Clarke & Buettcher (2009). "Reciprocal rank fusion
outperforms condorcet and individual rank learning methods."

**The problem:** How do you merge results from multiple rankers (BM25,
vector, graph) with incomparable score scales?

**The solution:** For each document *d*:

```
score(d) = Σ 1 / (k + rank_i + 1)
```

where *rank_i* is the 0-based rank of *d* in result set *i*, and *k* is
a constant (Alaya uses k=60). Documents that appear in multiple result
sets accumulate score from each. The formula is rank-based, so it does
not require score calibration between retrieval methods.

**In Alaya:**

`retrieval/fusion.rs` implements RRF. The pipeline feeds it up to three
ranked lists:

1. **BM25** (`retrieval/bm25.rs`) — keyword matching via SQLite FTS5
2. **Vector** (`retrieval/vector.rs`) — cosine similarity on embeddings
   (only when embeddings are provided)
3. **Graph** — nodes discovered via spreading activation that were not
   already in the BM25/vector results

The fused ranking is then reranked by context similarity and recency.

### BM25 via FTS5

SQLite's FTS5 extension provides BM25-ranked full-text search with no
external dependencies. Alaya stores episode content in an FTS5 virtual
table and queries it with standard FTS5 syntax. This gives keyword
retrieval that works without embeddings, enabling Alaya's graceful
degradation: no embedding model? BM25-only retrieval still works.

### Cosine Similarity

When the agent provides embeddings (via `NewEpisode.embedding` or
`Query.embedding`), Alaya stores them as binary blobs and computes
cosine similarity in Rust. This captures *semantic* similarity that
keyword matching misses ("programming in Rust" and "writing Rust code"
have low keyword overlap but high semantic similarity).

---

## Yogacara Buddhist Psychology

Alaya's architecture is named after and structurally informed by the
Yogacara (瑜伽行派) school of Buddhist philosophy, primarily as
systematized in the *Yogacarabhumi-sastra* (瑜伽師地論) attributed to
Maitreya/Asanga and translated into Chinese by Xuanzang (646-648 CE),
and the *Cheng Weishi Lun* (成唯識論, Xuanzang, 659 CE).

### Alaya-vijnana (阿賴耶識) — Storehouse Consciousness

**Concept:** The eighth consciousness, a persistent substrate that holds
all *seeds* (bija). It is neither conscious nor unconscious — it is a
continuous stream that underlies all mental activity, persisting through
sleep, meditation, and between lifetimes. It receives impressions from
every experience and stores them as potentials.

**In Alaya:**

The `AlayaStore` itself is the alaya-vijnana: a persistent SQLite
database that holds all memory seeds (episodes, semantic nodes,
preferences, graph links, embeddings, strengths). It has no agency of
its own — it stores potentials that the agent activates through queries
and lifecycle operations.

The key verse from the *Jiexhenmi Jing* (解深密經):

> 阿陀那識甚深細，一切種子如瀑流，我於凡愚不開演，恐彼分別執為我
>
> *The adana consciousness is exceedingly deep and subtle; all its seeds
> are like a torrential flood. I do not reveal it to the foolish, for
> fear they would grasp it as a self.*

### Bija (種子) — Seeds

**Concept:** Seeds are living potentials stored in the alaya-vijnana.
They are not static data — they have conditions for ripening, they
influence each other, and they can be strengthened or weakened. A seed
ripens into manifest experience when the right conditions converge.

**In Alaya:**

Every node in Alaya's three stores is a seed:

| Seed Type | Store | Ripening Condition |
|-----------|-------|--------------------|
| Episode | Episodic | Query matches content or context |
| Semantic Node | Semantic | Query activates via BM25, vector, or graph |
| Preference | Implicit | Agent calls `preferences()` for a domain |
| Graph Link | Graph Overlay | Connected node is activated during retrieval |

Seeds have *strength* (the `NodeStrength` dual model) that determines
how easily they ripen. Strong seeds are easily activated; weak seeds
require stronger cues.

### Vasana (薰習) — Perfuming

**Concept:** Each experience leaves a subtle trace (*vasana*) on the
alaya-vijnana, like incense gradually permeating cloth. No single
experience is transformative, but the accumulation of many similar
traces shapes behavior. The cloth does not choose to absorb the scent —
it is a natural process.

**In Alaya:**

The perfuming cycle (`lifecycle/perfuming.rs`) implements this directly:

1. **Impression extraction:** The `ConsolidationProvider` examines each
   interaction and extracts impressions — observations about the user's
   behavior, organized by domain (e.g., "communication,"
   "technical_preferences"), with a valence (+/- signal).

2. **Accumulation:** Impressions accumulate in the Implicit Store
   (`store/implicit.rs`). Each impression is a single vasana trace.

3. **Crystallization:** When a domain accumulates 5+ impressions
   (the `CRYSTALLIZATION_THRESHOLD`), the system crystallizes a
   *preference* — a stable behavioral pattern extracted from the
   accumulated traces. This is like the moment when the cloth has
   absorbed enough incense to carry the scent independently.

4. **Reinforcement:** Once a preference exists, further impressions in
   the same domain reinforce it (increasing `evidence_count` and
   updating `last_reinforced`). Preferences are *emergent*, not
   declared — they arise from accumulated observation, not from the
   user explicitly stating them.

### Asraya-paravrtti (轉依) — Transformation of the Basis

**Concept:** A periodic turning or transformation of the alaya-vijnana
itself. Not merely adding new content, but restructuring what exists —
resolving contradictions, purifying distortions, and moving the
storehouse toward greater clarity.

In Yogacara, the ultimate asraya-paravrtti is the transformation of the
eight consciousnesses into the four wisdoms (轉識成智):

| Consciousness | Transforms Into |
|---------------|----------------|
| Alaya-vijnana (8th) | 大圓鏡智 — Great Mirror Wisdom (reflecting reality without distortion) |
| Manas (7th) | 平等性智 — Equality Wisdom (seeing without self-other bias) |
| Mano-vijnana (6th) | 妙觀察智 — Subtle Observation Wisdom (discerning clearly) |
| Five sense consciousnesses (1st-5th) | 成所作智 — Accomplishing Wisdom (skillful action) |

**In Alaya:**

The `transform()` function (`lifecycle/transformation.rs`) implements a
practical version of asraya-paravrtti:

1. **Deduplication** — merging semantic nodes with near-identical
   embeddings (cosine similarity ≥ 0.95). The kept node's corroboration
   count increases, and the duplicate's links are transferred. This
   moves toward the "Great Mirror" — reflecting the user accurately
   without redundant distortion.

2. **Link pruning** — removing graph links below 0.02 weight.
   Associations that were never reinforced are cleared.

3. **Preference decay** — un-reinforced preferences lose confidence
   over time (half-life: 30 days). Preferences that drop below 0.05
   confidence are pruned entirely. This prevents stale behavioral
   patterns from persisting.

4. **Impression pruning** — impressions older than 90 days are removed.
   The traces have either crystallized into preferences or faded.

### Vipaka (異熟) — Maturation

**Concept:** Seeds ripen (mature) into manifest experience when
conditions align. The ripening is *heterogeneous* (異熟 literally means
"differently ripened") — the result differs in nature from the cause.
Moral seeds ripen into pleasure or pain, not into more morality.

**In Alaya:**

Consolidation is a maturation process: episodic seeds (raw
conversations) ripen into semantic nodes (structured knowledge) that
differ in nature from the original episodes. The semantic node
"User has been learning Rust for six months" is a *different kind of
thing* from the conversation that produced it — it is decontextualized,
structured, and reusable. This heterogeneous ripening is why Alaya
maintains separate stores rather than a single memory pool.

### Vijnaptimatrata (唯識) — Consciousness-Only

**Concept:** Memory is perspective-relative, not objective. What is
stored is not reality itself but a representation shaped by the context
and conditions of its formation. There is no "view from nowhere."

**In Alaya:**

Every episode stores the *perspective* of its formation: who said it
(`Role`), when (`timestamp`), in what session (`session_id`), about what
topics, with what sentiment, and following which preceding episode. The
same words spoken by the user vs. the assistant are stored as different
memories with different contexts. The retrieval system uses this context
to match memories to the current situation (encoding specificity).

---

## How Everything Fits Together

The retrieval pipeline integrates all of these theories in a single
query path:

```
Query arrives
│
├─ BM25 (keyword matching)              ← Information Retrieval
├─ Vector (semantic similarity)          ← Information Retrieval
├─ Graph (spreading activation)          ← Collins & Loftus
│   └─ through Hebbian-weighted edges    ← Hebb / LTP / LTD
│
├─ RRF Fusion (merge ranked lists)       ← Cormack et al.
│
├─ Rerank (context + recency)            ← Encoding Specificity + Recency
│   ├─ topic/entity/sentiment match      ← Tulving & Thomson
│   └─ exponential recency decay
│
├─ Truncate to 3-5 results              ← Working Memory Limits (Cowan)
│
└─ Post-retrieval updates
    ├─ on_access() → reset retrieval     ← Bjork dual-strength
    │   strength, boost storage
    └─ on_co_retrieval() → strengthen    ← Hebbian LTP
        links between retrieved nodes
```

The lifecycle processes run in the background:

```
consolidate()  → CLS replay (hippocampus → neocortex)     ← McClelland et al.
perfume()      → vasana accumulation → crystallization     ← Yogacara
transform()    → asraya-paravrtti (dedup, prune, decay)    ← Yogacara
forget()       → Bjork dual-strength decay + archival      ← Bjork & Bjork
```

Each retrieval changes the graph (Hebbian strengthening), each lifecycle
sweep changes the store (consolidation, transformation, forgetting).
Memory is not a static database — it is a living system that reshapes
itself through use.

---

## References

### Neuroscience

- Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can
  cause forgetting: Retrieval dynamics in long-term memory. *Journal of
  Experimental Psychology: Learning, Memory, and Cognition*, 20(5), 1063-1087.

- Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old
  theory of stimulus fluctuation. In A. Healy et al. (Eds.), *From learning
  processes to cognitive processes: Essays in honor of William K. Estes*
  (Vol. 2, pp. 35-67). Erlbaum.

- Bliss, T. V. P., & Lomo, T. (1973). Long-lasting potentiation of
  synaptic transmission in the dentate area of the anaesthetized rabbit
  following stimulation of the perforant path. *Journal of Physiology*,
  232(2), 331-356.

- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory
  of semantic processing. *Psychological Review*, 82(6), 407-428.

- Cowan, N. (2001). The magical number 4 in short-term memory: A
  reconsideration of mental storage capacity. *Behavioral and Brain
  Sciences*, 24(1), 87-114.

- Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.

- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why
  there are complementary learning systems in the hippocampus and
  neocortex: Insights from the successes and failures of connectionist
  models of learning and memory. *Psychological Review*, 102(3), 419-457.

- Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval
  processes in episodic memory. *Psychological Review*, 80(5), 352-373.

### Information Retrieval

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal
  rank fusion outperforms condorcet and individual rank learning methods.
  In *Proceedings of SIGIR 2009* (pp. 758-759). ACM.

### Buddhist Psychology

- *Yogacarabhumi-sastra* (瑜伽師地論). Attributed to Maitreya/Asanga,
  translated by Xuanzang (646-648 CE). T30 No.1579. Especially 卷51
  (fascicle 51) on alaya-vijnana.

- *Cheng Weishi Lun* (成唯識論). Xuanzang (659 CE). T31 No.1585.
  The definitive Chinese Yogacara synthesis, especially 卷九-十 on
  asraya-paravrtti and the four wisdoms.

- *Samdhinirmocana Sutra* (解深密經). T16 No.676. The key verse on
  adana-vijnana (阿陀那識甚深細) appears in fascicle 1.

### Taxonomies and Surveys

- Sumers, T. R., et al. (2024). Cognitive Architectures for Language
  Agents (CoALA).

- Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large
  Language Models: A Survey.

- Zhang, P., et al. (2024). A Survey on Retrieval-Augmented Generation.
