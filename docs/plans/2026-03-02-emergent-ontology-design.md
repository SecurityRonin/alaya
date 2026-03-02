# Emergent Category Formation — Design Document

**Date:** 2026-03-02
**Status:** Approved
**Scope:** MVP (flat categories); full Yogacara hierarchy deferred to v2

## Problem

Alaya's semantic store currently classifies knowledge into 4 static types
(Fact, Relationship, Event, Concept) assigned by the LLM provider during
consolidation. There is no emergent categorization — the system cannot
discover that "Alice manages the auth team" and "Bob handles frontend
reviews" both belong to a "team-structure" category unless told explicitly.

Human cognition forms categories organically through experience. In Yogacara
terms, vikalpa (conceptual construction) and nama-rupa (name-and-form)
describe how the mind labels and structures percepts into categories without
explicit instruction. The MVP implements a pragmatic approximation: categories
emerge from clustering patterns in the semantic store during lifecycle
operations.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Flat categories (no hierarchy) | YAGNI; hierarchy is v2 |
| API shape | Graph nodes + materialized tags | Graph enables spreading activation; tags enable fast queries |
| Incremental (consolidation) | Assign to existing categories only | Prevents churn from sparse data |
| Discovery (transform) | Dual-signal clustering + cheap LLM naming | Graph + embedding is robust; LLM only for labels |
| Minimum cluster size | 3 nodes | Matches consolidation corroboration threshold |
| LLM dependency | Optional | Placeholder labels if unavailable |
| Embedding dependency | Optional | Falls back to graph-only clustering |

## Data Model

### New `Category` struct

```rust
pub struct Category {
    pub id: CategoryId,
    pub label: String,              // LLM-generated name ("cooking", "rust-learning")
    pub prototype_node: NodeId,     // Most representative semantic node
    pub member_count: u32,          // Number of semantic nodes in this category
    pub centroid_embedding: Option<Vec<f32>>, // Average embedding of members
    pub created_at: i64,
    pub last_updated: i64,
    pub stability: f32,             // 0.0-1.0, how stable across transform cycles
}
```

### New SQL table

```sql
CREATE TABLE IF NOT EXISTS categories (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    label               TEXT NOT NULL,
    prototype_node_id   INTEGER REFERENCES semantic_nodes(id),
    member_count        INTEGER DEFAULT 0,
    centroid_embedding   BLOB,
    created_at          INTEGER NOT NULL,
    last_updated        INTEGER NOT NULL,
    stability           REAL DEFAULT 0.0
);

ALTER TABLE semantic_nodes ADD COLUMN category_id INTEGER REFERENCES categories(id);
CREATE INDEX IF NOT EXISTS idx_semantic_category ON semantic_nodes(category_id);
```

### Graph integration

Categories are graph nodes via `NodeRef::Category(id)`. Members connect via
`LinkType::MemberOf` edges. This enables spreading activation to flow through
categories — querying "cooking" activates all cooking-related semantic nodes.

## Consolidation Phase (Incremental Assignment)

When `consolidate()` creates new semantic nodes, a new step runs:

```
For each new SemanticNode:
  1. If node has embedding:
     - Cosine similarity to each category's centroid_embedding
     - Best match above threshold (0.6) → assign
  2. If no embedding match, check graph neighborhood:
     - Look at source episodes' existing semantic links
     - If >50% of linked semantic nodes share a category → assign
  3. If match found:
     - Set node.category_id
     - Update category centroid (running average)
     - Update category member_count
     - Create MemberOf link in graph
  4. If no match: leave uncategorized (transform handles it)
```

**Key constraint:** Consolidation never creates new categories. No LLM cost.

## Transform Phase (Category Discovery)

During `transform()`, after dedup/pruning:

```
1. Gather all uncategorized semantic nodes with embeddings
2. If count < 3: skip
3. Cluster:
   a. Pairwise cosine similarity matrix
   b. Merge nodes above 0.7 threshold
   c. For each cluster >= 3 members:
      - Check graph support (shared Hebbian links)
      - Combined score = 0.6 * embedding_sim + 0.4 * graph_overlap
      - Form category if combined > threshold
4. For each new cluster:
   a. Pick prototype (highest corroboration_count)
   b. Compute centroid embedding (mean of members)
   c. Cheap LLM: "Label this group in 2-3 words: [top 5 contents]"
   d. Create Category, assign members, create MemberOf links
5. Maintenance on existing categories:
   a. Detect member drift (embedding far from centroid)
   b. Merge converging categories (centroids > 0.85 similarity)
   c. Update stability (survives transform → stability increases)
   d. Dissolve unstable categories (stability < 0.2 after 3+ cycles)
   e. Garbage-collect empty categories (all members forgotten)
```

**LLM cost:** One cheap call per new category only.

## API Surface

### New public methods

```rust
impl AlayaStore {
    pub fn categories(&self, min_stability: Option<f32>) -> Result<Vec<Category>>;
    pub fn node_category(&self, node_id: NodeId) -> Result<Option<Category>>;
}
```

### Extended existing APIs

```rust
pub struct KnowledgeFilter {
    pub node_type: Option<SemanticType>,   // existing
    pub min_confidence: Option<f32>,        // existing
    pub category: Option<String>,           // NEW
}

pub struct Query {
    pub text: String,                       // existing
    pub top_k: usize,                       // existing
    pub boost_categories: Option<Vec<String>>, // NEW
}
```

### MCP tools

| Tool | Change |
|------|--------|
| `categories` | New tool: list emergent categories |
| `knowledge` | Extended: optional `category` filter |
| `recall` | Extended: optional `boost_categories` |

No new lifecycle methods. Scheduling unchanged.

## Error Handling

- **No embeddings:** Falls back to graph-only clustering (neighbor counting / Hebbian link overlap)
- **No LLM:** Placeholder labels from prototype node content (first 3 words); updated on next successful transform
- **Category drift:** Members diverge → stability drops → dissolve after 3+ cycles below 0.2
- **Category merge:** Two centroids converge > 0.85 → higher-stability absorbs lower
- **Empty categories:** All members archived by Bjork decay → garbage-collected during transform

## Testing Strategy

### Unit tests
- Category CRUD
- Incremental assignment via embedding similarity
- Incremental assignment via graph neighbors
- Uncategorized when no match
- Centroid recomputation after new member

### Integration tests
- Full lifecycle: episodes → consolidate → transform → categories emerge
- Graph-only fallback (no embeddings)
- Stability tracking across multiple transform cycles
- Category merging when clusters converge
- Category dissolution when members diverge
- Forgetting integration (empty category cleanup)
- NoOpProvider path (placeholder labels, no LLM)

## Future Work (v2 — Yogacara Grounding)

- **Hierarchy:** Categories form parent-child trees via `is-a` links (vikalpa)
- **Prototype theory:** Categories defined by prototypical exemplars, not boundaries (nama-rupa)
- **Category seeds (bija):** Dormant categories that can re-emerge when new evidence arrives
- **Conceptual transformation (asraya-paravrtti):** Categories themselves evolve through use
- **Cross-domain bridging:** Spreading activation through category nodes enables analogical reasoning
