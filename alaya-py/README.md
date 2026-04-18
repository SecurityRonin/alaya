# alaya-memory

The only memory engine with neuroscience-grounded memory dynamics — Bjork dual-strength forgetting, retrieval-induced suppression, and Hebbian co-activation — Python bindings for [Alaya](https://github.com/SecurityRonin/alaya).

Stores conversation episodes, consolidates knowledge through a cognitive lifecycle, and builds a personal knowledge graph — all in a local SQLite database.

## Install

```bash
pip install alaya-memory
```

## Quick start

```python
from alaya import Alaya, NewEpisode, Query

mem = Alaya("agent.db")

# Store an episode
mem.store_episode(NewEpisode(
    content="User prefers concise answers",
    session_id="sess-001",
    role="user",
))

# Query with hybrid retrieval (BM25 + vector + graph)
results = mem.query(Query(text="communication preferences", limit=5))
for r in results:
    print(r.score, r.content)

# Run the full dream cycle (consolidate → transform → forget)
mem.dream()
```

## Why Alaya

| Problem | File-based memory | Alaya |
|---|---|---|
| **Token waste** | Full-context injection (~35K tokens/msg) | Ranked retrieval — only top-k relevant memories |
| **No structure** | Everything in one file | Three typed stores: episodes, knowledge, preferences |
| **No forgetting** | Files grow until manually curated | Bjork dual-strength decay + retrieval-induced forgetting |
| **No associations** | Flat files, no links | Hebbian co-retrieval graph (LTP/LTD) |
| **Brittle preferences** | Agent-authored summary, drifts | Implicit preferences emerge from impressions (vasana) |
| **LLM required** | Can't function without one | Graceful degradation: no embeddings → BM25-only |

## Features

- **Episodic memory** — store and retrieve conversation turns with session context
- **Semantic consolidation** — episodes distilled into durable knowledge nodes via cognitive lifecycle
- **Forgetting curves** — Bjork dual-strength decay separates storage from retrieval strength
- **Retrieval-induced forgetting** — retrieving A actively suppresses competing memories B and C
- **Hebbian graph** — co-retrieved memories strengthen connections (LTP); unused links decay (LTD)
- **Implicit preferences** — behavioral patterns crystallize from accumulated impressions (vasana), no LLM required
- **Encrypted storage** — optional SQLCipher encryption via `open_encrypted(path, key)`
- **Graceful degradation** — works without embeddings (BM25-only) or without an LLM

## API

```python
mem = Alaya("agent.db")                    # persistent store
mem = Alaya.in_memory()                    # ephemeral (tests)
mem = Alaya.open_encrypted("agent.db", key)  # SQLCipher encrypted

mem.store_episode(episode)                 # store conversation turn
mem.query(Query(text=..., limit=5))        # hybrid retrieval
mem.knowledge(filter=None)                 # semantic nodes
mem.preferences(domain=None)               # crystallized preferences
mem.categories(min_stability=None)         # emergent taxonomy
mem.neighbors(node_type, node_id, depth)   # graph traversal
mem.status()                               # memory statistics

mem.consolidate()                          # episodes → knowledge
mem.transform()                            # dedup, decay, prune
mem.forget()                               # Bjork strength decay
mem.dream()                                # full cycle: consolidate + transform + forget
```

## Links

- [GitHub](https://github.com/SecurityRonin/alaya)
- [Documentation](https://docs.rs/alaya)
- [crates.io](https://crates.io/crates/alaya) (Rust library)
- [npm](https://www.npmjs.com/package/alaya-mcp) (MCP server)

## License

MIT
