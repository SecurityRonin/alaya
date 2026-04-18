# alaya-memory

A persistent memory engine for conversational AI agents — Python bindings for [Alaya](https://github.com/SecurityRonin/alaya).

## Install

```bash
pip install alaya-memory
```

## Quick start

```python
from alaya import Alaya, NewEpisode, Query

# Open (or create) a memory store
mem = Alaya("agent.db")

# Store an episode
mem.store_episode(NewEpisode(
    content="User prefers concise answers",
    session_id="sess-001",
    role="user",
))

# Query
results = mem.query(Query(text="communication preferences", limit=5))
for r in results:
    print(r.score, r.content)

# Run the full dream cycle (consolidate → transform → forget)
mem.dream()
```

## Features

- **Episodic memory** — store and retrieve conversation turns
- **Semantic consolidation** — episodes are distilled into durable knowledge nodes
- **Forgetting curves** — weak or contradicted knowledge decays over time
- **Knowledge graph** — typed links between episodes, nodes, preferences, and categories
- **Encrypted storage** — optional SQLCipher encryption (`open_encrypted`)
- **MCP server** — expose memory over the Model Context Protocol via `alaya-mcp` on npm

## Documentation

See the [Alaya repository](https://github.com/SecurityRonin/alaya) for full API docs and the Rust crate.

## License

MIT
