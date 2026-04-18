# alaya-mcp

The only memory engine with neuroscience-grounded memory dynamics — Bjork dual-strength forgetting, retrieval-induced suppression, and Hebbian co-activation — in a zero-dependency embeddable Rust library.

Stores conversation episodes, consolidates knowledge through a cognitive lifecycle, and builds a personal knowledge graph — all in a local SQLite database.

## Install

### Claude Code

Add to `~/.claude/claude_code_config.json`:

```json
{
  "mcpServers": {
    "alaya": {
      "command": "npx",
      "args": ["-y", "alaya-mcp"]
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "alaya": {
      "command": "npx",
      "args": ["-y", "alaya-mcp"],
      "env": {
        "ALAYA_LLM_API_KEY": "sk-...",
        "ALAYA_LLM_API_URL": "https://api.openai.com/v1/chat/completions",
        "ALAYA_LLM_MODEL": "gpt-4o-mini"
      }
    }
  }
}
```

The `ALAYA_LLM_*` env vars are optional — they enable automatic knowledge extraction. Without them, the agent extracts knowledge from its own context.

## MCP Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a conversation episode |
| `recall` | Semantic search across memories |
| `status` | Memory system health and stats |
| `knowledge` | Browse extracted knowledge |
| `learn` | Store pre-extracted facts and relationships |
| `categories` | View emergent category taxonomy |
| `preferences` | Track crystallized user preferences |
| `neighbors` | Explore the knowledge graph |
| `lifecycle` | Trigger maintenance (strengthen, transform, forget) |
| `configure` | Set embedding provider and other options |

## How It Works

1. **Remember** — store conversation episodes as they happen
2. **Consolidate** — extract facts, relationships, and concepts from episodes
3. **Strengthen** — co-retrieved memories strengthen their connections (Hebbian LTP); retrieving memory A actively suppresses competing memories B and C (retrieval-induced forgetting)
4. **Categorize** — emergent categories form automatically from your knowledge
5. **Forget** — Bjork dual-strength decay separates storage strength (how well-encoded) from retrieval strength (how easily found); weak memories fade, strong ones persist
6. **Crystallize** — implicit preferences emerge from accumulated impressions (vasana), no LLM required

All data stays on your machine in `~/.alaya/memory.db`.

## Why Alaya

| Problem | File-based memory | Alaya |
|---|---|---|
| **Token waste** | Full-context injection (~35K tokens/msg) | Ranked retrieval — only top-k relevant memories |
| **No structure** | Everything in one file | Three typed stores: episodes, knowledge, preferences |
| **No forgetting** | Files grow until manually curated | Bjork dual-strength decay + retrieval-induced forgetting |
| **No associations** | Flat files, no links | Hebbian co-retrieval graph (LTP/LTD) |
| **Brittle preferences** | Agent-authored summary, drifts | Implicit preferences emerge from impressions (vasana) |
| **LLM required** | Can't function without one | Graceful degradation: no embeddings → BM25-only |

## Links

- [GitHub](https://github.com/SecurityRonin/alaya)
- [Documentation](https://docs.rs/alaya)
- [crates.io](https://crates.io/crates/alaya) (Rust library)
- [PyPI](https://pypi.org/project/alaya-memory/) (Python bindings)

## License

MIT

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=c18dc510-7aec-427a-868b-2753233f9a35" />
