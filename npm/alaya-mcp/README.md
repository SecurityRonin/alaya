# alaya-mcp

A local memory engine for AI agents. Stores conversation episodes, consolidates knowledge through a neuroscience-inspired lifecycle, and builds a personal knowledge graph — all in a local SQLite database.

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
3. **Strengthen** — frequently accessed memories grow stronger (Bjork model)
4. **Categorize** — emergent categories form automatically from your knowledge
5. **Forget** — weak, unused memories fade naturally over time

All data stays on your machine in `~/.alaya/memory.db`.

## Links

- [GitHub](https://github.com/SecurityRonin/alaya)
- [Documentation](https://docs.rs/alaya)
- [crates.io](https://crates.io/crates/alaya) (Rust library)

## License

MIT
