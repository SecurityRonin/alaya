# OpenClaw Memory Architecture: Deep Technical Research

> Research date: 2026-02-28
> Source: [openclaw/openclaw](https://github.com/openclaw/openclaw) on GitHub
> Docs: [docs.openclaw.ai/concepts/memory](https://docs.openclaw.ai/concepts/memory)

## 1. Overview

OpenClaw is an open-source autonomous AI agent framework (formerly "Clawdbot",
then "Moltbot") created by Peter Steinberger. It uses messaging platforms
(WhatsApp, Telegram, Discord, Slack, Signal, iMessage, etc.) as its primary
interface and runs a single Gateway process that owns all session and memory state.

The memory system follows a **Markdown-first** architecture: plain `.md` files
in the agent workspace are the canonical source of truth. Everything else --
SQLite indices, embeddings, FTS tables -- is derived and rebuildable from
Markdown.

---

## 2. File Layout and Structure

### 2.1 Workspace Root

Default workspace: `~/.openclaw/workspace` (configurable via
`agents.defaults.workspace`).

```
~/.openclaw/workspace/
  AGENTS.md          # Agent behavior instructions (injected into context every turn)
  SOUL.md            # Identity, tone, personality
  USER.md            # Info about the human user
  TOOLS.md           # Environment-specific tool notes
  MEMORY.md          # Long-term curated memory (durable facts, decisions, preferences)
  HEARTBEAT.md       # Background task checklist
  BOOTSTRAP.md       # First-run instructions (deleted after use)
  memory/
    YYYY-MM-DD.md    # Daily append-only logs (one file per day)
    heartbeat-state.json  # Tracks last heartbeat check timestamps
```

### 2.2 Two-Layer Memory Organization

| Layer | File | Purpose | Injection |
|-------|------|---------|-----------|
| **Long-term** | `MEMORY.md` | Curated durable facts, decisions, preferences, lessons learned | Injected into context every turn in main (direct/private) sessions. NEVER injected in group/shared contexts (security). |
| **Daily logs** | `memory/YYYY-MM-DD.md` | Append-only daily notes, raw context, running logs | NOT auto-injected. Accessed on demand via `memory_search` and `memory_get` tools. Today + yesterday read at session start by agent convention. |

### 2.3 Proposed Future Layout (from research doc)

The research doc (`docs/experiments/research/memory.md`) proposes an extended
"bank" structure for typed memory pages:

```
bank/
  world.md           # Objective facts
  experience.md      # First-person agent experiences
  opinions.md        # Subjective preferences with confidence scores
  entities/
    Peter.md
    The-Castle.md
    ...
```

This is exploratory and not yet implemented in the main codebase.

### 2.4 Derived Index Storage

```
~/.openclaw/memory/<agentId>.sqlite    # Per-agent SQLite database
```

Contains: `meta`, `files`, `chunks`, `chunks_fts` (FTS5), `chunks_vec` (vec0),
`embedding_cache` tables.

---

## 3. MEMORY.md Conventions

### 3.1 What Goes In MEMORY.md

From the default AGENTS.md template:

- Decisions and their rationale
- User preferences (e.g., "prefers concise replies <1500 chars on WhatsApp")
- Durable facts about people, projects, systems
- Lessons learned and mistakes to avoid
- Opinions and evolving beliefs
- Things explicitly requested to be remembered

### 3.2 What Does NOT Go In MEMORY.md

- Raw daily logs (those go in `memory/YYYY-MM-DD.md`)
- Secrets (unless explicitly asked)
- Transient/temporary information

### 3.3 Security Model

MEMORY.md is **only loaded in the main session** (direct chat with the human
owner). It is explicitly excluded from group chats, Discord channels, and
sessions with other people. This prevents personal context from leaking.

### 3.4 Size Constraints

MEMORY.md is injected into the context window every turn, so it directly
consumes tokens. There is a per-file truncation limit:
- `agents.defaults.bootstrapMaxChars` (default: 20,000 chars)
- `agents.defaults.bootstrapTotalMaxChars` (default: 150,000 chars across all bootstrap files)

If MEMORY.md grows too large, it causes more frequent compaction cycles.

---

## 4. Daily Summarization / Memory Maintenance

### 4.1 No Built-In Daily or Weekly Summarization

OpenClaw does **not** have a built-in automated daily or weekly summarization
pipeline. Instead, it relies on:

1. **Agent-driven maintenance**: The AGENTS.md template instructs the agent to
   periodically (every few days, during heartbeats) review recent daily files
   and update MEMORY.md with distilled learnings.

2. **Heartbeat-driven review**: During heartbeat polls (background periodic
   checks), the agent is instructed to:
   - Read through recent `memory/YYYY-MM-DD.md` files
   - Identify significant events, lessons, insights
   - Update `MEMORY.md` with what is worth keeping
   - Remove outdated info from MEMORY.md

3. **Pre-compaction memory flush**: Before auto-compaction erases context,
   a silent agentic turn writes durable memories to disk (see Section 5).

### 4.2 Community Consolidation Patterns

The community project `s1nthagent/openclaw-memory` implements a cron-based
consolidation: every 6 hours, `memory-consolidate.py` scans the last 7 days
of daily notes, extracts significant events, updates a RECENT HISTORY section
in MEMORY.md, and prunes entries older than 7 days.

### 4.3 Research: Retain / Recall / Reflect Loop

The research doc proposes a more structured approach:

- **Retain**: At end of day, add `## Retain` sections with typed, entity-tagged
  narrative facts (e.g., `W @Peter: Currently in Marrakech`)
- **Recall**: Query the derived index for entity-centric, temporal, or
  opinion-based retrieval
- **Reflect**: Scheduled job that updates entity pages, adjusts opinion
  confidence scores, and proposes edits to core memory

This is aspirational and not yet in production code.

---

## 5. Context Persistence Across Sessions

### 5.1 The Core Problem

The agent wakes up fresh each session with no memory. Continuity lives entirely
in files.

### 5.2 Session Start Protocol

Every session, the agent (per AGENTS.md instructions):

1. Reads `SOUL.md` (identity)
2. Reads `USER.md` (who the human is)
3. Reads `memory/YYYY-MM-DD.md` for today + yesterday
4. In main sessions: also reads `MEMORY.md`

These files are injected into the system prompt under "Project Context" by the
bootstrap mechanism.

### 5.3 Pre-Compaction Memory Flush

When the session context approaches auto-compaction limits, OpenClaw triggers
a **silent agentic turn** to flush durable memories to disk.

Implementation in `src/auto-reply/reply/memory-flush.ts`:

```typescript
export const DEFAULT_MEMORY_FLUSH_PROMPT = [
  "Pre-compaction memory flush.",
  "Store durable memories now (use memory/YYYY-MM-DD.md; create memory/ if needed).",
  "IMPORTANT: If the file already exists, APPEND new content only ...",
  "If nothing to store, reply with NO_REPLY.",
].join(" ");
```

Trigger condition:
```
totalTokens >= contextWindow - reserveTokensFloor - softThresholdTokens
```

Configuration:
```json5
{
  agents: {
    defaults: {
      compaction: {
        reserveTokensFloor: 20000,  // default
        memoryFlush: {
          enabled: true,             // default
          softThresholdTokens: 4000, // default
          systemPrompt: "...",
          prompt: "...",
        }
      }
    }
  }
}
```

Key behaviors:
- One flush per compaction cycle (tracked via `memoryFlushCompactionCount`)
- Skipped if workspace is read-only
- Uses `NO_REPLY` token to suppress user-visible output
- Date stamp in filename is resolved to user's timezone

### 5.4 Session Transcripts

Sessions are persisted as JSONL files:
```
~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl
```

Structure: first line is session header, then entries with `id` + `parentId`
(tree structure). Entry types include `message`, `custom_message`, `compaction`,
`branch_summary`.

Session transcripts can optionally be indexed for memory search
(`memorySearch.experimental.sessionMemory: true`).

### 5.5 Compaction

When context exceeds limits, compaction summarizes older conversation into a
persisted summary entry. Future turns see the summary plus recent messages.

Known issues:
- Can produce empty or "Summary unavailable" summaries (GitHub issues #2851, #12921)
- Uses the same (expensive) model as conversation (feature request #7926 for override)
- Can silently destroy context that was loaded into the window

---

## 6. Memory Search: Hybrid Retrieval

### 6.1 Chunking

Markdown files are split into chunks:
- Target: ~400 tokens per chunk (using `tokens * 4` chars heuristic)
- Overlap: ~80 tokens between chunks
- Hash-based deduplication to avoid re-embedding unchanged chunks

Implementation in `src/memory/internal.ts`:
```typescript
export function chunkMarkdown(
  content: string,
  chunking: { tokens: number; overlap: number },
): MemoryChunk[]
```

### 6.2 Hybrid Search Pipeline

The search combines **vector similarity** (cosine) with **BM25 keyword relevance**:

1. **Vector search**: Top `maxResults * candidateMultiplier` results by cosine similarity
2. **BM25 keyword search**: Top `maxResults * candidateMultiplier` by FTS5 rank
3. **Score fusion**: `vectorWeight * vectorScore + textWeight * textScore`
   (default: 0.7 vector + 0.3 text)
4. **Temporal decay** (post-processing)
5. **MMR re-ranking** (post-processing for diversity)

### 6.3 Embedding Providers (Auto-Selected)

Priority order when `memorySearch.provider` is unset:
1. Local (if model path exists) -- default: `embeddinggemma-300m` GGUF (~0.6 GB)
2. OpenAI
3. Gemini
4. Voyage
5. Mistral
6. Disabled

FTS-only fallback mode works when no embedding provider is available.

### 6.4 Query Expansion

For FTS-only mode, `src/memory/query-expansion.ts` extracts meaningful keywords
from conversational queries. Supports stop-word filtering in 8 languages: English,
Spanish, Portuguese, Arabic, Korean, Japanese, Chinese, and mixed-script queries.

### 6.5 Tools Exposed to the Agent

**`memory_search`**: Semantic search over indexed memory. Returns snippets
(~700 char cap), file path, line range, relevance score, and provider info.
Description explicitly says "Mandatory recall step" -- the agent should call
this before answering questions about prior work.

**`memory_get`**: Targeted file read with optional line-range selection.
Gracefully returns `{ text: "", path }` for missing files. Only reads `.md`
files within the workspace memory paths (security boundary).

---

## 7. Temporal Decay

### 7.1 Formula

```
decayedScore = score * e^(-lambda * ageInDays)
lambda = ln(2) / halfLifeDays
```

Default half-life: **30 days**.

| Age | Multiplier |
|-----|-----------|
| Today | 100% |
| 7 days | ~84% |
| 30 days | 50% |
| 90 days | 12.5% |

### 7.2 Evergreen Files

`MEMORY.md` and non-dated files under `memory/` (e.g., topic files like
`memory/projects.md`) are **never decayed**. Only dated daily files
(`memory/YYYY-MM-DD.md`) have temporal decay applied.

### 7.3 Date Extraction

- Dated files: date parsed from filename pattern `memory/YYYY-MM-DD.md`
- Non-memory files: falls back to filesystem `mtime`
- Evergreen memory files: return `null` timestamp (no decay)

Implementation in `src/memory/temporal-decay.ts`.

### 7.4 Configuration

```json5
agents: {
  defaults: {
    memorySearch: {
      query: {
        hybrid: {
          temporalDecay: {
            enabled: true,       // default: false (opt-in)
            halfLifeDays: 30,    // default
          }
        }
      }
    }
  }
}
```

---

## 8. MMR Re-Ranking (Diversity)

### 8.1 Algorithm

Maximal Marginal Relevance (Carbonell & Goldstein, 1998) iteratively selects
results that maximize:

```
MMR = lambda * relevance - (1 - lambda) * max_similarity_to_selected
```

- Similarity computed via **Jaccard tokenized text similarity** (not embeddings)
- Lambda: 0 = max diversity, 1 = pure relevance (default: 0.7)
- Scores normalized to [0, 1] before MMR computation
- Original score used as tiebreaker

### 8.2 Purpose

Prevents returning near-duplicate snippets from daily logs that often repeat
similar information across days.

---

## 9. QMD Backend (Experimental)

An alternative backend using a local sidecar process:

- Combines BM25 + vector search + reranking
- Runs via Bun + node-llama-cpp (auto-downloads GGUF models)
- Isolated per agent under `~/.openclaw/agents/<agentId>/qmd/`
- Supports session JSONL indexing
- Scope-based access control (can restrict by channel/chatType)
- Falls back to builtin SQLite if QMD fails

---

## 10. SQLite Schema

From `src/memory/memory-schema.ts`:

```sql
CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE files (
  path TEXT PRIMARY KEY,
  source TEXT NOT NULL DEFAULT 'memory',  -- 'memory' | 'sessions'
  hash TEXT NOT NULL,
  mtime INTEGER NOT NULL,
  size INTEGER NOT NULL
);

CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'memory',
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  hash TEXT NOT NULL,
  model TEXT NOT NULL,       -- embedding model identifier
  text TEXT NOT NULL,
  embedding TEXT NOT NULL,   -- JSON-serialized float array
  updated_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text, id UNINDEXED, path UNINDEXED, source UNINDEXED,
  model UNINDEXED, start_line UNINDEXED, end_line UNINDEXED
);

-- Optional: sqlite-vec for accelerated vector search
CREATE VIRTUAL TABLE chunks_vec USING vec0(...);

CREATE TABLE embedding_cache (
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  provider_key TEXT NOT NULL,
  hash TEXT NOT NULL,
  embedding TEXT NOT NULL,
  dims INTEGER,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (provider, model, provider_key, hash)
);
```

---

## 11. Architecture Decisions and Design Philosophy

### 11.1 Markdown as Source of Truth

- Human-readable, git-friendly, zero vendor lock-in
- Can be backed up by making workspace a git repo
- Inspired projects like [memsearch](https://github.com/zilliztech/memsearch)
  which generalize this pattern

### 11.2 Agent-Driven Memory Management

Rather than automated pipelines, memory maintenance is delegated to the agent
itself. The agent is instructed (via AGENTS.md) to read, write, and curate
its own memory files. This is a deliberate design choice that treats memory
management as part of the agent's cognitive loop.

### 11.3 Security Boundaries

- MEMORY.md only loaded in private/direct sessions
- `memory_get` tool only reads `.md` files within allowed paths
- Symlinks are ignored when listing memory files
- Session transcript indexing is isolated per agent
- QMD backend supports scope-based access control rules

### 11.4 Cache Stability

System prompt time handling only includes timezone (not dynamic clock) to
keep prompt cache-stable and reduce token churn.

---

## 12. Limitations

### 12.1 No Automated Consolidation

There is no built-in daily or weekly summarization pipeline. Memory
consolidation depends entirely on the agent following its AGENTS.md
instructions during heartbeats. This is unreliable -- the agent may not
always choose to do maintenance, and quality depends on the model's judgment.

### 12.2 MEMORY.md Token Pressure

Because MEMORY.md is injected into every turn's context, a growing MEMORY.md
directly increases token consumption and triggers more frequent compaction.
The truncation limit (20,000 chars default) provides a safety valve but
means content gets silently dropped when it grows too large.

### 12.3 Compaction Can Destroy Context

When auto-compaction fires, it summarizes older conversation. Known issues:
- Empty or "Summary unavailable" summaries (issues #2851, #12921)
- Uses expensive primary model for summarization (no cheaper fallback)
- Memory files loaded into context can be lost if not written to disk
  before compaction fires
- The pre-compaction flush mitigates this but is not guaranteed

### 12.4 No Entity or Relationship Tracking

The current system treats memory as flat text chunks. There is no entity
extraction, relationship graph, or structured knowledge base. The research
doc proposes entity pages with confidence-bearing opinions, but this is
not implemented.

### 12.5 Temporal Decay Is Opt-In and Coarse

Temporal decay defaults to **disabled**. When enabled, it uses a single
half-life parameter for all content, which may not match the actual
relevance decay pattern. A fact about someone's address decays the same
as a fact about today's weather.

### 12.6 Embedding Provider Dependency

Semantic search requires an embedding provider (local model or API key).
Without one, the system falls back to FTS-only mode, which is significantly
less capable for semantic queries. The local model requires ~0.6 GB of disk
and native builds.

### 12.7 Session Memory Is Experimental

Indexing session transcripts for memory search is experimental and has
limitations: results return snippets only, `memory_get` only reads memory
files, and session indexing is sensitive to delta thresholds.

### 12.8 No Conflict Resolution

When MEMORY.md contains contradictory information (e.g., "prefers dark mode"
and a later "switched to light mode"), there is no mechanism to detect or
resolve conflicts. The research doc proposes opinion confidence tracking
but this is not implemented.

### 12.9 Single-Agent Memory Isolation

Each agent has its own SQLite index and workspace. There is no built-in
mechanism for sharing or synchronizing memory across agents.

### 12.10 Git-Based Backup Is Manual

The docs recommend making the workspace a git repo for backup, but this
is not automated. There is no built-in versioning, diffing, or rollback
for memory files.

---

## 13. Comparison with Alaya's Approach

Key differences to consider for Alaya's memory system:

| Aspect | OpenClaw | Potential Alaya Approach |
|--------|----------|------------------------|
| Source of truth | Markdown files | Could use structured formats (TOML, JSON) alongside Markdown |
| Consolidation | Agent-driven (unreliable) | Could automate with scheduled Rust tasks |
| Memory decay | Exponential decay on search scores | Could implement at storage level (pruning old entries) |
| Entity tracking | None (flat text chunks) | Could extract entities and relationships |
| Conflict resolution | None | Could detect contradictions via embedding similarity |
| Context injection | Full MEMORY.md every turn | Could inject only relevant sections |
| Embedding | External providers or local GGUF | Could use local embedding in Rust |
| Search | Hybrid BM25 + vector | Similar, but could add graph-based retrieval |

---

## Sources

- [openclaw/openclaw GitHub Repository](https://github.com/openclaw/openclaw)
- [OpenClaw Memory Documentation](https://docs.openclaw.ai/concepts/memory)
- [OpenClaw System Prompt Documentation](https://docs.openclaw.ai/concepts/system-prompt)
- [Session Management & Compaction Deep Dive](https://docs.openclaw.ai/reference/session-management-compaction)
- [AGENTS.md Template](https://github.com/openclaw/openclaw/blob/main/docs/reference/templates/AGENTS.md)
- [Default AGENTS.md](https://github.com/openclaw/openclaw/blob/main/docs/reference/AGENTS.default.md)
- [Memory Research Notes](https://github.com/openclaw/openclaw/blob/main/docs/experiments/research/memory.md)
- [s1nthagent/openclaw-memory - Community Consolidation](https://github.com/s1nthagent/openclaw-memory)
- [zilliztech/memsearch - Markdown-First Memory Library](https://github.com/zilliztech/memsearch)
- [supermemoryai/openclaw-supermemory](https://github.com/supermemoryai/openclaw-supermemory)
- [Mem0 for OpenClaw](https://mem0.ai/blog/mem0-memory-for-openclaw)
- [Compaction Model Override Feature Request (Issue #7926)](https://github.com/openclaw/openclaw/issues/7926)
- [Compaction Empty Summary Bug (Issue #12921)](https://github.com/openclaw/openclaw/issues/12921)
- [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw)
