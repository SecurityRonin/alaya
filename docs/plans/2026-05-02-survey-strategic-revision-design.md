# Survey Strategic Revision Design — May 2026

## Context

The survey "A Taxonomy of Memory Architectures for LLM-Based Agents" currently covers 138 systems (133 in matrix) with an April 2026 cutoff. Since that cutoff, the field has accelerated dramatically: 7 competing surveys have appeared, ~15 new systems merit inclusion, ~10 existing entries need updates, an RL-based memory management trend has coalesced (5+ papers), a dedicated ICLR 2026 workshop on agent memory launched, and memory security has emerged as a subfield with its own survey and attack taxonomy.

This design covers a strategic revision that maintains the paper's taxonomic authority while positioning it against the new landscape.

## Scope

### 1. New Systems to Add (~18 entries, corpus 138 → ~156)

#### Production/Commercial Systems (4)
| System | Category | Why Add |
|--------|----------|---------|
| Claude Code Memory (CLAUDE.md + Auto Memory + Auto Dream) | File-Based | Most deployed file-based agent memory; 4-layer arch with automated consolidation |
| Claude Memory Tool (API) | Production | Official Anthropic memory primitive (GA); 6 CRUD ops |
| Claude Managed Agents Memory | Production | Enterprise persistent memory (public beta Apr 2026); Netflix, Rakuten |
| AWS Bedrock AgentCore Memory | Production | First major cloud-managed memory service (GA Oct 2025) |

#### Dedicated Memory Systems (6)
| System | Category | Why Add |
|--------|----------|---------|
| Memanto | Dedicated | SOTA LoCoMo 87.1% without graphs; information-theoretic retrieval |
| MAGMA | Dedicated | ACL 2026 main; 4 orthogonal graphs + vector DB |
| Honcho | Dedicated | Dialectic user modeling; novel preference mechanism |
| Hermes Agent | Framework | 64K+ stars; 8 pluggable memory providers; Curator agent |
| ByteRover | Dedicated | Markdown context tree; no vector DB; full-text search |
| RetainDB | Dedicated | Cloud-based; delta compression |

#### Research Architectures (6)
| System | Category | Why Add |
|--------|----------|---------|
| O-Mem | Research | 3-tier persona memory (OPPO); active user profiling |
| MemVerse | Research | Multimodal lifelong learning; parametric distillation |
| Dynamic Cheatsheet | Research | Procedural memory via self-curation; EACL 2026 |
| Agent Workflow Memory | Research | Procedural; ICML 2025 poster |
| Animesis/CMA | Research | Governance-first; "Memory-as-Ontology" paradigm |
| NGC (Neural Garbage Collection) | Research | RL-learned inference-time forgetting; Stanford |

#### Coding Agent Memory (2)
| System | Category | Why Add |
|--------|----------|---------|
| claude-mem (thedotmack) | Coding Agent | Most-referenced Claude Code memory plugin; ChromaDB + MCP |
| coolmanns/openclaw-memory-architecture | Coding Agent | 12-layer stack; activation/decay; Hebbian search |

### 2. Existing Systems to Update (~10)

| System | Key Changes |
|--------|-------------|
| OpenClaw | Active Memory plugin (passive→active), Dreaming formalized (Light/REM/Deep, scoring thresholds), QMD backend, 347K stars |
| Mem0 | Graph memory GA, new single-call algorithm, Apache Cassandra + Valkey backends, FalkorDB plugin, 48K stars |
| Letta (MemGPT) | MemFS, Conversations API (shared memory), sleep-time compute, Letta Code App |
| Zep/Graphiti | 20K stars, Community Edition deprecated, self-host requires Graphiti + graph DB |
| CrewAI | Unified Memory class, adaptive-depth recall, composite scoring, MemoryMatch explainability |
| LangChain/LangGraph | LangGraph 1.0 checkpointing, LangMem SDK (semantic/episodic/procedural), legacy memory deprecated |
| AG2 (AutoGen) | MemoryStream API, Microsoft AutoGen in maintenance mode |
| LlamaIndex | Memory class + MemoryBlock abstractions, token-ratio flush |
| EverMemOS | 93% LoCoMo reasoning accuracy, 5 retrieval modes, evaluation framework |
| mcp-memory-service | v10+, Knowledge Graph Dashboard, REST API, Session Harvest hook |

### 3. Taxonomy Extensions

#### 3a. Forgetting Axis: New Category — "RL-Learned Eviction"
Current forgetting categories: TTL, LRU, LLM-judgment, Ebbinghaus decay.
Add: **RL-learned** — forgetting policy trained end-to-end from task reward.
Systems: NGC (inference-time KV cache), AtomMem (CRUD policy via POMDP), MemRL (Q-value weighted), Memory-R1 (PPO/GRPO trained manager), MemAgent (ICLR 2026 Oral, DAPO-extended).

This is a genuine new trend (5+ papers in 6 months) that our taxonomy should capture.

#### 3b. Lifecycle Axis: Add "Meta-Evolution"
MemEvolve introduces dual-loop meta-evolution: inner loop for experience accumulation, outer loop for architectural mutation. This goes beyond individual lifecycle operations — the architecture itself evolves.

#### 3c. Governance Layer (Cross-Cutting)
Animesis/CMA and SSGM introduce memory governance as a cross-cutting concern. Three axioms (Memory Inalienability, Model Substitutability, Governance Precedes Function). This doesn't fit neatly into our four axes but should be discussed.

### 4. New Sections and Subsections

#### 4a. "State of the Field: May 2026" subsection (§06 or §09)
Position our survey against 7 competing surveys:
- Differentiate: we are the only survey with (a) per-system machine-readable coding, (b) reproducible benchmark baselines, (c) preference emergence axis, (d) file-based and coding agent categories
- Acknowledge where they complement us (security taxonomy from 2604.16548, RL-learned memory from 2603.07670)

#### 4b. "RL-Based Memory Management" subsection (§04 or §09)
Dedicated discussion of the RL cluster: NGC, AtomMem, MemRL, Memory-R1, MemAgent, Mem-alpha, Live-Evo. This is the clearest new trend in the field — treating memory management as a learnable policy rather than a hand-designed heuristic.

#### 4c. "Memory Security" paragraph (§09 or §10)
Cite: MEMFLOW (memory control flow attacks), MINJA (query-only injection), MEXTRA (extraction), EHR poisoning, Mnemonic Sovereignty survey. Note this as an emerging concern our taxonomy does not yet address.

#### 4d. "ICLR 2026 MemAgents Workshop" reference
Evidence of field maturation — dedicated venue for agent memory research.

### 5. Benchmark Table Updates

Add to benchmark-table-fragment.tex:
| System | Scores |
|--------|--------|
| Memanto | LoCoMo 87.1%, LongMemEval 89.8% |
| SimpleMem/Omni-SimpleMem | LoCoMo F1 0.613, Mem-Gallery F1 0.810 |
| EverMemOS | 93% LoCoMo reasoning accuracy |

Add new benchmarks to discussion in §09:
- MemoryAgentBench (ICLR 2026): multi-turn incremental evaluation
- Evo-Memory (DeepMind): streaming benchmark for self-evolving memory
- PersonaMem-v2: implicit personalization (preference axis)
- Memora: forgetting-aware metric (FAMA)
- Agent Memory Benchmark (Vectorize): agentic multi-query mode

### 6. Statistics Refresh

After adding ~18 systems (corpus → ~156):
- Rerun `generate_coding_records.py` for CSV/JSON
- Recompute co-occurrence matrix
- Recompute feature prevalence counts
- Recompute RAG distribution
- Update feature-breadth histogram
- Update bar chart category counts
- Update all "138" references → new count
- Update all derived percentages

### 7. Reference Updates

Add ~25 new BibTeX entries:
- NGC, MAGMA, Memanto, Honcho, O-Mem, MemVerse, Dynamic Cheatsheet, AWM, APC, Animesis
- AtomMem, MemRL, Memory-R1, MemAgent, MemEvolve, Live-Evo, Omni-SimpleMem
- 7 competing surveys
- MEMFLOW, MINJA, Mnemonic Sovereignty
- ICLR MemAgents Workshop
- Bedrock AgentCore docs

### 8. Cutoff Date

Update from "April 2026" to "May 2026" in:
- §02-methodology.tex (2 occurrences)
- §10-limitations.tex (1 occurrence)
- coding-records.json metadata

## Implementation Approach

This revision is large but can be parallelized into independent workstreams:

1. **Matrix & Data** — Add 18 rows to system-matrix.tex, rerun statistics
2. **Taxonomy Text** — Write RL-based memory subsection, governance discussion
3. **System Descriptions** — Write entries for new production/dedicated/research systems
4. **Competing Surveys** — Write "State of the Field" positioning subsection
5. **Benchmarks** — Update benchmark table + add new benchmark discussion
6. **References** — Add 25 BibTeX entries
7. **Statistics Sweep** — Update all counts/percentages/figures
8. **Security** — Write memory security paragraph

## Non-Goals

- We are NOT restructuring the four-axis taxonomy (it's validated and differentiated)
- We are NOT adding a fifth axis (governance is discussed, not formalized as an axis)
- We are NOT replacing any existing content — only extending and updating
- We are NOT re-running our own benchmarks (those are already published)

## Risk

- **Corpus inflation**: Adding 18 systems in one revision could dilute the corpus. Mitigation: each addition must meet the same inclusion criteria (persistent memory, 2023-2026, usable system or reference impl, LLM-agent-focused).
- **Statistics cascade**: Changing the corpus size triggers updates in ~20 places. Mitigation: use the Python parser to recompute all statistics programmatically.
- **Competing survey positioning**: Must be respectful but firm. We differentiate on methodology (per-system coding + benchmarks), not on coverage breadth.
