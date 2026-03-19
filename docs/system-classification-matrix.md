# Memory System Classification Matrix

Extracted from the survey paper (sections 04-taxonomy, 05-systems, 06-comparison, 07-memory-md-problem, 08-yogacara, 09-open-problems, benchmark-table-fragment).

## Legend

- **Category**: dedicated / framework / coding / file / research / additional
- **Structure**: W=Working, E=Episodic, S=Semantic, P=Procedural, I=Implicit
- **Retrieval**: N=Naive RAG, A=Advanced RAG, M=Modular RAG
- **Lifecycle**: F=Formation, C=Consolidation, Fg=Forgetting, T=Transformation
- **Preference**: none / extract / opinion-track / emergent
- **Graph**: yes/no/static/dynamic
- **Source**: file and line where classification data found

## Key Finding: No Per-System Structured Data Exists

**IMPORTANT**: The paper does NOT contain a per-system classification table mapping each of the 88 systems to all four axes. The data exists as:

1. **Detailed prose descriptions** for ~20 systems in `05-systems.tex` (lines 21-302)
2. **Aggregate statistics** in `06-comparison.tex` tables (feature prevalence, temporal trends, RAG distribution, capability scores)
3. **Representative examples** in taxonomy axis descriptions in `04-taxonomy.tex`
4. **Benchmark scores** for ~18 systems in `benchmark-table-fragment.tex`
5. **Brief mentions** of additional systems scattered across sections 06, 07, 08, 09

The 88 systems break down as: Dedicated (27), Framework (13), Coding Agent (12), File-Based (2), Research (12), plus 22 additional classified but not in primary tables.

---

## Systems with Detailed Classification Data (from 05-systems.tex)

### Production Systems (dedicated)

```
SYSTEM_NAME              | Category  | Structure   | Retrieval | Lifecycle     | Preference     | Graph          | Source
Mem0                     | dedicated | E,S,I       | A         | F,Fg,C        | extract        | static (Neo4j) | 05-systems:21-43
Mem0g (graph variant)    | dedicated | E,S,I       | A         | F,Fg,C        | extract        | static (Neo4j) | 05-systems:24,40-43
Zep / Graphiti           | dedicated | E,S         | M         | F,Fg(bi-temp) | none           | static (Neo4j) | 05-systems:45-66
Letta (MemGPT)           | dedicated | W,E,S       | N         | F,Fg(summ)    | none           | no             | 05-systems:68-88
Hindsight                | dedicated | E,S,I       | A         | F             | opinion-track  | no             | 05-systems:90-110
```

### Standalone Memory Servers (dedicated)

```
SYSTEM_NAME              | Category  | Structure   | Retrieval | Lifecycle     | Preference     | Graph          | Source
Motorhead                | dedicated | W           | N         | F,Fg(window)  | none           | no             | 05-systems:118-121
Engram                   | dedicated | S           | N         | F             | none           | no             | 05-systems:122-124
OpenViking               | dedicated | W,E,S       | N         | F             | none           | no             | 05-systems:126-132
```

### Framework-Level Memory (framework)

```
SYSTEM_NAME              | Category  | Structure   | Retrieval | Lifecycle     | Preference     | Graph          | Source
LangChain                | framework | W           | N         | F,Fg(trunc)   | none           | no             | 05-systems:140-143; 06:185,195
LlamaIndex               | framework | W,E         | N         | F,Fg(FIFO)    | none           | no             | 05-systems:144-147; 06:185,195
LangMem SDK              | framework | S,P         | N         | F             | extract        | no             | 05-systems:148-156
Haystack                 | framework | W           | N         | F             | none           | no             | 06:195
Agency Swarm             | framework | W           | N         | F             | none           | no             | 06:185,195,561
```

### Coding Agent Memory (coding)

```
SYSTEM_NAME              | Category  | Structure   | Retrieval | Lifecycle     | Preference     | Graph          | Source
memsearch                | coding    | E,S         | M         | F             | none           | no             | 05-systems:167-169; 07:55-57
QMD                      | coding    | E,S         | M         | F             | none           | no             | 05-systems:170-173; 07:51-53
Beads                    | coding    | E,S         | A         | F             | none           | yes(dep graph) | 05-systems:174-176
Basic Memory             | coding    | S           | A         | F             | none           | yes(wiki-link) | 05-systems:177-182
Clawdbot-Next TGAA       | coding    | E,S         | A         | F             | none           | no             | 07:59-61
MemoryMesh               | coding    | ?           | N         | F             | none           | no             | 06:196
ClaudeHistory Cloud      | coding    | ?           | N         | F             | none           | no             | 06:196
```

### File-Based Memory (file)

```
SYSTEM_NAME              | Category  | Structure   | Retrieval | Lifecycle     | Preference     | Graph          | Source
MEMORY.md pattern        | file      | W,S         | N         | F             | none           | no             | 05-systems:190-203
Claudesidian             | file      | W,S         | N         | F             | none           | yes(wiki-link) | 05-systems:194-196
```

### Research Architectures (research)

```
SYSTEM_NAME              | Category  | Structure   | Retrieval | Lifecycle     | Preference     | Graph          | Source
Generative Agents        | research  | E,S         | A         | F,C(reflect)  | none           | no             | 05-systems:209-229; 06:84,186
HippoRAG                 | research  | S           | M         | F             | none           | static(KG)     | 05-systems:231-256; 06:187
HippoRAG 2              | research  | S           | M         | F             | none           | static(KG)     | 05-systems:247-249
SYNAPSE                  | research  | E,S         | M         | F,Fg(decay),T | none           | dynamic        | 05-systems:258-279; 06:187
Mem-alpha                | research  | W,E,S       | A         | F,C,Fg(RL)    | none           | no             | 05-systems:281-302
```

---

## Systems with Partial Classification Data (from other sections)

### Mentioned in 06-comparison.tex and other sections

```
SYSTEM_NAME              | Category    | Structure   | Retrieval | Lifecycle     | Preference     | Graph          | Source
Supermemory              | dedicated   | ?,I         | ?         | F             | extract        | ?              | 04:129; 06:155; 09:40
Memobase                 | dedicated   | ?,I         | ?         | F             | extract        | ?              | 04:129; benchmark:46
TiMem                    | dedicated   | ?           | ?         | F             | ?              | ?              | 06:155; benchmark:41-42
Papr Memory              | dedicated   | ?           | ?         | F             | ?              | ?              | 06:155
EverMemOS                | dedicated   | E,S         | ?         | F,C           | ?              | yes            | 06:564; 09:74; 11:19
MemOS                    | dedicated   | ?           | ?         | F             | ?              | yes            | 06:564; benchmark:45
Vestige                  | dedicated   | ?           | ?         | ?             | ?              | yes            | 06:564
MemoryOS                 | dedicated   | W,E,S       | ?         | F             | ?              | ?              | 05-systems:86
Backboard                | dedicated   | ?           | ?         | F             | ?              | ?              | benchmark:30
Mastra Obs. Memory       | dedicated   | ?           | ?         | F             | ?              | ?              | benchmark:28-29,39
EmergenceMem             | dedicated   | ?           | ?         | F             | ?              | ?              | benchmark:36,40
MemMachine               | dedicated   | ?           | ?         | F             | ?              | ?              | benchmark:35
OpenAI Memory            | dedicated   | ?           | ?         | F             | ?              | ?              | benchmark:55
LightMem                 | research    | E,S         | ?         | F,C(sleep)    | ?              | ?              | 06:85; 08:73,105; 09:73; 11:19
A-MEM                    | dedicated?  | ?,I         | A         | F             | extract        | ?              | 06:186; 08:118
Memory-R1                | research    | ?           | ?         | F,Fg(RL)      | ?              | ?              | 06:160; 09:101
MemRL                    | research    | ?           | ?         | F,Fg(RL)      | ?              | ?              | 06:160; 05:299; 09:101
AgeMem                   | research    | ?           | ?         | F,Fg(RL)      | ?              | ?              | 06:160; 05:300
Memvid                   | dedicated   | E           | A         | F             | ?              | no             | 06:200,562
MemoryBank               | dedicated?  | ?           | N         | F,Fg(Ebbhaus) | ?              | ?              | 04:142; 09:14
CortexGraph              | dedicated?  | ?           | ?         | F,Fg(Ebbhaus) | ?              | ?              | 09:15
G-Memory                 | research    | ?           | ?         | F             | ?              | ?              | 09:118
Qwen-Agent               | framework   | W           | N         | F             | none           | no             | 06:561
AutoGPT                  | framework   | W           | N         | F             | none           | no             | 06:561
```

---

## Systems Only in Benchmark Table (benchmark-table-fragment.tex)

These systems appear only with benchmark scores, no detailed taxonomy classification:

```
SYSTEM_NAME              | Benchmark Scores
Mastra Obs. Memory       | LongMemEval: 94.87% (GPT-5-mini), 93.27% (Gemini-3-Pro), 84.23% (GPT-4o)
Backboard                | LoCoMo: 90.10%, LongMemEval: 93.40%
EverMemOS                | LoCoMo: 92.30%, LongMemEval: 83.00%
Hindsight                | LoCoMo: 89.61%, LongMemEval: 91.40% (Gemini-3-Pro)
MemMachine v0.2          | LoCoMo: 84.87%
EmergenceMem Integrated  | LongMemEval: 86.00%
Supermemory              | LongMemEval: 85.20% (Gemini-3-Pro), 81.60% (GPT-4o)
EmergenceMem Simple      | LongMemEval: 82.40%
TiMem                    | LoCoMo: 75.30%, LongMemEval: 78.96%
MemOS                    | LoCoMo: 75.80%
Memobase v0.0.37         | LoCoMo: 75.78%
Zep / Graphiti           | LoCoMo: 75.14%, LongMemEval: 71.20%, DMR: 94.80%
Letta (MemGPT)           | LoCoMo: 74.00%, DMR: 93.40%
Mem0 (graph)             | LoCoMo: 68.44%
Mem0                     | LoCoMo: 66.88%
LangMem                  | LoCoMo: 58.10%
OpenAI Memory            | LoCoMo: 52.90%
Zep (DMR-only)           | DMR: 98.20%
```

---

## Aggregate Statistics (from 06-comparison.tex tables)

### Feature Prevalence (Table 1, n=88)
| Feature | Count | % |
|---------|-------|---|
| Has episodic memory | 48 | 54.5 |
| Has semantic memory | 45 | 51.1 |
| Has procedural memory | 6 | 6.8 |
| Has implicit/preference mem | 27 | 30.7 |
| Has working memory | 31 | 35.2 |
| Uses BM25/sparse search | 19 | 21.6 |
| Uses vector/dense search | 59 | 67.0 |
| Uses graph traversal | 28 | 31.8 |
| Uses hybrid fusion (RRF) | 14 | 15.9 |
| Uses reranking | 12 | 13.6 |
| Has any forgetting | 34 | 38.6 |
| Has consolidation | 16 | 18.2 |
| Has contradiction resolution | 7 | 8.0 |
| Has transformation | 11 | 12.5 |
| Has any preference learning | 27 | 30.7 |
| via LLM extraction | 16 | 18.2 |
| via emergent/accumulated | 4 | 4.5 |
| Has any graph structure | 28 | 31.8 |
| Static KG (LLM-built) | 20 | 22.7 |
| Dynamic/Hebbian | 4 | 4.5 |
| Requires 0 external services | 28 | 31.8 |
| Requires 1-2 external services | 37 | 42.0 |
| Requires 3+ external services | 6 | 6.8 |
| Requires LLM for memory ops | 62 | 70.5 |

### RAG Distribution (Table 3, n=88)
| RAG Type | Count | % | Representative Systems |
|----------|-------|---|----------------------|
| Naive RAG | 42 | 47.7 | LangChain, LlamaIndex, Engram, Agency Swarm |
| Advanced RAG | 30 | 34.1 | Mem0, Generative Agents, Hindsight, A-MEM |
| Modular RAG | 16 | 18.2 | Zep/Graphiti, SYNAPSE, HippoRAG, QMD |

### Capability Score Distribution (Table 5, n=88)
| Score | Systems | % | Examples |
|-------|---------|---|----------|
| 0-1 | 24 | 27.3 | Agency Swarm, Qwen-Agent, AutoGPT |
| 2-3 | 33 | 37.5 | LangChain, Engram, Memvid, HippoRAG |
| 4-5 | 20 | 22.7 | Mem0, Hindsight, SYNAPSE, LightMem |
| 6-7 | 9 | 10.2 | Zep/Graphiti, EverMemOS, MemOS, Vestige |
| 8 | 2 | 2.3 | --- |

### Temporal Trends (Table 2, n=88)
| Year | N | Graph% | Forgetting% | Consolidation% | 0-dep% |
|------|---|--------|-------------|----------------|--------|
| 2023 | 9 | 11.1 | 33.3 | 22.2 | 33.3 |
| 2024 | 14 | 28.6 | 28.6 | 14.3 | 28.6 |
| 2025 | 42 | 33.3 | 40.5 | 19.0 | 31.0 |
| 2026 | 23 | 39.1 | 47.8 | 21.7 | 30.4 |

---

## Gap Analysis: Missing Data

The paper surveys 88 systems but only provides detailed per-system taxonomy classification for approximately 20 systems in Section 5. The remaining ~68 systems are classified in aggregate (contributing to the statistics in Tables 1-5) but their individual classifications are NOT published in the paper text.

### Systems explicitly named but without full 4-axis classification (~23 systems):
Supermemory, Memobase, TiMem, Papr Memory, EverMemOS, MemOS, Vestige, MemoryOS, Backboard, Mastra Obs. Memory, EmergenceMem, MemMachine, OpenAI Memory, LightMem, A-MEM, Memory-R1, MemRL, AgeMem, Memvid, MemoryBank, CortexGraph, G-Memory, MEMTRACK

### Systems entirely unnamed (~45 systems):
The paper states 88 systems across 5 categories (27+13+12+2+12=66 in primary tables, plus 22 additional). Only ~43 distinct system names appear anywhere in the paper. The remaining ~45 systems contributed to aggregate statistics but are never individually named in the surveyed tex files.

### Where the missing data likely resides:
- The authors' internal spreadsheet/database used to compute the aggregate statistics
- Supplementary materials not included in the main paper
- The bibliography (references.bib) which would list all 88+ citations
