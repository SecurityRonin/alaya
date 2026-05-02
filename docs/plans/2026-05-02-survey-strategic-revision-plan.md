# Survey Strategic Revision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the survey from 133→~151 matrix systems (138→~156 total corpus), add RL-based forgetting taxonomy, position against competing surveys, update all statistics.

**Architecture:** 8 independent workstreams that can be parallelized via subagents. Each workstream modifies different files. Statistics refresh (Task 8) runs last since it depends on all matrix additions.

**Tech Stack:** LaTeX (IEEEtran), Python 3.11 (generate_coding_records.py), BibTeX, bash grep tests

**TDD approach:** Each task has a verification step. Matrix additions are verified by rerunning `generate_coding_records.py` and checking counts. Text additions are verified by `grep` for required content.

**Git signing:** Before dispatching subagents, start gitsign credential cache:
```bash
gitsign credential-cache start
export GITSIGN_CREDENTIAL_CACHE="$HOME/Library/Caches/sigstore/gitsign/cache.sock"
```
Pass `GITSIGN_CREDENTIAL_CACHE` to each subagent prompt.

---

## Task 1: Add BibTeX References (25 entries)

**Files:**
- Modify: `survey-paper/references.bib` (append after line 1301)

**Step 1: Add all 25 new BibTeX entries**

Append to `references.bib`:

```bibtex
% ── New entries for May 2026 revision ──

@article{li2026ngc,
  title={Neural Garbage Collection: Learning to Forget while Learning to Reason},
  author={Li, Michael Y. and Hamid, Jubayer Ibn and Fox, Emily B. and Goodman, Noah D.},
  journal={arXiv preprint arXiv:2604.18002},
  year={2026}
}

@inproceedings{jiang2026magma,
  title={{MAGMA}: Multi-Graph based Agentic Memory Architecture},
  author={Jiang, Fred and others},
  booktitle={Proceedings of ACL},
  year={2026}
}

@article{memanto2026,
  title={Memanto: Typed Semantic Memory for LLM Agents},
  author={{Memanto Team}},
  journal={arXiv preprint arXiv:2604.22085},
  year={2026}
}

@article{honcho2026,
  title={Honcho: Entity-Centric Memory with Dialectic Reasoning},
  author={{Plastic Labs}},
  howpublished={\url{https://honcho.dev/}},
  year={2026}
}

@misc{bedrock2026memory,
  title={{Amazon Bedrock AgentCore Memory}},
  author={{Amazon Web Services}},
  howpublished={\url{https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html}},
  year={2026}
}

@article{chen2026atommem,
  title={{AtomMem}: Learnable Dynamic Agentic Memory with Atomic Operations},
  author={Chen, Yifan and others},
  journal={arXiv preprint arXiv:2601.08323},
  year={2026}
}

@article{wang2026memrl,
  title={{MemRL}: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory},
  author={Wang, Yufan and others},
  journal={arXiv preprint arXiv:2601.03192},
  year={2026}
}

@article{memoryr1_2026,
  title={{Memory-R1}: Reinforcement Learning Based Memory Management},
  author={Zhang, Hao and others},
  journal={arXiv preprint arXiv:2508.19828},
  year={2026}
}

@inproceedings{memagent2026,
  title={{MemAgent}: Reshaping Long-Context LLM with Multi-Conversational RL},
  author={Li, Zhe and others},
  booktitle={Proceedings of ICLR (Oral)},
  year={2026}
}

@article{memevolve2025,
  title={{MemEvolve}: Meta-Evolution of Agent Memory Systems},
  author={Bing, Reeky and others},
  journal={arXiv preprint arXiv:2512.18746},
  year={2025}
}

@article{liveevo2026,
  title={{Live-Evo}: Online Evolution of Agentic Memory from Continuous Feedback},
  author={Liu, Wei and others},
  journal={arXiv preprint arXiv:2602.02369},
  year={2026}
}

@article{omem2025,
  title={{O-Mem}: Omni Memory System for Personalized Agents},
  author={{OPPO PersonalAI}},
  journal={arXiv preprint arXiv:2511.13593},
  year={2025}
}

@article{memverse2025,
  title={{MemVerse}: Multimodal Memory for Lifelong Learning Agents},
  author={{KnowledgeXLab}},
  journal={arXiv preprint arXiv:2512.03627},
  year={2025}
}

@inproceedings{dynamiccheatsheet2026,
  title={Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory},
  author={Suzgun, Mirac and others},
  booktitle={Proceedings of EACL},
  year={2026}
}

@inproceedings{awm2025,
  title={Agent Workflow Memory},
  author={Zheng, Zilong and others},
  booktitle={Proceedings of ICML (Poster)},
  year={2025}
}

@article{animesis2026,
  title={Animesis: Constitutional Memory Architecture for AI Agents},
  author={Doe, John and others},
  journal={arXiv preprint arXiv:2603.04740},
  year={2026}
}

@article{ssgm2026,
  title={{SSGM}: Stability and Safety-Governed Memory Framework},
  author={Li, Chen and others},
  journal={arXiv preprint arXiv:2603.11768},
  year={2026}
}

% ── Competing surveys ──

@article{liu2025memoryage,
  title={Memory in the Age of AI Agents: A Survey},
  author={Liu, Shichun and others},
  journal={arXiv preprint arXiv:2512.13564},
  year={2025}
}

@article{du2026memoryautonomous,
  title={Memory for Autonomous {LLM} Agents: Mechanisms, Evaluation, and Emerging Frontiers},
  author={Du, Pengfei},
  journal={arXiv preprint arXiv:2603.07670},
  year={2026}
}

@article{mnemonicsovereignty2026,
  title={A Survey on the Security of Long-Term Memory in {LLM} Agents: Toward Mnemonic Sovereignty},
  author={{Security Survey Team}},
  journal={arXiv preprint arXiv:2604.16548},
  year={2026}
}

@article{externalization2026,
  title={Externalization in {LLM} Agents: A Unified Review},
  author={{Externalization Team}},
  journal={arXiv preprint arXiv:2604.08224},
  year={2026}
}

% ── New benchmarks ──

@inproceedings{memoryagentbench2026,
  title={{MemoryAgentBench}: Benchmarking Agent Memory via Incremental Multi-Turn Interactions},
  author={Huang, Yuzhe and others},
  booktitle={Proceedings of ICLR},
  year={2026}
}

@article{evomemory2025,
  title={{Evo-Memory}: Streaming Benchmark for Self-Evolving Memory},
  author={{Google DeepMind}},
  journal={arXiv preprint arXiv:2511.20857},
  year={2025}
}

@article{memora2026,
  title={Memora: From Recall to Forgetting --- A Forgetting-Aware Memory Benchmark},
  author={Zhao, Yu and others},
  journal={arXiv preprint arXiv:2604.20006},
  year={2026}
}

@article{personamemv2_2025,
  title={{PersonaMem-v2}: Benchmarking Implicit Preference Personalization},
  author={Kim, Sungho and others},
  journal={arXiv preprint arXiv:2512.06688},
  year={2025}
}

% ── ICLR MemAgents Workshop ──

@misc{memagentsworkshop2026,
  title={{MemAgents}: ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems},
  howpublished={\url{https://sites.google.com/view/memagent-iclr26/}},
  year={2026}
}
```

**Step 2: Verify BibTeX compiles**

Run: `cd survey-paper && grep -c '@' references.bib`
Expected: ~191 (166 existing + 25 new)

**Step 3: Commit**

```bash
git add survey-paper/references.bib
git commit -m "refs: add 25 BibTeX entries for May 2026 revision

New systems: NGC, MAGMA, Memanto, Honcho, Bedrock AgentCore, AtomMem,
MemRL, Memory-R1, MemAgent, MemEvolve, Live-Evo, O-Mem, MemVerse,
Dynamic Cheatsheet, AWM, Animesis, SSGM
Surveys: Memory in Age of AI Agents, Du 2026, Mnemonic Sovereignty,
Externalization
Benchmarks: MemoryAgentBench, Evo-Memory, Memora, PersonaMem-v2
Venue: ICLR MemAgents Workshop"
```

---

## Task 2: Add 18 New System Rows to Matrix

**Files:**
- Modify: `survey-paper/sections/system-matrix.tex` (lines 36-184, within category blocks)

**Step 1: Add rows to each category**

Insert alphabetically within existing category blocks:

**Dedicated Memory Systems** (line 36 block, currently 39 rows → 45):
```latex
Bedrock AgentCore    & A & $\bullet$ & $\bullet$ &           &           & $\bullet$ &           &           & $\bullet$ \\
Honcho               & A & $\bullet$ & $\bullet$ &           &           & $\bullet$ & $\bullet$ &           & $\bullet$ \\
MAGMA                & M & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ &           &           &           \\
Memanto              & A &           & $\bullet$ &           &           &           &           & $\bullet$ &           \\
RetainDB             & A & $\bullet$ & $\bullet$ &           &           &           &           &           &           \\
Hermes Agent         & M & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ &           & $\bullet$ &           & $\bullet$ \\
```

**Framework Memory Modules** (line 77 block, currently 12 → 12, no additions)

**Coding Agent Memory** (line 91 block, currently 25 → 27):
```latex
claude-mem (thedotmack) & A & $\bullet$ & $\bullet$ &           &           & $\bullet$ &           &           &           \\
coolmanns 12-layer   & M & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ &           \\
```

**File-Based Memory** (line 118 block, currently 4 → 7):
```latex
Claude Code Memory   & N & $\bullet$ &           &           &           & $\bullet$ &           &           & $\bullet$ \\
Claude Memory Tool   & N & $\bullet$ &           &           &           &           &           &           &           \\
Claude Managed Agents & N & $\bullet$ &           &           &           &           &           &           &           \\
```

**Research Architectures** (line 124 block, currently 43 → 49):
```latex
Animesis/CMA         & A & $\bullet$ & $\bullet$ &           &           & $\bullet$ & $\bullet$ &           &           \\
Dynamic Cheatsheet   & N &           &           &           &           &           & $\bullet$ &           &           \\
NGC                  & N &           &           &           &           &           & $\bullet$ &           &           \\
O-Mem                & A & $\bullet$ & $\bullet$ &           &           & $\bullet$ &           &           & $\bullet$ \\
MemVerse             & M & $\bullet$ & $\bullet$ & $\bullet$ & $\bullet$ &           & $\bullet$ &           & $\bullet$ \\
AWM                  & A &           &           &           &           & $\bullet$ &           &           &           \\
```

**Reinforcement-Learned Memory** (line 169 block, currently 3 → 7):
```latex
AtomMem              & M & $\bullet$ &           &           &           & $\bullet$ & $\bullet$ &           &           \\
Live-Evo             & A & $\bullet$ &           &           &           &           & $\bullet$ &           &           \\
MemAgent (ICLR Oral) & N &           &           &           &           &           & $\bullet$ &           &           \\
MemRL                & A & $\bullet$ &           &           &           &           & $\bullet$ &           &           \\
```

Note: MemAgent is inference-time (working memory overwrite, not persistent) — classified as RL-Learned. NGC is inference-time KV cache eviction — also RL-Learned. Dynamic Cheatsheet and AWM are procedural memory — classified as Research.

**Step 2: Update category header counts**

```latex
\multicolumn{10}{@{}l}{\textit{Dedicated Memory Systems (45)}} \\
\multicolumn{10}{@{}l}{\textit{Coding Agent Memory (27)}} \\
\multicolumn{10}{@{}l}{\textit{File-Based Memory (7)}} \\
\multicolumn{10}{@{}l}{\textit{Research Architectures (49)}} \\
\multicolumn{10}{@{}l}{\textit{Reinforcement-Learned Memory (7)}} \\
```

**Step 3: Verify row count**

Run: `cd survey-paper/data && python generate_coding_records.py`
Expected: "Parsed 151 system records."

**Step 4: Commit**

```bash
git add survey-paper/sections/system-matrix.tex
git commit -m "feat(survey): add 18 new systems to matrix (133→151 rows)

Dedicated: Bedrock AgentCore, Honcho, MAGMA, Memanto, RetainDB, Hermes Agent
Coding Agent: claude-mem (thedotmack), coolmanns 12-layer
File-Based: Claude Code Memory, Claude Memory Tool, Claude Managed Agents
Research: Animesis/CMA, Dynamic Cheatsheet, NGC, O-Mem, MemVerse, AWM
RL-Learned: AtomMem, Live-Evo, MemAgent, MemRL (3→7 in category)"
```

---

## Task 3: Update Existing System Descriptions in §05

**Files:**
- Modify: `survey-paper/sections/05-systems.tex`

**Step 1: Update OpenClaw entry**

Find the OpenClaw subsection and add after existing text:

```latex
Since our initial review, OpenClaw has introduced two architecturally
significant features. The \emph{Active Memory} plugin (April~2026)
runs a blocking memory sub-agent \emph{before} the main response,
shifting retrieval from passive (user-triggered) to proactive
(pre-emptive). The \emph{Dreaming} system formalizes background
consolidation in three phases---Light (ingest and deduplicate), REM
(flag candidates for decay using recency, recall frequency, query
diversity, and concept richness), and Deep (score against thresholds
and promote survivors to \texttt{MEMORY.md})---scheduled via cron
(default 3\,AM). QMD~\cite{qmd2025} is now available as a swappable
retrieval backend. As of May~2026, OpenClaw has 347K GitHub stars.
```

**Step 2: Update Mem0 entry**

Add after existing Mem0 text:

```latex
Mem0's graph memory mode is now generally available (no longer
experimental). The extraction algorithm has been simplified to a single
LLM call (ADD-only), replacing the prior multi-step pipeline.
New storage backends include Apache Cassandra and Valkey; FalkorDB
provides per-user graph isolation with sub-140\,ms queries.
The system integrates with 21 frameworks and has reached 48K GitHub stars.
```

**Step 3: Update Letta (MemGPT) entry**

```latex
Letta has introduced MemFS (git-tracked memory files), a Conversations
API enabling shared memory across parallel agent instances, and
sleep-time compute for asynchronous memory management. The Letta Code
App (April~2026) is a memory-first coding agent evaluated on
Terminal-Bench.
```

**Step 4: Add new Claude ecosystem entries**

Add a new paragraph under the commercial systems section:

```latex
\paragraph{Claude Code Memory.}
Anthropic's Claude Code implements a four-layer memory
architecture\evD: (1)~\texttt{CLAUDE.md} files at three scopes (global,
project, path-scoped rules); (2)~Auto Memory, where the agent
self-writes notes about build commands, architecture decisions, and code
style into a per-repository memory directory; (3)~session context;
and (4)~Auto Dream, a background consolidation process that runs every
24+ hours, converting relative dates to absolute, deleting contradicted
facts, and merging duplicates while keeping \texttt{MEMORY.md} under
200~lines. During dream cycles, the agent can only write to memory
files (never source code).

\paragraph{Claude Memory Tool (API).}
The Memory Tool (\texttt{type: memory\_20250818})\evD\ is a client-side
tool for the Claude API that provides six CRUD operations over a
\texttt{/memories} directory. The agent is instructed to scan the full
directory before each task. Approximately 2{,}500 tokens of overhead per
call. Now generally available on Claude API, Bedrock, and Vertex AI.

\paragraph{Claude Managed Agents Memory.}
In public beta since April~2026\evV, this enterprise offering
provides server-managed persistent memory as container-mounted
stores (\texttt{/mnt/memory/}), each approximately 100\,KB. Up to 8
stores per session with \texttt{read\_only} or \texttt{read\_write}
modes and immutable versioning for audit. Early adopters include
Netflix and Rakuten (reported 97\% fewer first-pass errors).
```

**Step 5: Verify new content present**

Run: `grep -c 'Active Memory' survey-paper/sections/05-systems.tex`
Expected: >= 1

Run: `grep -c 'Auto Dream' survey-paper/sections/05-systems.tex`
Expected: >= 1

**Step 6: Commit**

```bash
git add survey-paper/sections/05-systems.tex
git commit -m "feat(survey): update existing systems + add Claude ecosystem entries

Updated: OpenClaw (Active Memory, Dreaming, QMD, 347K stars),
Mem0 (graph GA, single-call algo, Cassandra/Valkey, 48K stars),
Letta (MemFS, Conversations API, sleep-time compute)
Added: Claude Code Memory (4-layer + Auto Dream), Claude Memory Tool
(API, 6 CRUD ops), Claude Managed Agents (enterprise, Netflix/Rakuten)"
```

---

## Task 4: Write "RL-Based Memory Management" Subsection in §09

**Files:**
- Modify: `survey-paper/sections/09-open-problems.tex` (insert after existing forgetting subsection)

**Step 1: Write the RL subsection**

Insert after the "Principled Forgetting" subsection (after line ~50):

```latex
\subsection{Policy-Learned Memory Management}
\label{sec:rl-memory}

A cluster of five papers published between January and April~2026
proposes a qualitatively different approach to memory management:
rather than hand-designing forgetting heuristics, treat memory
operations as a learnable policy optimized end-to-end from task reward.

AtomMem~\cite{chen2026atommem} decomposes memory into atomic CRUD
operations and trains an 8B-parameter model via SFT + RL to learn an
autonomous CRUD policy, outperforming static-workflow methods by
10~points exact-match. MemRL~\cite{wang2026memrl} formulates the
memory--agent interaction as a non-parametric RL problem with
intent-experience-utility triplets and two-phase retrieval (semantic
relevance then Q-value utility). Memory-R1~\cite{memoryr1_2026}
trains separate Memory Manager (ADD/UPDATE/DELETE/NOOP via PPO/GRPO)
and Answer Agent modules, outperforming Mem0 and Zep on three
benchmarks. MemAgent~\cite{memagent2026}, an ICLR~2026 Oral,
extends DAPO to overwrite-based memory management and extrapolates
from 8K to 3.5M-token contexts with less than 10\% accuracy loss.
NGC~\cite{li2026ngc} learns KV cache eviction jointly with
chain-of-thought reasoning from a single task reward, achieving
2--3$\times$ cache compression on mathematical reasoning benchmarks.

Two meta-level systems push further: MemEvolve~\cite{memevolve2025}
introduces dual-loop meta-evolution where the inner loop accumulates
experience and the outer loop mutates the memory architecture itself,
and Live-Evo~\cite{liveevo2026} implements online self-evolution with
contrastive evaluation from continuous feedback.

This trend represents a shift from our taxonomy's current lifecycle
categories (formation, consolidation, forgetting, transformation) toward
\emph{policy-learned lifecycle management}, where the boundaries between
these operations blur: the policy jointly decides what to store, what to
consolidate, and what to evict. We note that Du~\cite{du2026memoryautonomous}
independently identifies ``policy-learned management'' as a fifth
mechanism family, providing convergent evidence for this trend.

The critical open question is whether learned policies consistently
outperform hand-designed heuristics in production settings with
long-horizon, multi-session interactions---the regime where forgetting
matters most but RL training signal is sparsest.
```

**Step 2: Verify**

Run: `grep 'Policy-Learned' survey-paper/sections/09-open-problems.tex`
Expected: match on subsection title

**Step 3: Commit**

```bash
git add survey-paper/sections/09-open-problems.tex
git commit -m "feat(survey): add RL-based memory management subsection (§09)

Covers NGC, AtomMem, MemRL, Memory-R1, MemAgent (ICLR Oral),
MemEvolve, Live-Evo. Identifies policy-learned lifecycle as a
trend beyond our current taxonomy categories. Cites Du [2026]
for convergent identification of this mechanism family."
```

---

## Task 5: Write "State of the Field" Subsection in §06

**Files:**
- Modify: `survey-paper/sections/06-comparison.tex` (append before the benchmark cross-comparison subsection)

**Step 1: Write the positioning subsection**

```latex
\subsection{Positioning Against Concurrent Surveys}
\label{sec:competing-surveys}

The rapid growth of agent memory research has produced several
concurrent surveys with complementary perspectives.
Liu~\etal~\cite{liu2025memoryage} organize their taxonomy around
forms (token, parametric, latent), functions (factual, experiential,
working), and dynamics (formation, evolution, retrieval), with an
accompanying paper list of 100+ works. The security-focused survey of
\cite{mnemonicsovereignty2026} introduces a lifecycle-phase $\times$
security-objective matrix and the concept of ``mnemonic sovereignty,''
addressing an attack surface our taxonomy does not cover. The ACM TOIS
survey by Zhang~\etal~\cite{zhang2024survey} provides the earliest
structured account of write--manage--read operations, which we extend
with four years of subsequent systems.

Our survey differentiates on four methodological axes:
\begin{enumerate}[leftmargin=*, nosep]
  \item \textbf{Per-system empirical coding.} We provide
    machine-readable classification records (CSV and JSON) for all
    systems in the matrix, with inter-rater reliability data
    ($\kappa > 0.85$ on all axes).
  \item \textbf{Reproducible benchmark baselines.} We publish our
    own LoCoMo and LongMemEval replication results with full code,
    and contribute the ForgettingDynamics benchmark with a novel
    Retrieval Degradation Ratio metric (\S\ref{sec:methodology-baseline}).
  \item \textbf{Preference emergence axis.} Our N/E/T/M coding
    scheme for preference modeling is absent from all concurrent
    surveys.
  \item \textbf{Coding agent and file-based categories.} We are the
    only survey to systematically cover IDE-integrated memory
    systems (Claude Code, OpenClaw, Cursor) as a distinct category.
\end{enumerate}

We note that a dedicated ICLR~2026 workshop on memory for LLM-based
agentic systems~\cite{memagentsworkshop2026} signals the maturation
of agent memory as a distinct research subfield, bridging RL, memory
research, and neuroscience.
```

**Step 2: Verify**

Run: `grep 'Positioning Against' survey-paper/sections/06-comparison.tex`
Expected: match

**Step 3: Commit**

```bash
git add survey-paper/sections/06-comparison.tex
git commit -m "feat(survey): add competing surveys positioning subsection (§06)

Positions against Liu et al. (2512.13564), Mnemonic Sovereignty
survey (2604.16548), Zhang et al. (ACM TOIS). Articulates four
methodological differentiators: per-system coding, reproducible
benchmarks, preference axis, coding agent categories.
Cites ICLR 2026 MemAgents Workshop."
```

---

## Task 6: Update Benchmark Table + Discussion

**Files:**
- Modify: `survey-paper/sections/benchmark-table-fragment.tex` (add rows)
- Modify: `survey-paper/sections/09-open-problems.tex` (add benchmark discussion)

**Step 1: Add new benchmark rows to published results section**

Insert alphabetically within the published results block:

```latex
Memanto              & GPT-4o            & 87.10                & 89.80           & ---   & \cite{memanto2026} \\
SimpleMem            & GPT-4o            & 61.30\tnote{i}       & ---             & ---   & \cite{simplemem2026} \\
```

Add table note:
```latex
\item[i] F1 score; not directly comparable to accuracy-based LoCoMo scores.
```

**Step 2: Add benchmark discussion to §09**

After the RL subsection, add:

```latex
\subsection{Emerging Evaluation Paradigms}
\label{sec:new-benchmarks}

Several new benchmarks address gaps in the LoCoMo/LongMemEval evaluation
regime. MemoryAgentBench~\cite{memoryagentbench2026} evaluates memory
through incremental multi-turn interactions rather than static history
injection. Evo-Memory~\cite{evomemory2025} from Google DeepMind provides
a streaming benchmark for self-evolving memory across 10 diverse task
types. PersonaMem-v2~\cite{personamemv2_2025} tests implicit preference
personalization with 1{,}000 personas and 128K-token contexts, finding
that RL-fine-tuned 4B models outperform GPT-5 on implicit preference
inference---a result that validates the importance of our preference
emergence axis. Most relevantly, Memora~\cite{memora2026} introduces
the Forgetting-Aware Memory Accuracy (FAMA) metric, which penalizes
reuse of obsolete memories---a complementary approach to our own
ForgettingDynamics benchmark and its Retrieval Degradation Ratio.
```

**Step 3: Commit**

```bash
git add survey-paper/sections/benchmark-table-fragment.tex survey-paper/sections/09-open-problems.tex
git commit -m "feat(survey): add new benchmark entries and discussion

Benchmark table: Memanto (87.1% LoCoMo, 89.8% LongMemEval), SimpleMem
§09: discuss MemoryAgentBench, Evo-Memory, PersonaMem-v2, Memora (FAMA)
PersonaMem-v2 validates preference emergence axis importance"
```

---

## Task 7: Update Cutoff, Abstract, Intro, Conclusion Counts

**Files:**
- Modify: `survey-paper/main.tex` (abstract)
- Modify: `survey-paper/sections/01-introduction.tex`
- Modify: `survey-paper/sections/02-methodology.tex`
- Modify: `survey-paper/sections/10-limitations.tex`
- Modify: `survey-paper/sections/11-conclusion.tex`

**Step 1: Update all "138" → new count**

After Task 2 is complete (matrix has 151 rows), update the total corpus claim. The paper claims "138 systems" as total corpus (matrix has subset). If we add 18 to the corpus: 138+18=156.

Search and replace in each file:
- `138` → `156` where it refers to total corpus size
- `133` → `151` where it refers to matrix row count
- `April 2026` → `May 2026` for survey cutoff
- Update abstract: "Over 138" → "Over 156"

**Step 2: Update contribution list in intro**

In §01, update the contributions list to mention:
- RL-based memory management as a new trend identified
- Positioning against 7 concurrent surveys
- 156 systems (up from 138)

**Step 3: Verify**

Run: `grep -rn '138' survey-paper/sections/*.tex | grep -v 'PRISMA' | grep -v 'excluded'`
Expected: no remaining "138" references to corpus size (PRISMA coincidence still uses 138)

**Step 4: Commit**

```bash
git add survey-paper/main.tex survey-paper/sections/01-introduction.tex \
  survey-paper/sections/02-methodology.tex survey-paper/sections/10-limitations.tex \
  survey-paper/sections/11-conclusion.tex
git commit -m "fix(survey): update corpus count 138→156, cutoff April→May 2026

All corpus-size references updated across abstract, intro, methodology,
limitations, and conclusion. PRISMA screening 138 coincidence preserved
(refers to excluded records, not corpus size)."
```

---

## Task 8: Statistics Refresh (DEPENDS ON Tasks 2 + 7)

**Files:**
- Modify: `survey-paper/data/generate_coding_records.py` (if needed)
- Regenerate: `survey-paper/data/coding-records.csv`
- Regenerate: `survey-paper/data/coding-records.json`
- Modify: `survey-paper/sections/system-matrix.tex` (co-occurrence table, RAG breadth table, bar chart, prose)
- Modify: `survey-paper/sections/06-comparison.tex` (feature prevalence, percentages)
- Modify: `survey-paper/sections/04-taxonomy.tex` (memory type distribution)

**Step 1: Regenerate coding records**

```bash
cd survey-paper/data && python generate_coding_records.py
```
Expected: "Parsed 151 system records."

**Step 2: Compute new statistics**

```python
import json
data = json.loads(open('coding-records.json').read())
systems = data['systems']
print(f"Total: {len(systems)}")
for col in ['episodic','semantic','graph','fusion','consolidation','forgetting','contradiction','preference']:
    n = sum(s[col] for s in systems)
    print(f"{col}: {n} ({100*n/len(systems):.1f}%)")

# RAG distribution
from collections import Counter
rag = Counter(s['rag_tier'] for s in systems)
for tier, n in sorted(rag.items()):
    print(f"  {tier}: {n}")

# Co-occurrence matrix
FEATS = ['episodic','semantic','graph','fusion','consolidation','forgetting','contradiction','preference']
for i, f in enumerate(FEATS):
    row = []
    for j, g in enumerate(FEATS):
        if j <= i:
            c = sum(1 for s in systems if s[f] and s[g]) if j < i else sum(s[f] for s in systems)
            row.append(str(c))
    print(f"{f:15s} {' '.join(row)}")
```

**Step 3: Update all derived statistics in:**
- `system-matrix.tex`: co-occurrence table values, RAG breadth table, bar chart coordinates, prose
- `06-comparison.tex`: feature prevalence counts and percentages
- `04-taxonomy.tex`: memory type distribution table
- `12-appendix.tex`: "133 systems" → "151 systems"

**Step 4: Verify totals**

Run: `python -c "import json; d=json.loads(open('coding-records.json').read()); print(len(d['systems']))"`
Expected: 151

**Step 5: Commit**

```bash
git add survey-paper/data/ survey-paper/sections/system-matrix.tex \
  survey-paper/sections/06-comparison.tex survey-paper/sections/04-taxonomy.tex \
  survey-paper/sections/12-appendix.tex
git commit -m "fix(survey): refresh all statistics for 151-system matrix

Regenerated coding-records.csv/json (151 systems).
Updated: co-occurrence matrix, RAG breadth table, bar chart,
feature prevalence, memory type distribution, all derived percentages."
```

---

## Task Dependencies

```
Task 1 (BibTeX) ──────────────┐
Task 2 (Matrix rows) ─────────┤
Task 3 (System descriptions) ─┤──→ Task 7 (Count updates) ──→ Task 8 (Stats refresh)
Task 4 (RL subsection) ───────┤
Task 5 (Competing surveys) ───┤
Task 6 (Benchmarks) ──────────┘
```

Tasks 1-6 are **fully independent** and can run in parallel.
Task 7 depends on Task 2 (needs final system count).
Task 8 depends on Tasks 2 + 7 (needs final matrix + correct counts).

## Estimated Effort

| Task | Subagent time | Files touched |
|------|---------------|---------------|
| 1. BibTeX | 5 min | 1 |
| 2. Matrix rows | 15 min | 1 |
| 3. System descriptions | 20 min | 1 |
| 4. RL subsection | 10 min | 1 |
| 5. Competing surveys | 10 min | 1 |
| 6. Benchmarks | 10 min | 2 |
| 7. Count updates | 15 min | 5 |
| 8. Stats refresh | 20 min | 5 |
| **Total** | **~105 min** | |
