# Benchmark Harness Design

**Date:** 2026-03-01
**Goal:** Scaffold a Python evaluation harness that runs LoCoMo and LongMemEval benchmarks against Alaya and competing memory systems, producing scores for the taxonomy paper's benchmark table.

## Architecture

```
bench/
├── pyproject.toml              # uv/pip project config
├── README.md                   # Setup + run instructions
├── datasets/
│   ├── download.py             # Fetch LoCoMo + LongMemEval from HuggingFace
│   └── .gitkeep
├── adapters/
│   ├── base.py                 # MemoryAdapter ABC
│   ├── alaya.py                # Alaya via subprocess (cargo run)
│   ├── mem0_adapter.py         # Mem0 via Python SDK
│   ├── zep_adapter.py          # Zep via Python SDK
│   ├── fullcontext.py          # Baseline: entire history in prompt
│   └── naive_rag.py            # Baseline: chromadb vector similarity
├── runners/
│   ├── locomo.py               # LoCoMo evaluation runner
│   └── longmemeval.py          # LongMemEval evaluation runner
├── judge/
│   └── llm_judge.py            # LLM-as-Judge scorer (configurable model)
├── results/
│   └── .gitkeep
└── run.py                      # CLI entrypoint: select benchmark + systems
```

## Target Systems

| System | Adapter Strategy | Why Included |
|--------|-----------------|-------------|
| Alaya | Subprocess (`alaya-bench` CLI, stdin/stdout JSON) | Our system |
| Mem0 | Python SDK (`mem0ai`) | Most popular production system |
| Zep/Graphiti | Python SDK (`zep-python`) | Main rival, temporal KG |
| Full-context | Direct prompt injection (no memory) | Upper-bound baseline |
| Naive RAG | ChromaDB vector similarity | Lower-bound baseline |

## Data Flow

```
Dataset (HuggingFace)
  → Runner loads conversations + questions + gold answers
  → For each (conversation, question):
      → Adapter.reset()
      → Adapter.ingest(messages)       # Feed history to memory system
      → answer = Adapter.query(question, llm)  # Retrieve + generate
      → score = Judge.score(answer, gold)       # LLM-as-Judge
  → Aggregate scores per system
  → Write results/{benchmark}_{system}_{timestamp}.json
  → Print summary table (paper format)
```

## MemoryAdapter Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Message:
    role: str       # "user" or "assistant"
    content: str
    session_id: str
    timestamp: str  # ISO-8601

class MemoryAdapter(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state."""
        ...

    @abstractmethod
    def ingest(self, messages: list[Message]) -> None:
        """Feed conversation messages into the memory system."""
        ...

    @abstractmethod
    def query(self, question: str, llm: "LLMClient") -> str:
        """Retrieve relevant context and generate an answer."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """System name for results table."""
        ...
```

## Key Decisions

- **Alaya adapter**: Shells out to a small Rust CLI tool (`alaya-bench`) that exposes `ingest` and `query` subcommands over stdin/stdout JSON. Avoids PyO3/FFI complexity.
- **LLM client**: Thin wrapper around `litellm` so any model works (OpenAI, Anthropic, Gemini, local). Configurable via `LLM_MODEL` and `LLM_API_KEY` env vars.
- **Judge model**: Defaults to `gpt-4o-mini` (matching LoCoMo's default judge). Configurable via `JUDGE_MODEL` env var to test judge sensitivity.
- **Cost control**: `--dry-run` flag estimates token count and cost before running. `--limit N` runs only first N questions.
- **Results format**: JSON per run with full per-question scores, plus a summary table that matches the paper's `benchmark-table-fragment.tex` format.

## Benchmarks

### LoCoMo (Maharana et al., ACL 2024)
- Dataset: HuggingFace `locomo-benchmark/locomo`
- ~300 conversations, 16-26K tokens each
- LLM-as-Judge scoring against gold references
- Score: overall accuracy (%)

### LongMemEval (Wu et al., ICLR 2025)
- Dataset: HuggingFace `LongMemEval/LongMemEval`
- 500 questions over ~115K-token histories
- Accuracy on factual recall
- Score: accuracy (%)

## Cost Estimate

| Benchmark | Questions | Est. input tokens | Est. cost (gpt-4o-mini) |
|-----------|-----------|-------------------|------------------------|
| LoCoMo | ~300 × 5 systems | ~50M | ~$75 |
| LongMemEval_S | 500 × 5 systems | ~100M | ~$150 |
| Judge scoring | ~4,000 calls | ~8M | ~$12 |
| **Total** | | | **~$237** |

## Out of Scope

- Docker orchestration
- CI/CD integration
- Web dashboard
- Criterion/Rust microbenchmarks (separate concern)
- DMR benchmark (saturated, not worth running)
