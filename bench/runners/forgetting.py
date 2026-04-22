"""ForgettingDynamics benchmark runner.

Measures whether a memory system's recall quality degrades (or improves) as
the memory store grows over time.  Unlike existing benchmarks that ingest a
static history and query once, this runner uses an *interleaved* protocol:

  1. Ingest seed sessions  → query seed questions (pre-growth baseline)
  2. Ingest growth sessions (no reset) → query seed questions again (post-growth)
  3.                                    → query growth questions
  4.                                    → query contradiction questions

Key metric: Retrieval Degradation Ratio (RDR)
    RDR = post_growth_seed_accuracy / pre_growth_seed_accuracy

  RDR < 1.0  →  memory growth degrades seed recall (noise accumulation winning)
  RDR = 1.0  →  memory is stable under growth
  RDR > 1.0  →  forgetting/pruning actively improves retrieval quality

Dataset format (JSON list of scenario dicts):
  {
    "scenario_id": str,
    "domain": str,
    "seed_sessions": [{"role", "content", "session_id", "timestamp"}, ...],
    "growth_sessions": [...],
    "seed_questions":       [{"id", "question", "gold", "type": "seed_retrieval"}, ...],
    "growth_questions":     [{"id", "question", "gold", "type": "growth_retrieval"}, ...],
    "contradiction_questions": [{"id", "question", "gold", "gold_before_update", "type": "contradiction"}, ...],
  }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from adapters.base import MemoryAdapter, Message


@dataclass
class ForgettingResult:
    """Results from one ForgettingDynamics run on a single adapter."""

    system: str
    pre_growth_accuracy: float = 0.0
    post_growth_accuracy: float = 0.0
    growth_accuracy: float = 0.0
    contradiction_accuracy: float = 0.0
    total_seed_questions: int = 0
    total_growth_questions: int = 0
    total_contradiction_questions: int = 0
    elapsed_seconds: float = 0.0
    per_question: list[dict] = field(default_factory=list)
    # rdr is a dataclass field (not a property) so asdict() serializes it.
    # init=False: excluded from __init__; computed by __post_init__.
    rdr: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Compute rdr from pre/post accuracy at construction time."""
        if self.pre_growth_accuracy > 0:
            self.rdr = self.post_growth_accuracy / self.pre_growth_accuracy
        else:
            self.rdr = 0.0

    @property
    def accuracy(self) -> float:
        """Overall accuracy across all question types (for results table compat)."""
        total = (
            self.total_seed_questions
            + self.total_growth_questions
            + self.total_contradiction_questions
        )
        if total == 0:
            return 0.0
        correct = sum(1 for q in self.per_question if q.get("score", 0) >= 0.5)
        return correct / total

    @property
    def total_questions(self) -> int:
        return (
            self.total_seed_questions
            + self.total_growth_questions
            + self.total_contradiction_questions
        )

    @property
    def correct(self) -> int:
        return sum(1 for q in self.per_question if q.get("score", 0) >= 0.5)


def _msgs_from_dicts(raw: list[dict]) -> list[Message]:
    return [
        Message(
            role=m["role"],
            content=m["content"],
            session_id=m["session_id"],
            timestamp=m["timestamp"],
        )
        for m in raw
    ]


def _query_and_score(
    adapter: MemoryAdapter,
    questions: list[dict],
    phase: str,
    judge_fn: Callable,
    llm_call: Callable,
    result: ForgettingResult,
) -> float:
    """Query the adapter on a list of questions; return accuracy for that phase."""
    correct = 0
    for q in questions:
        prediction = adapter.query(q["question"], llm_call)
        score = judge_fn(q["question"], q["gold"], prediction)
        correct += int(score >= 0.5)
        result.per_question.append({
            "phase": phase,
            "scenario_id": q.get("scenario_id", ""),
            "question_id": q.get("id", ""),
            "question": q["question"],
            "gold": q["gold"],
            "prediction": prediction,
            "score": score,
        })
    return correct / len(questions) if questions else 0.0


def load_forgetting_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_forgetting(
    adapter: MemoryAdapter,
    dataset_path: Path,
    judge_fn: Callable,
    llm_call: Callable,
    limit: int | None = None,
    dry_run: bool = False,
) -> ForgettingResult:
    """Run ForgettingDynamics on a single adapter.

    Args:
        adapter:      Memory system under test.
        dataset_path: Path to forgetting_fixture.json or full dataset.
        judge_fn:     Callable(question, gold, prediction) → float.
        llm_call:     Callable(prompt) → str.
        limit:        Max number of scenarios to process (None = all).
        dry_run:      If True, print token estimates without running.
    """
    scenarios = load_forgetting_dataset(dataset_path)
    if limit is not None:
        scenarios = scenarios[:limit]

    result = ForgettingResult(system=adapter.name)

    if dry_run:
        n_seed_q = sum(len(s["seed_questions"]) for s in scenarios)
        n_growth_q = sum(len(s["growth_questions"]) for s in scenarios)
        n_contra_q = sum(len(s["contradiction_questions"]) for s in scenarios)
        # Seed questions asked twice; estimate ~8K tokens per query round
        total_q = 2 * n_seed_q + n_growth_q + n_contra_q
        est_tokens = total_q * 8_000
        print(
            f"[DRY RUN] {adapter.name}: {len(scenarios)} scenarios, "
            f"{total_q} query calls, ~{est_tokens:,} input tokens"
        )
        result.total_seed_questions = n_seed_q
        result.total_growth_questions = n_growth_q
        result.total_contradiction_questions = n_contra_q
        return result

    start = time.time()

    pre_growth_scores: list[float] = []
    post_growth_scores: list[float] = []
    growth_scores: list[float] = []
    contradiction_scores: list[float] = []

    for scenario in tqdm(scenarios, desc=f"ForgettingDynamics [{adapter.name}]"):
        sid = scenario["scenario_id"]

        # Annotate questions with scenario_id for the per_question log
        seed_qs = [{**q, "scenario_id": sid} for q in scenario["seed_questions"]]
        growth_qs = [{**q, "scenario_id": sid} for q in scenario["growth_questions"]]
        contra_qs = [{**q, "scenario_id": sid} for q in scenario["contradiction_questions"]]

        # ── Phase 0: reset and ingest seed sessions ──
        adapter.reset()
        seed_messages = _msgs_from_dicts(scenario["seed_sessions"])
        adapter.ingest(seed_messages)

        # ── Phase 1: pre-growth seed queries ──
        pre_acc = _query_and_score(
            adapter, seed_qs, "pre_growth_seed", judge_fn, llm_call, result
        )
        pre_growth_scores.append(pre_acc)

        # ── Phase 2: ingest growth sessions (NO reset — memory grows) ──
        growth_messages = _msgs_from_dicts(scenario["growth_sessions"])
        adapter.ingest(growth_messages)

        # ── Phase 3: post-growth seed queries (same questions) ──
        post_acc = _query_and_score(
            adapter, seed_qs, "post_growth_seed", judge_fn, llm_call, result
        )
        post_growth_scores.append(post_acc)

        # ── Phase 4: growth-phase queries ──
        g_acc = _query_and_score(
            adapter, growth_qs, "growth", judge_fn, llm_call, result
        )
        growth_scores.append(g_acc)

        # ── Phase 5: contradiction queries ──
        c_acc = _query_and_score(
            adapter, contra_qs, "contradiction", judge_fn, llm_call, result
        )
        contradiction_scores.append(c_acc)

    # Aggregate accuracy values
    result.pre_growth_accuracy = (
        sum(pre_growth_scores) / len(pre_growth_scores) if pre_growth_scores else 0.0
    )
    result.post_growth_accuracy = (
        sum(post_growth_scores) / len(post_growth_scores) if post_growth_scores else 0.0
    )
    result.growth_accuracy = (
        sum(growth_scores) / len(growth_scores) if growth_scores else 0.0
    )
    result.contradiction_accuracy = (
        sum(contradiction_scores) / len(contradiction_scores) if contradiction_scores else 0.0
    )
    # Recompute rdr now that pre/post accuracy are finalised
    result.rdr = (
        result.post_growth_accuracy / result.pre_growth_accuracy
        if result.pre_growth_accuracy > 0 else 0.0
    )

    # Count questions (seed questions counted once, despite being asked twice)
    result.total_seed_questions = sum(
        len(s["seed_questions"]) for s in scenarios
    )
    result.total_growth_questions = sum(
        len(s["growth_questions"]) for s in scenarios
    )
    result.total_contradiction_questions = sum(
        len(s["contradiction_questions"]) for s in scenarios
    )

    result.elapsed_seconds = time.time() - start
    return result
