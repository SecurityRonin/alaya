"""LongMemEval benchmark runner.

Loads the LongMemEval dataset, feeds conversation histories through
adapters, and scores answers using LLM-as-Judge.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from adapters.base import MemoryAdapter, Message


@dataclass
class LongMemEvalResult:
    system: str
    total_questions: int = 0
    correct: int = 0
    scores_by_type: dict[str, list[float]] = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_questions if self.total_questions else 0.0


def load_longmemeval(path: Path) -> list[dict]:
    """Load LongMemEval dataset from JSON."""
    with open(path) as f:
        return json.load(f)


def haystack_to_messages(item: dict) -> list[Message]:
    """Convert LongMemEval haystack sessions to Message objects."""
    messages = []
    dates = item.get("haystack_dates", [])
    sessions = item.get("haystack_sessions", [])
    for i, session in enumerate(sessions):
        session_ids = item.get("haystack_session_ids", [])
        session_id = session_ids[i] if i < len(session_ids) else f"s_{i}"
        date = dates[i] if i < len(dates) else f"2024-01-{i + 1:02d}"
        for turn in session:
            messages.append(Message(
                role=turn["role"],
                content=turn["content"],
                session_id=session_id,
                timestamp=f"{date}T00:00:00",
            ))
    return messages


def run_longmemeval(
    adapter: MemoryAdapter,
    dataset_path: Path,
    judge_fn: callable,
    llm_call: callable,
    limit: int | None = None,
    dry_run: bool = False,
) -> LongMemEvalResult:
    """Run LongMemEval benchmark on a single adapter.

    Args:
        adapter: The memory system adapter.
        dataset_path: Path to longmemeval_s.json or longmemeval_oracle.json.
        judge_fn: Callable(question, gold, prediction) -> float.
        llm_call: Callable(prompt) -> str for answer generation.
        limit: Max questions to evaluate (None = all).
        dry_run: If True, count questions and estimate cost without running.
    """
    data = load_longmemeval(dataset_path)
    result = LongMemEvalResult(system=adapter.name)

    if limit:
        data = data[:limit]

    if dry_run:
        result.total_questions = len(data)
        avg_tokens = 115000  # LongMemEval-S average
        print(f"[DRY RUN] {adapter.name}: {len(data)} questions, "
              f"~{len(data) * avg_tokens:,} input tokens")
        return result

    start = time.time()
    for item in tqdm(data, desc=f"LongMemEval [{adapter.name}]"):
        adapter.reset()
        messages = haystack_to_messages(item)
        adapter.ingest(messages)

        question = item["question"]
        gold = item["answer"]
        q_type = item.get("question_type", "unknown")

        prediction = adapter.query(question, llm_call)
        score = judge_fn(question, str(gold), prediction)

        result.total_questions += 1
        result.correct += int(score >= 0.5)
        result.scores_by_type.setdefault(q_type, []).append(score)
        result.per_question.append({
            "question_id": item.get("question_id", ""),
            "question": question,
            "gold": str(gold),
            "prediction": prediction,
            "score": score,
            "question_type": q_type,
        })

    result.elapsed_seconds = time.time() - start
    return result
