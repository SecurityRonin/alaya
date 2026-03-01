"""LoCoMo benchmark runner.

Loads the LoCoMo dataset, feeds conversations through adapters,
and scores answers using LLM-as-Judge.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from tqdm import tqdm

from adapters.base import MemoryAdapter, Message


@dataclass
class LoCoMoResult:
    system: str
    total_questions: int = 0
    correct: int = 0
    scores_by_category: dict[int, list[float]] = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_questions if self.total_questions else 0.0


def load_locomo(path: Path) -> list[dict]:
    """Load LoCoMo dataset from JSON."""
    with open(path) as f:
        return json.load(f)


def conversation_to_messages(conv: dict) -> list[Message]:
    """Convert a LoCoMo conversation to a list of Message objects."""
    messages = []
    session_idx = 1
    while f"session_{session_idx}" in conv["conversation"]:
        session_key = f"session_{session_idx}"
        timestamp = conv["conversation"].get(f"{session_key}_date_time", f"2024-01-{session_idx:02d}T00:00:00")
        for turn in conv["conversation"][session_key]:
            messages.append(Message(
                role="user" if turn["speaker"] == conv["conversation"]["speaker_a"] else "assistant",
                content=turn["text"],
                session_id=f"session_{session_idx}",
                timestamp=timestamp,
            ))
        session_idx += 1
    return messages


def run_locomo(
    adapter: MemoryAdapter,
    dataset_path: Path,
    judge_fn: callable,
    llm_call: callable,
    limit: int | None = None,
    dry_run: bool = False,
) -> LoCoMoResult:
    """Run LoCoMo benchmark on a single adapter.

    Args:
        adapter: The memory system adapter.
        dataset_path: Path to locomo10.json.
        judge_fn: Callable(question, gold, prediction) -> float.
        llm_call: Callable(prompt) -> str for answer generation.
        limit: Max questions to evaluate (None = all).
        dry_run: If True, count questions and estimate cost without running.
    """
    data = load_locomo(dataset_path)
    result = LoCoMoResult(system=adapter.name)

    # Collect all (conversation, question) pairs
    pairs = []
    for sample in data:
        messages = conversation_to_messages(sample)
        for qa in sample["qa"]:
            pairs.append((messages, qa))

    if limit:
        pairs = pairs[:limit]

    if dry_run:
        result.total_questions = len(pairs)
        avg_tokens = 20000  # approximate LoCoMo conversation length
        print(f"[DRY RUN] {adapter.name}: {len(pairs)} questions, "
              f"~{len(pairs) * avg_tokens:,} input tokens")
        return result

    start = time.time()
    for messages, qa in tqdm(pairs, desc=f"LoCoMo [{adapter.name}]"):
        adapter.reset()
        adapter.ingest(messages)
        prediction = adapter.query(qa["question"], llm_call)

        category = qa.get("category", 0)
        gold = qa.get("answer", "")

        # Skip adversarial (category 5) -- no gold answer
        if category == 5:
            continue

        score = judge_fn(qa["question"], gold, prediction)

        result.total_questions += 1
        result.correct += int(score >= 0.5)
        result.scores_by_category.setdefault(category, []).append(score)
        result.per_question.append({
            "question": qa["question"],
            "gold": gold,
            "prediction": prediction,
            "score": score,
            "category": category,
        })

    result.elapsed_seconds = time.time() - start
    return result
