"""End-to-end smoke test using full-context adapter with mock LLM."""

import json
import tempfile
from pathlib import Path

from adapters.base import Message
from adapters.fullcontext import FullContextAdapter
from runners.locomo import conversation_to_messages, run_locomo


MINI_LOCOMO = [
    {
        "sample_id": "test_1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"speaker": "Alice", "dia_id": "1_1", "text": "I just got a golden retriever puppy named Max."},
                {"speaker": "Bob", "dia_id": "1_2", "text": "That's adorable! How old is Max?"},
                {"speaker": "Alice", "dia_id": "1_3", "text": "He's 3 months old."},
            ],
            "session_1_date_time": "2024-01-15 10:30:00",
        },
        "qa": [
            {
                "question": "What is the name of Alice's puppy?",
                "answer": "Max",
                "category": 1,
                "evidence": ["1_1"],
            }
        ],
    }
]


def test_conversation_to_messages():
    msgs = conversation_to_messages(MINI_LOCOMO[0])
    assert len(msgs) == 3
    assert msgs[0].role == "user"
    assert msgs[0].content == "I just got a golden retriever puppy named Max."
    assert msgs[0].session_id == "session_1"


def test_e2e_locomo_with_mock():
    """Run LoCoMo on a tiny dataset with a mock LLM and judge."""
    # Write mini dataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_LOCOMO, f)
        dataset_path = Path(f.name)

    adapter = FullContextAdapter()

    def mock_llm(prompt: str) -> str:
        # The prompt should contain "Max" from the conversation
        if "Max" in prompt:
            return "The puppy's name is Max."
        return "I don't know."

    def mock_judge(question: str, gold: str, prediction: str) -> float:
        return 1.0 if gold.lower() in prediction.lower() else 0.0

    result = run_locomo(adapter, dataset_path, mock_judge, mock_llm)

    assert result.system == "Full-context"
    assert result.total_questions == 1
    assert result.correct == 1
    assert result.accuracy == 1.0

    dataset_path.unlink()


def test_dry_run():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_LOCOMO, f)
        dataset_path = Path(f.name)

    adapter = FullContextAdapter()
    result = run_locomo(adapter, dataset_path, lambda *a: 0.0, lambda p: "", dry_run=True)
    assert result.total_questions == 1
    assert result.correct == 0  # dry run doesn't score

    dataset_path.unlink()
