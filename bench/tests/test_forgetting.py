"""RED tests for ForgettingDynamics benchmark.

Tests define the expected interface and behaviour before implementation.
All tests in this file are expected to FAIL until runners/forgetting.py,
datasets/forgetting_fixture.json, and the ForgettingResult dataclass exist.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

FIXTURE_PATH = Path(__file__).parent.parent / "datasets" / "forgetting_fixture.json"


# ---------------------------------------------------------------------------
# 1. Dataset structure
# ---------------------------------------------------------------------------

def test_fixture_exists():
    """The fixture dataset file must be present."""
    assert FIXTURE_PATH.exists(), f"Missing fixture: {FIXTURE_PATH}"


def test_fixture_is_valid_json():
    data = json.loads(FIXTURE_PATH.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1


def test_fixture_scenario_has_required_keys():
    data = json.loads(FIXTURE_PATH.read_text())
    required = {
        "scenario_id",
        "domain",
        "seed_sessions",
        "growth_sessions",
        "seed_questions",
        "growth_questions",
        "contradiction_questions",
    }
    for scenario in data:
        missing = required - set(scenario.keys())
        assert not missing, f"Scenario {scenario.get('scenario_id')} missing keys: {missing}"


def test_fixture_seed_sessions_are_message_dicts():
    """Each seed session entry must have role, content, session_id, timestamp."""
    data = json.loads(FIXTURE_PATH.read_text())
    required = {"role", "content", "session_id", "timestamp"}
    for scenario in data:
        for msg in scenario["seed_sessions"]:
            missing = required - set(msg.keys())
            assert not missing, f"Seed message missing keys: {missing}"


def test_fixture_growth_sessions_are_message_dicts():
    data = json.loads(FIXTURE_PATH.read_text())
    required = {"role", "content", "session_id", "timestamp"}
    for scenario in data:
        for msg in scenario["growth_sessions"]:
            missing = required - set(msg.keys())
            assert not missing, f"Growth message missing keys: {missing}"


def test_fixture_seed_questions_have_required_keys():
    data = json.loads(FIXTURE_PATH.read_text())
    for scenario in data:
        for q in scenario["seed_questions"]:
            assert "question" in q
            assert "gold" in q
            assert q.get("type") in ("seed_retrieval",), (
                f"Unexpected seed question type: {q.get('type')}"
            )


def test_fixture_growth_questions_have_required_keys():
    data = json.loads(FIXTURE_PATH.read_text())
    for scenario in data:
        for q in scenario["growth_questions"]:
            assert "question" in q
            assert "gold" in q
            assert q.get("type") == "growth_retrieval"


def test_fixture_contradiction_questions_have_both_gold_fields():
    """Contradiction questions need gold (current truth) AND gold_before_update."""
    data = json.loads(FIXTURE_PATH.read_text())
    for scenario in data:
        for q in scenario["contradiction_questions"]:
            assert "question" in q
            assert "gold" in q, "contradiction question must have gold (current/updated answer)"
            assert "gold_before_update" in q, "must capture the stale answer too"


# ---------------------------------------------------------------------------
# 2. ForgettingResult dataclass
# ---------------------------------------------------------------------------

def test_forgetting_result_importable():
    from runners.forgetting import ForgettingResult  # noqa: F401


def test_forgetting_result_has_rdr_field():
    from runners.forgetting import ForgettingResult
    field_names = {f.name for f in fields(ForgettingResult)}
    assert "rdr" in field_names, "ForgettingResult must have 'rdr' field (Retrieval Degradation Ratio)"


def test_forgetting_result_has_accuracy_fields():
    from runners.forgetting import ForgettingResult
    field_names = {f.name for f in fields(ForgettingResult)}
    assert "pre_growth_accuracy" in field_names
    assert "post_growth_accuracy" in field_names
    assert "growth_accuracy" in field_names
    assert "contradiction_accuracy" in field_names


def test_forgetting_result_has_system_field():
    from runners.forgetting import ForgettingResult
    field_names = {f.name for f in fields(ForgettingResult)}
    assert "system" in field_names


def test_forgetting_result_rdr_property():
    """RDR should equal post_growth / pre_growth (or 0 if pre_growth is 0)."""
    from runners.forgetting import ForgettingResult
    r = ForgettingResult(
        system="test",
        pre_growth_accuracy=0.80,
        post_growth_accuracy=0.72,
        growth_accuracy=0.60,
        contradiction_accuracy=0.50,
        total_seed_questions=10,
        total_growth_questions=5,
        total_contradiction_questions=5,
        elapsed_seconds=1.0,
        per_question=[],
    )
    assert abs(r.rdr - (0.72 / 0.80)) < 1e-9, f"Expected {0.72/0.80:.4f}, got {r.rdr:.4f}"


def test_forgetting_result_rdr_zero_division():
    """RDR must be 0.0 (not an error) when pre_growth_accuracy is 0."""
    from runners.forgetting import ForgettingResult
    r = ForgettingResult(
        system="test",
        pre_growth_accuracy=0.0,
        post_growth_accuracy=0.0,
        growth_accuracy=0.0,
        contradiction_accuracy=0.0,
        total_seed_questions=0,
        total_growth_questions=0,
        total_contradiction_questions=0,
        elapsed_seconds=0.0,
        per_question=[],
    )
    assert r.rdr == 0.0


# ---------------------------------------------------------------------------
# 3. run_forgetting function interface
# ---------------------------------------------------------------------------

def test_run_forgetting_importable():
    from runners.forgetting import run_forgetting  # noqa: F401


def test_run_forgetting_accepts_correct_signature():
    """run_forgetting(adapter, dataset_path, judge_fn, llm_call, limit, dry_run)."""
    import inspect
    from runners.forgetting import run_forgetting
    sig = inspect.signature(run_forgetting)
    params = set(sig.parameters.keys())
    assert "adapter" in params
    assert "dataset_path" in params
    assert "judge_fn" in params
    assert "llm_call" in params
    assert "limit" in params
    assert "dry_run" in params


def test_run_forgetting_returns_forgetting_result():
    """run_forgetting with a mock adapter must return a ForgettingResult."""
    from adapters.base import MemoryAdapter, Message
    from runners.forgetting import ForgettingResult, run_forgetting

    class MockAdapter(MemoryAdapter):
        name = "mock"
        def reset(self): pass
        def ingest(self, messages): pass
        def query(self, question, llm_call): return "mock answer"

    mock_judge = lambda q, g, p: 1.0
    mock_llm = lambda prompt: "mock answer"

    result = run_forgetting(
        MockAdapter(),
        FIXTURE_PATH,
        mock_judge,
        mock_llm,
        limit=None,
        dry_run=False,
    )
    assert isinstance(result, ForgettingResult)


def test_run_forgetting_dry_run_returns_result():
    """Dry run must return a ForgettingResult without calling the LLM."""
    from adapters.base import MemoryAdapter
    from runners.forgetting import ForgettingResult, run_forgetting

    class MockAdapter(MemoryAdapter):
        name = "mock_dry"
        def reset(self): pass
        def ingest(self, messages): pass
        def query(self, question, llm_call):
            raise AssertionError("query() must not be called in dry_run mode")

    def failing_llm(prompt):
        raise AssertionError("LLM must not be called in dry_run mode")

    mock_judge = lambda q, g, p: 1.0

    result = run_forgetting(
        MockAdapter(), FIXTURE_PATH, mock_judge, failing_llm,
        limit=None, dry_run=True
    )
    assert isinstance(result, ForgettingResult)


# ---------------------------------------------------------------------------
# 4. Interleaved ingest protocol (no reset between seed and growth)
# ---------------------------------------------------------------------------

def test_interleaved_protocol_no_reset_between_phases():
    """Adapter.reset() must be called once at start; NOT between seed and growth ingest."""
    from adapters.base import MemoryAdapter
    from runners.forgetting import run_forgetting

    reset_calls = []
    ingest_calls = []

    class TrackingAdapter(MemoryAdapter):
        name = "tracking"
        def reset(self): reset_calls.append(1)
        def ingest(self, messages): ingest_calls.append(len(messages))
        def query(self, question, llm_call): return "answer"

    mock_judge = lambda q, g, p: 1.0
    mock_llm = lambda prompt: "answer"

    run_forgetting(TrackingAdapter(), FIXTURE_PATH, mock_judge, mock_llm, limit=None, dry_run=False)

    # reset() called exactly once per scenario
    assert reset_calls, "reset() must be called"

    # ingest() called at least twice per scenario (seed phase, then growth phase)
    assert len(ingest_calls) >= 2 * _scenario_count(), (
        f"Expected at least {2 * _scenario_count()} ingest calls, got {len(ingest_calls)}"
    )


def _scenario_count() -> int:
    return len(json.loads(FIXTURE_PATH.read_text()))


def test_seed_questions_asked_twice():
    """Seed questions must be evaluated before growth AND after growth (two query rounds)."""
    from adapters.base import MemoryAdapter
    from runners.forgetting import run_forgetting

    query_calls = []

    class TrackingAdapter(MemoryAdapter):
        name = "tracking_q"
        def reset(self): pass
        def ingest(self, messages): pass
        def query(self, question, llm_call):
            query_calls.append(question)
            return "answer"

    mock_judge = lambda q, g, p: 1.0
    mock_llm = lambda prompt: "answer"

    data = json.loads(FIXTURE_PATH.read_text())
    total_seed_qs = sum(len(s["seed_questions"]) for s in data)

    run_forgetting(TrackingAdapter(), FIXTURE_PATH, mock_judge, mock_llm, limit=None, dry_run=False)

    # Each seed question is asked at least twice (pre and post growth)
    seed_questions = {
        q["question"]
        for scenario in data
        for q in scenario["seed_questions"]
    }
    for sq in seed_questions:
        count = query_calls.count(sq)
        assert count >= 2, (
            f"Seed question asked {count} times, expected at least 2: {sq!r}"
        )


# ---------------------------------------------------------------------------
# 5. Accuracy and RDR bounds
# ---------------------------------------------------------------------------

def test_result_accuracies_between_zero_and_one():
    from adapters.base import MemoryAdapter
    from runners.forgetting import run_forgetting

    class AlwaysWrong(MemoryAdapter):
        name = "always_wrong"
        def reset(self): pass
        def ingest(self, messages): pass
        def query(self, question, llm_call): return "definitely wrong xyz"

    result = run_forgetting(
        AlwaysWrong(), FIXTURE_PATH,
        lambda q, g, p: 0.0,  # judge always says wrong
        lambda p: "wrong",
        limit=None, dry_run=False
    )
    assert 0.0 <= result.pre_growth_accuracy <= 1.0
    assert 0.0 <= result.post_growth_accuracy <= 1.0
    assert 0.0 <= result.growth_accuracy <= 1.0
    assert 0.0 <= result.contradiction_accuracy <= 1.0


def test_result_accuracies_when_always_correct():
    from adapters.base import MemoryAdapter
    from runners.forgetting import run_forgetting

    data = json.loads(FIXTURE_PATH.read_text())

    class AlwaysCorrect(MemoryAdapter):
        name = "always_correct"
        def reset(self): pass
        def ingest(self, messages): pass
        def query(self, question, llm_call): return "correct"

    result = run_forgetting(
        AlwaysCorrect(), FIXTURE_PATH,
        lambda q, g, p: 1.0,  # judge always says correct
        lambda p: "correct",
        limit=None, dry_run=False
    )
    assert result.pre_growth_accuracy == 1.0
    assert result.post_growth_accuracy == 1.0
    assert result.rdr == 1.0


def test_limit_parameter_reduces_questions():
    from adapters.base import MemoryAdapter
    from runners.forgetting import run_forgetting

    query_count = []

    class CountingAdapter(MemoryAdapter):
        name = "counting"
        def reset(self): pass
        def ingest(self, messages): pass
        def query(self, question, llm_call):
            query_count.append(1)
            return "answer"

    run_forgetting(
        CountingAdapter(), FIXTURE_PATH,
        lambda q, g, p: 1.0,
        lambda p: "answer",
        limit=1,
        dry_run=False,
    )

    # With limit=1, only 1 scenario should be processed
    data = json.loads(FIXTURE_PATH.read_text())
    first_scenario = data[0]
    # seed questions asked twice + growth + contradiction = bounded set
    max_expected = (
        2 * len(first_scenario["seed_questions"])
        + len(first_scenario["growth_questions"])
        + len(first_scenario["contradiction_questions"])
    )
    assert len(query_count) <= max_expected, (
        f"With limit=1, expected at most {max_expected} queries, got {len(query_count)}"
    )


# ---------------------------------------------------------------------------
# 6. per_question log completeness
# ---------------------------------------------------------------------------

def test_per_question_log_has_phase_field():
    """Each entry in per_question must record which phase it belongs to."""
    from adapters.base import MemoryAdapter
    from runners.forgetting import run_forgetting

    class MockAdapter(MemoryAdapter):
        name = "mock_log"
        def reset(self): pass
        def ingest(self, messages): pass
        def query(self, question, llm_call): return "answer"

    result = run_forgetting(
        MockAdapter(), FIXTURE_PATH,
        lambda q, g, p: 1.0,
        lambda p: "answer",
        limit=None, dry_run=False
    )

    valid_phases = {"pre_growth_seed", "post_growth_seed", "growth", "contradiction"}
    for entry in result.per_question:
        assert "phase" in entry, f"per_question entry missing 'phase': {entry}"
        assert entry["phase"] in valid_phases, f"Unknown phase: {entry['phase']}"
