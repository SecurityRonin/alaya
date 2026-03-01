"""Tests for LLM client and judge scoring."""

import os

import pytest

from judge.llm_judge import score_answer


def test_llm_call_uses_mock():
    """Test that llm_call works with a live backend."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    from judge.llm_judge import llm_call
    result = llm_call("Say 'hello'", model="gpt-4o-mini", max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0


def test_score_answer_correct():
    """Test scoring for correct answers."""
    score = score_answer(
        question="What is 2+2?",
        gold="4",
        prediction="The answer is 4.",
        judge_fn=lambda prompt: "yes",
    )
    assert score == 1.0


def test_score_answer_wrong():
    score = score_answer(
        question="What is 2+2?",
        gold="4",
        prediction="The answer is 5.",
        judge_fn=lambda prompt: "no",
    )
    assert score == 0.0
