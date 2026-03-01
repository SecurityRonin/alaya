"""LLM client and judge scoring for benchmark evaluation."""

from __future__ import annotations

import os
from functools import partial
from typing import Callable

import litellm


def llm_call(prompt: str, model: str | None = None, max_tokens: int = 512) -> str:
    """Call an LLM with a prompt and return the response text.

    Uses litellm so any provider works (OpenAI, Anthropic, Gemini, local).
    Model is configured via LLM_MODEL env var or explicit parameter.
    """
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def make_llm_call(model: str | None = None) -> Callable:
    """Create an llm_call partial with a specific model."""
    return partial(llm_call, model=model)


def score_answer(
    question: str,
    gold: str,
    prediction: str,
    judge_fn: Callable | None = None,
) -> float:
    """Score a prediction against a gold answer using LLM-as-Judge.

    Uses the LongMemEval judge prompt (binary yes/no).
    Returns 1.0 for correct, 0.0 for incorrect.

    Args:
        question: The benchmark question.
        gold: The gold reference answer.
        prediction: The system's predicted answer.
        judge_fn: Optional custom judge callable(prompt) -> str.
            Defaults to llm_call with JUDGE_MODEL.
    """
    judge_prompt = (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response is equivalent to the correct answer or contains "
        "all the intermediate steps to get the correct answer, you should also answer "
        "yes. If the response only contains a subset of the information required by "
        "the answer, answer no.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {gold}\n"
        f"Model Response: {prediction}\n\n"
        "Answer (yes or no):"
    )
    if judge_fn is None:
        judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        judge_fn = partial(llm_call, model=judge_model, max_tokens=10)
    result = judge_fn(judge_prompt)
    return 1.0 if "yes" in result.lower() else 0.0
