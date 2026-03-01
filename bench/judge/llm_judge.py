"""LLM client and judge scoring for benchmark evaluation."""

from __future__ import annotations

import os
from functools import partial
from typing import Callable

import litellm


# ── Provider presets ──

PROVIDERS = {
    "openai": {
        "model_prefix": "",
        "default_model": "gpt-4o-mini",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "model_prefix": "openrouter/",
        "default_model": "openai/gpt-4o-mini",
        "api_base": None,  # litellm handles openrouter natively
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "opencode": {
        "model_prefix": "openai/",
        "default_model": "big-pickle",
        "api_base": "https://opencode.ai/zen/v1",
        "api_key_env": "OPENCODE_API_KEY",
    },
}


def provider_config() -> dict:
    """Build LLM call config from environment variables.

    Provider selection (LLM_PROVIDER env var):
        openai    — OpenAI direct (default)
        openrouter — OpenRouter (set OPENROUTER_API_KEY)
        opencode  — OpenCode Zen free models (set OPENCODE_API_KEY)

    Direct overrides (take precedence over provider presets):
        LLM_MODEL    — model name
        LLM_API_BASE — custom API base URL
        LLM_API_KEY  — custom API key
    """
    provider_name = os.environ.get("LLM_PROVIDER", "openai")
    preset = PROVIDERS.get(provider_name, PROVIDERS["openai"])

    # Start from preset defaults
    raw_model = os.environ.get("LLM_MODEL") or preset["default_model"]
    model = preset["model_prefix"] + raw_model
    api_base = preset["api_base"]
    api_key = os.environ.get(preset["api_key_env"])

    # Direct overrides win
    if os.environ.get("LLM_MODEL") and not os.environ.get("LLM_PROVIDER"):
        model = os.environ["LLM_MODEL"]
    if os.environ.get("LLM_API_BASE"):
        api_base = os.environ["LLM_API_BASE"]
    if os.environ.get("LLM_API_KEY"):
        api_key = os.environ["LLM_API_KEY"]

    return {"model": model, "api_base": api_base, "api_key": api_key}


def llm_call(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 512,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call an LLM with a prompt and return the response text.

    Uses litellm so any provider works (OpenAI, Anthropic, OpenRouter, OpenCode Zen).
    """
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    if api_key is not None:
        kwargs["api_key"] = api_key
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content.strip()


def make_llm_call(model: str | None = None) -> Callable:
    """Create an llm_call partial with provider config from env."""
    cfg = provider_config()
    return partial(
        llm_call,
        model=model or cfg["model"],
        api_base=cfg["api_base"],
        api_key=cfg["api_key"],
    )


def score_answer(
    question: str,
    gold: str,
    prediction: str,
    judge_fn: Callable | None = None,
) -> float:
    """Score a prediction against a gold answer using LLM-as-Judge.

    Uses the LongMemEval judge prompt (binary yes/no).
    Returns 1.0 for correct, 0.0 for incorrect.
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
        cfg = provider_config()
        judge_model = os.environ.get("JUDGE_MODEL", cfg["model"])
        judge_fn = partial(
            llm_call,
            model=judge_model,
            max_tokens=10,
            api_base=cfg["api_base"],
            api_key=cfg["api_key"],
        )
    result = judge_fn(judge_prompt)
    return 1.0 if "yes" in result.lower() else 0.0
