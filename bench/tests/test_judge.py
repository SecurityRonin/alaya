"""Tests for LLM client and judge scoring."""

import os
from unittest.mock import patch, MagicMock

import pytest

from judge.llm_judge import score_answer, llm_call, make_llm_call, provider_config, judge_config


# ── Provider configuration tests ──

def test_provider_config_defaults():
    """Default config uses gpt-4o-mini with no custom api_base."""
    with patch.dict(os.environ, {}, clear=True):
        cfg = provider_config()
    assert cfg["model"] == "gpt-4o-mini"
    assert cfg["api_base"] is None
    assert cfg["api_key"] is None


def test_provider_config_openrouter():
    """OpenRouter sets model prefix and uses OPENROUTER_API_KEY."""
    env = {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-test-key",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = provider_config()
    assert cfg["model"] == "openrouter/openai/gpt-4o-mini"
    assert cfg["api_key"] == "sk-or-test-key"
    assert cfg["api_base"] is None  # openrouter uses litellm native routing


def test_provider_config_openrouter_custom_model():
    """OpenRouter with a custom model name."""
    env = {
        "LLM_PROVIDER": "openrouter",
        "LLM_MODEL": "google/gemini-2.0-flash-001",
        "OPENROUTER_API_KEY": "sk-or-test-key",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = provider_config()
    assert cfg["model"] == "openrouter/google/gemini-2.0-flash-001"


def test_provider_config_opencode_zen():
    """OpenCode Zen sets api_base and uses openai/ prefix."""
    env = {
        "LLM_PROVIDER": "opencode",
        "LLM_MODEL": "big-pickle",
        "OPENCODE_API_KEY": "oc-test-key",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = provider_config()
    assert cfg["model"] == "openai/big-pickle"
    assert cfg["api_base"] == "https://opencode.ai/zen/v1"
    assert cfg["api_key"] == "oc-test-key"


def test_provider_config_opencode_zen_default_model():
    """OpenCode Zen defaults to big-pickle when no model specified."""
    env = {
        "LLM_PROVIDER": "opencode",
        "OPENCODE_API_KEY": "oc-test-key",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = provider_config()
    assert cfg["model"] == "openai/big-pickle"


def test_provider_config_direct_override():
    """Direct env vars override provider defaults."""
    env = {
        "LLM_MODEL": "claude-3-haiku-20240307",
        "LLM_API_BASE": "https://custom.api.com/v1",
        "LLM_API_KEY": "custom-key",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = provider_config()
    assert cfg["model"] == "claude-3-haiku-20240307"
    assert cfg["api_base"] == "https://custom.api.com/v1"
    assert cfg["api_key"] == "custom-key"


# ── Judge config tests ──

def test_judge_config_falls_back_to_llm_config():
    """When no JUDGE_* vars set, judge uses LLM provider config."""
    env = {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-key",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = judge_config()
    assert cfg["model"] == "openrouter/openai/gpt-4o-mini"
    assert cfg["api_key"] == "sk-or-key"


def test_judge_config_separate_provider():
    """Judge can use a different provider than system LLM."""
    env = {
        "LLM_PROVIDER": "openrouter",
        "LLM_MODEL": "google/gemini-2.0-flash-001",
        "OPENROUTER_API_KEY": "sk-or-key",
        "JUDGE_PROVIDER": "opencode",
        "OPENCODE_API_KEY": "oc-key",
    }
    with patch.dict(os.environ, env, clear=True):
        llm_cfg = provider_config()
        j_cfg = judge_config()
    # System LLM uses OpenRouter + Gemini
    assert llm_cfg["model"] == "openrouter/google/gemini-2.0-flash-001"
    # Judge uses OpenCode Zen + default big-pickle
    assert j_cfg["model"] == "openai/big-pickle"
    assert j_cfg["api_base"] == "https://opencode.ai/zen/v1"
    assert j_cfg["api_key"] == "oc-key"


def test_judge_config_model_override():
    """JUDGE_MODEL overrides the model within the judge provider."""
    env = {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-key",
        "JUDGE_MODEL": "meta-llama/llama-3.3-70b-instruct:free",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = judge_config()
    # Judge model overridden, but still uses LLM provider (openrouter)
    assert cfg["model"] == "openrouter/meta-llama/llama-3.3-70b-instruct:free"


def test_judge_config_separate_provider_and_model():
    """Judge can have both different provider and model."""
    env = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-openai",
        "JUDGE_PROVIDER": "openrouter",
        "JUDGE_MODEL": "meta-llama/llama-3.3-70b-instruct:free",
        "OPENROUTER_API_KEY": "sk-or-key",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = judge_config()
    assert cfg["model"] == "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    assert cfg["api_key"] == "sk-or-key"


def test_score_answer_uses_judge_config():
    """score_answer uses judge_config when no judge_fn provided."""
    env = {
        "JUDGE_PROVIDER": "opencode",
        "JUDGE_MODEL": "big-pickle",
        "OPENCODE_API_KEY": "oc-key",
    }
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "yes"

    with patch.dict(os.environ, env, clear=True):
        with patch("judge.llm_judge.litellm.completion", return_value=mock_response) as mock_comp:
            score = score_answer("Q?", "A", "A is correct")

    assert score == 1.0
    call_kwargs = mock_comp.call_args[1]
    assert call_kwargs["model"] == "openai/big-pickle"
    assert call_kwargs["api_base"] == "https://opencode.ai/zen/v1"
    assert call_kwargs["api_key"] == "oc-key"


# ── llm_call passes provider config to litellm ──

def test_llm_call_passes_api_base():
    """llm_call forwards api_base and api_key to litellm."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "hello"

    with patch("judge.llm_judge.litellm.completion", return_value=mock_response) as mock_comp:
        result = llm_call(
            "Say hello",
            model="openai/big-pickle",
            api_base="https://opencode.ai/zen/v1",
            api_key="oc-key",
        )

    assert result == "hello"
    mock_comp.assert_called_once()
    call_kwargs = mock_comp.call_args[1]
    assert call_kwargs["model"] == "openai/big-pickle"
    assert call_kwargs["api_base"] == "https://opencode.ai/zen/v1"
    assert call_kwargs["api_key"] == "oc-key"


def test_llm_call_omits_none_kwargs():
    """llm_call doesn't pass api_base/api_key when they're None."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "hi"

    with patch("judge.llm_judge.litellm.completion", return_value=mock_response) as mock_comp:
        llm_call("Say hi", model="gpt-4o-mini")

    call_kwargs = mock_comp.call_args[1]
    assert "api_base" not in call_kwargs
    assert "api_key" not in call_kwargs


# ── make_llm_call uses provider_config ──

def test_make_llm_call_uses_provider_config():
    """make_llm_call reads provider config from env."""
    env = {
        "LLM_PROVIDER": "opencode",
        "LLM_MODEL": "big-pickle",
        "OPENCODE_API_KEY": "oc-key",
    }
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "result"

    with patch.dict(os.environ, env, clear=True):
        fn = make_llm_call()

    with patch("judge.llm_judge.litellm.completion", return_value=mock_response) as mock_comp:
        fn("test prompt")

    call_kwargs = mock_comp.call_args[1]
    assert call_kwargs["model"] == "openai/big-pickle"
    assert call_kwargs["api_base"] == "https://opencode.ai/zen/v1"
    assert call_kwargs["api_key"] == "oc-key"


# ── Score answer tests (unchanged) ──

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


# ── Live integration test (skipped without API key) ──

def test_llm_call_live():
    """Test that llm_call works with a live backend."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    result = llm_call("Say 'hello'", model="gpt-4o-mini", max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0
