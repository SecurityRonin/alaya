"""Tests for the MemoryAdapter interface and baseline adapters."""

import pytest

from adapters.base import MemoryAdapter, Message
from adapters.fullcontext import FullContextAdapter
from adapters.naive_rag import NaiveRAGAdapter


# ── Base class tests ──

def test_message_dataclass():
    msg = Message(role="user", content="hello", session_id="s1", timestamp="2024-01-01T00:00:00")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_adapter_cannot_be_instantiated():
    """ABC should not be directly instantiable."""
    with pytest.raises(TypeError):
        MemoryAdapter()


# ── FullContext adapter tests ──

def test_fullcontext_name():
    adapter = FullContextAdapter()
    assert adapter.name == "Full-context"


def test_fullcontext_ingest_and_query():
    adapter = FullContextAdapter()
    adapter.reset()
    msgs = [
        Message(role="user", content="My name is Alice.", session_id="s1", timestamp="2024-01-01T00:00:00"),
        Message(role="assistant", content="Hello Alice!", session_id="s1", timestamp="2024-01-01T00:01:00"),
    ]
    adapter.ingest(msgs)

    # Mock LLM that echoes the prompt length
    def mock_llm(prompt: str) -> str:
        return f"prompt_len={len(prompt)}"

    answer = adapter.query("What is my name?", mock_llm)
    assert "prompt_len=" in answer
    # The prompt should contain the ingested messages
    assert len(adapter._history) == 2


def test_fullcontext_reset_clears():
    adapter = FullContextAdapter()
    msgs = [Message(role="user", content="data", session_id="s1", timestamp="2024-01-01T00:00:00")]
    adapter.ingest(msgs)
    assert len(adapter._history) > 0
    adapter.reset()
    assert len(adapter._history) == 0


# ── NaiveRAG adapter tests ──

def test_naive_rag_name():
    adapter = NaiveRAGAdapter()
    assert adapter.name == "Naive RAG"


def test_naive_rag_ingest_and_query():
    adapter = NaiveRAGAdapter()
    adapter.reset()
    msgs = [
        Message(role="user", content="I love hiking in the mountains.", session_id="s1", timestamp="2024-01-01T00:00:00"),
        Message(role="assistant", content="That sounds fun!", session_id="s1", timestamp="2024-01-01T00:01:00"),
        Message(role="user", content="My favorite peak is Mount Rainier.", session_id="s1", timestamp="2024-01-01T00:02:00"),
    ]
    adapter.ingest(msgs)

    def mock_llm(prompt: str) -> str:
        return "Mount Rainier" if "Rainier" in prompt else "unknown"

    answer = adapter.query("What is my favorite peak?", mock_llm)
    assert "Rainier" in answer
