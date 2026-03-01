"""Mem0 adapter using the mem0ai Python SDK."""

import os
import uuid

from adapters.base import MemoryAdapter, Message

try:
    from mem0 import Memory
except ImportError:
    Memory = None


class Mem0Adapter(MemoryAdapter):
    """Adapter for Mem0 (mem0ai package).

    Requires: pip install mem0ai
    Env vars: OPENAI_API_KEY (for Mem0's internal LLM extraction)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        if Memory is None:
            raise ImportError("mem0ai not installed. Run: pip install mem0ai")
        config = {
            "llm": {
                "provider": "openai",
                "config": {"model": model, "temperature": 0.1},
            },
        }
        self._mem = Memory.from_config(config)
        self._user_id = f"bench_{uuid.uuid4().hex[:8]}"

    def reset(self) -> None:
        try:
            self._mem.delete_all(user_id=self._user_id)
        except Exception:
            pass
        self._user_id = f"bench_{uuid.uuid4().hex[:8]}"

    def ingest(self, messages: list[Message]) -> None:
        # Group messages by session and feed as conversations
        sessions: dict[str, list[dict]] = {}
        for m in messages:
            sessions.setdefault(m.session_id, []).append(
                {"role": m.role, "content": m.content}
            )
        for session_msgs in sessions.values():
            self._mem.add(session_msgs, user_id=self._user_id)

    def query(self, question: str, llm_call: callable) -> str:
        results = self._mem.search(question, user_id=self._user_id)
        memories = results.get("results", [])
        context = "\n".join(m["memory"] for m in memories) if memories else ""
        prompt = (
            f"Based on the following memories about the user, answer the question.\n\n"
            f"Memories:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return llm_call(prompt)

    @property
    def name(self) -> str:
        return "Mem0"
