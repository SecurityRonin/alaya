"""Zep adapter using the zep-cloud Python SDK."""

import os
import uuid

from adapters.base import MemoryAdapter, Message

try:
    from zep_cloud.client import Zep
    from zep_cloud.types import Message as ZepMessage
except ImportError:
    Zep = None
    ZepMessage = None


class ZepAdapter(MemoryAdapter):
    """Adapter for Zep Cloud (zep-cloud package).

    Requires: pip install zep-cloud
    Env vars: ZEP_API_KEY
    """

    def __init__(self) -> None:
        if Zep is None:
            raise ImportError("zep-cloud not installed. Run: pip install zep-cloud")
        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise ValueError("ZEP_API_KEY environment variable not set")
        self._client = Zep(api_key=api_key)
        self._user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._session_ids: list[str] = []

    def reset(self) -> None:
        try:
            self._client.user.delete(self._user_id)
        except Exception:
            pass
        self._user_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._session_ids = []
        self._client.user.add(user_id=self._user_id)

    def ingest(self, messages: list[Message]) -> None:
        # Ensure user exists
        try:
            self._client.user.add(user_id=self._user_id)
        except Exception:
            pass  # already exists

        # Group by session
        sessions: dict[str, list[Message]] = {}
        for m in messages:
            sessions.setdefault(m.session_id, []).append(m)

        for session_id, session_msgs in sessions.items():
            zep_session_id = f"{self._user_id}_{session_id}"
            self._client.memory.add_session(
                session_id=zep_session_id, user_id=self._user_id
            )
            self._session_ids.append(zep_session_id)

            zep_msgs = [
                ZepMessage(
                    role=m.role,
                    content=m.content,
                    role_type="user" if m.role == "user" else "assistant",
                )
                for m in session_msgs
            ]
            self._client.memory.add(zep_session_id, messages=zep_msgs)

    def query(self, question: str, llm_call: callable) -> str:
        # Search the user's knowledge graph
        results = self._client.graph.search(
            user_id=self._user_id,
            query=question,
            scope="edges",
            limit=10,
        )
        context = "\n".join(
            edge.fact for edge in results if hasattr(edge, "fact") and edge.fact
        )
        prompt = (
            f"Based on the following facts from the user's memory, answer the question.\n\n"
            f"Facts:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return llm_call(prompt)

    @property
    def name(self) -> str:
        return "Zep"
