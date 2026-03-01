"""Full-context baseline: stuff entire conversation into the prompt."""

from adapters.base import MemoryAdapter, Message


class FullContextAdapter(MemoryAdapter):
    """Baseline that feeds the entire conversation history as context.

    No memory system involved -- just concatenate all messages into the prompt.
    This establishes the upper bound for what in-context learning can achieve.
    """

    def __init__(self) -> None:
        self._history: list[Message] = []

    def reset(self) -> None:
        self._history = []

    def ingest(self, messages: list[Message]) -> None:
        self._history.extend(messages)

    def query(self, question: str, llm_call: callable) -> str:
        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in self._history
        )
        prompt = (
            f"Based on the following conversation history, answer the question.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return llm_call(prompt)

    @property
    def name(self) -> str:
        return "Full-context"
