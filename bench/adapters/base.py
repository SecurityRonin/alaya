"""Base class for memory system adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    """A single conversation message."""

    role: str  # "user" or "assistant"
    content: str
    session_id: str
    timestamp: str  # ISO-8601


class MemoryAdapter(ABC):
    """Interface that each memory system must implement."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state. Called before each conversation."""
        ...

    @abstractmethod
    def ingest(self, messages: list[Message]) -> None:
        """Feed conversation messages into the memory system."""
        ...

    @abstractmethod
    def query(self, question: str, llm_call: "callable") -> str:
        """Retrieve relevant context and generate an answer.

        Args:
            question: The benchmark question to answer.
            llm_call: A callable(prompt: str) -> str for LLM generation.
                The adapter retrieves context, builds a prompt, and calls this.

        Returns:
            The generated answer string.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """System name for results table."""
        ...
