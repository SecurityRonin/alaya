"""Naive RAG baseline: ChromaDB vector similarity, no lifecycle."""

import uuid

import chromadb

from adapters.base import MemoryAdapter, Message


class NaiveRAGAdapter(MemoryAdapter):
    """Baseline that uses simple vector similarity retrieval.

    Messages are chunked and stored in ChromaDB. Queries retrieve
    top-k similar chunks by embedding distance. No forgetting,
    no consolidation, no graph -- just vector search.
    """

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k
        self._client = chromadb.Client()  # ephemeral in-memory
        self._collection = self._client.create_collection(
            name=f"bench_{uuid.uuid4().hex[:8]}",
        )

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.create_collection(
            name=f"bench_{uuid.uuid4().hex[:8]}",
        )

    def ingest(self, messages: list[Message]) -> None:
        docs = []
        ids = []
        metadatas = []
        for i, m in enumerate(messages):
            docs.append(f"{m.role}: {m.content}")
            ids.append(f"{m.session_id}_{m.timestamp}_{i}")
            metadatas.append({"role": m.role, "session_id": m.session_id})
        if docs:
            self._collection.add(documents=docs, ids=ids, metadatas=metadatas)

    def query(self, question: str, llm_call: callable) -> str:
        results = self._collection.query(
            query_texts=[question],
            n_results=min(self._top_k, self._collection.count()),
        )
        context = "\n".join(results["documents"][0]) if results["documents"][0] else ""
        prompt = (
            f"Based on the following retrieved context, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return llm_call(prompt)

    @property
    def name(self) -> str:
        return "Naive RAG"
