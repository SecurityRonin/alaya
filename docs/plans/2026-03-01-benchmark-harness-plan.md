# Benchmark Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python evaluation harness that runs LoCoMo and LongMemEval benchmarks against Alaya and competing memory systems, producing scores for the taxonomy paper.

**Architecture:** Adapter pattern with a `MemoryAdapter` ABC. Each system implements `reset()`, `ingest()`, `query()`. Runners load benchmark datasets, feed conversations through adapters, score with LLM-as-Judge, and output results. CLI entrypoint selects benchmark + systems.

**Tech Stack:** Python 3.11+, litellm, chromadb, mem0ai, zep-cloud, huggingface_hub, pytest

---

### Task 1: Project Scaffold

**Files:**
- Create: `bench/pyproject.toml`
- Create: `bench/README.md`
- Create: `bench/datasets/.gitkeep`
- Create: `bench/results/.gitkeep`

**Step 1: Create the bench directory structure**

```bash
mkdir -p bench/datasets bench/results bench/adapters bench/runners bench/judge
```

**Step 2: Write pyproject.toml**

Create `bench/pyproject.toml`:

```toml
[project]
name = "alaya-bench"
version = "0.1.0"
description = "Benchmark harness for AI agent memory systems"
requires-python = ">=3.11"
dependencies = [
    "litellm>=1.40",
    "chromadb>=0.5",
    "huggingface_hub>=0.20",
    "tqdm>=4.66",
    "click>=8.1",
]

[project.optional-dependencies]
mem0 = ["mem0ai>=1.0"]
zep = ["zep-cloud>=3.16"]
all = ["mem0ai>=1.0", "zep-cloud>=3.16"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[project.scripts]
alaya-bench = "run:cli"
```

**Step 3: Write .gitkeep files**

```bash
touch bench/datasets/.gitkeep bench/results/.gitkeep
```

**Step 4: Write README.md**

Create `bench/README.md`:

```markdown
# Alaya Benchmark Harness

Runs LoCoMo and LongMemEval benchmarks against multiple memory systems.

## Setup

```bash
cd bench
pip install -e ".[all,dev]"
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."        # Required for LLM calls
export JUDGE_MODEL="gpt-4o-mini"      # Judge model (default: gpt-4o-mini)
export LLM_MODEL="gpt-4o-mini"        # Answer generation model
export ZEP_API_KEY="..."              # Optional: for Zep adapter
```

## Usage

```bash
# Download datasets
python datasets/download.py

# Run LoCoMo on all systems
python run.py locomo --systems alaya,mem0,zep,fullcontext,naive_rag

# Run LongMemEval on specific systems
python run.py longmemeval --systems alaya,fullcontext --limit 50

# Dry run (estimate cost)
python run.py locomo --dry-run
```
```

**Step 5: Commit**

```bash
git add bench/
git commit -m "feat(bench): scaffold Python benchmark harness project"
```

---

### Task 2: Dataset Download Script

**Files:**
- Create: `bench/datasets/download.py`

**Step 1: Write the download script**

Create `bench/datasets/download.py`:

```python
"""Download LoCoMo and LongMemEval benchmark datasets."""

import json
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASETS_DIR = Path(__file__).parent


def download_locomo() -> Path:
    """Download LoCoMo dataset from GitHub (snap-research/locomo)."""
    dest = DATASETS_DIR / "locomo10.json"
    if dest.exists():
        print(f"LoCoMo already downloaded: {dest}")
        return dest
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    print(f"Downloading LoCoMo from {url} ...")
    urllib.request.urlretrieve(url, dest)
    with open(dest) as f:
        data = json.load(f)
    qa_count = sum(len(s["qa"]) for s in data)
    print(f"LoCoMo: {len(data)} conversations, {qa_count} QA pairs")
    return dest


def download_longmemeval_s() -> Path:
    """Download LongMemEval-S (115K tokens/question) from HuggingFace."""
    dest = DATASETS_DIR / "longmemeval_s.json"
    if dest.exists():
        print(f"LongMemEval-S already downloaded: {dest}")
        return dest
    print("Downloading LongMemEval-S from HuggingFace ...")
    path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename="longmemeval_s_cleaned.json",
        repo_type="dataset",
        local_dir=str(DATASETS_DIR),
    )
    # Rename to consistent name
    Path(path).rename(dest)
    with open(dest) as f:
        data = json.load(f)
    print(f"LongMemEval-S: {len(data)} questions")
    return dest


def download_longmemeval_oracle() -> Path:
    """Download LongMemEval-Oracle (small, evidence-only) from HuggingFace."""
    dest = DATASETS_DIR / "longmemeval_oracle.json"
    if dest.exists():
        print(f"LongMemEval-Oracle already downloaded: {dest}")
        return dest
    print("Downloading LongMemEval-Oracle from HuggingFace ...")
    path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename="longmemeval_oracle.json",
        repo_type="dataset",
        local_dir=str(DATASETS_DIR),
    )
    Path(path).rename(dest)
    with open(dest) as f:
        data = json.load(f)
    print(f"LongMemEval-Oracle: {len(data)} questions")
    return dest


if __name__ == "__main__":
    download_locomo()
    download_longmemeval_oracle()
    download_longmemeval_s()
    print("All datasets downloaded.")
```

**Step 2: Test the download script manually**

Run: `cd bench && python datasets/download.py`
Expected: Downloads 3 files to `bench/datasets/`, prints counts.

**Step 3: Add datasets to .gitignore**

Append to `bench/.gitignore`:

```
datasets/*.json
```

**Step 4: Commit**

```bash
git add bench/datasets/download.py bench/.gitignore
git commit -m "feat(bench): dataset download script for LoCoMo + LongMemEval"
```

---

### Task 3: MemoryAdapter Base Class

**Files:**
- Create: `bench/adapters/__init__.py`
- Create: `bench/adapters/base.py`
- Create: `bench/tests/test_adapters.py`

**Step 1: Write the failing test**

Create `bench/tests/__init__.py` (empty) and `bench/tests/test_adapters.py`:

```python
"""Tests for the MemoryAdapter interface."""

from adapters.base import MemoryAdapter, Message


def test_message_dataclass():
    msg = Message(role="user", content="hello", session_id="s1", timestamp="2024-01-01T00:00:00")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_adapter_cannot_be_instantiated():
    """ABC should not be directly instantiable."""
    import pytest
    with pytest.raises(TypeError):
        MemoryAdapter()
```

**Step 2: Run test to verify it fails**

Run: `cd bench && python -m pytest tests/test_adapters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adapters'`

**Step 3: Write the implementation**

Create `bench/adapters/__init__.py`:

```python
from adapters.base import MemoryAdapter, Message

__all__ = ["MemoryAdapter", "Message"]
```

Create `bench/adapters/base.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd bench && python -m pytest tests/test_adapters.py -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add bench/adapters/ bench/tests/
git commit -m "feat(bench): MemoryAdapter ABC and Message dataclass"
```

---

### Task 4: Full-Context Baseline Adapter

**Files:**
- Create: `bench/adapters/fullcontext.py`
- Modify: `bench/tests/test_adapters.py`

**Step 1: Write the failing test**

Append to `bench/tests/test_adapters.py`:

```python
from adapters.fullcontext import FullContextAdapter


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
```

**Step 2: Run test to verify it fails**

Run: `cd bench && python -m pytest tests/test_adapters.py::test_fullcontext_name -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `bench/adapters/fullcontext.py`:

```python
"""Full-context baseline: stuff entire conversation into the prompt."""

from adapters.base import MemoryAdapter, Message


class FullContextAdapter(MemoryAdapter):
    """Baseline that feeds the entire conversation history as context.

    No memory system involved — just concatenate all messages into the prompt.
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
```

**Step 4: Run test to verify it passes**

Run: `cd bench && python -m pytest tests/test_adapters.py -v`
Expected: 5 PASS

**Step 5: Commit**

```bash
git add bench/adapters/fullcontext.py bench/tests/test_adapters.py
git commit -m "feat(bench): full-context baseline adapter"
```

---

### Task 5: Naive RAG Baseline Adapter

**Files:**
- Create: `bench/adapters/naive_rag.py`
- Modify: `bench/tests/test_adapters.py`

**Step 1: Write the failing test**

Append to `bench/tests/test_adapters.py`:

```python
from adapters.naive_rag import NaiveRAGAdapter


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
```

**Step 2: Run test to verify it fails**

Run: `cd bench && python -m pytest tests/test_adapters.py::test_naive_rag_name -v`
Expected: FAIL

**Step 3: Write the implementation**

Create `bench/adapters/naive_rag.py`:

```python
"""Naive RAG baseline: ChromaDB vector similarity, no lifecycle."""

import uuid

import chromadb

from adapters.base import MemoryAdapter, Message


class NaiveRAGAdapter(MemoryAdapter):
    """Baseline that uses simple vector similarity retrieval.

    Messages are chunked and stored in ChromaDB. Queries retrieve
    top-k similar chunks by embedding distance. No forgetting,
    no consolidation, no graph — just vector search.
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
```

**Step 4: Run test to verify it passes**

Run: `cd bench && python -m pytest tests/test_adapters.py -v`
Expected: 7 PASS

**Step 5: Commit**

```bash
git add bench/adapters/naive_rag.py bench/tests/test_adapters.py
git commit -m "feat(bench): naive RAG baseline adapter with ChromaDB"
```

---

### Task 6: Mem0 Adapter

**Files:**
- Create: `bench/adapters/mem0_adapter.py`

**Step 1: Write the adapter**

Create `bench/adapters/mem0_adapter.py`:

```python
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
```

**Step 2: Commit**

```bash
git add bench/adapters/mem0_adapter.py
git commit -m "feat(bench): Mem0 adapter using mem0ai SDK"
```

---

### Task 7: Zep Adapter

**Files:**
- Create: `bench/adapters/zep_adapter.py`

**Step 1: Write the adapter**

Create `bench/adapters/zep_adapter.py`:

```python
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
```

**Step 2: Commit**

```bash
git add bench/adapters/zep_adapter.py
git commit -m "feat(bench): Zep adapter using zep-cloud SDK"
```

---

### Task 8: Alaya Subprocess Adapter

**Files:**
- Create: `bench/adapters/alaya.py`

**Step 1: Write the adapter**

Create `bench/adapters/alaya.py`:

```python
"""Alaya adapter via subprocess bridge.

Shells out to `alaya-bench` Rust CLI tool that communicates
via stdin/stdout JSON. This avoids PyO3/FFI complexity.

The Rust CLI is a separate binary in the alaya crate that
exposes ingest/query as subcommands. It must be built first:
    cd .. && cargo build --release --bin alaya-bench
"""

import json
import subprocess
import tempfile
from pathlib import Path

from adapters.base import MemoryAdapter, Message

ALAYA_BIN = Path(__file__).parent.parent.parent / "target" / "release" / "alaya-bench"


class AlayaAdapter(MemoryAdapter):
    """Adapter for Alaya memory system via subprocess.

    The alaya-bench binary accepts JSON on stdin and returns JSON on stdout.
    A temporary SQLite file is used for each conversation.
    """

    def __init__(self) -> None:
        self._db_path: Path | None = None
        self._tmpdir = tempfile.mkdtemp(prefix="alaya_bench_")
        if not ALAYA_BIN.exists():
            raise FileNotFoundError(
                f"alaya-bench binary not found at {ALAYA_BIN}. "
                "Build it with: cd .. && cargo build --release --bin alaya-bench"
            )

    def reset(self) -> None:
        self._db_path = Path(self._tmpdir) / f"bench.db"
        if self._db_path.exists():
            self._db_path.unlink()

    def _run(self, command: str, payload: dict) -> dict:
        """Run an alaya-bench subcommand with JSON I/O."""
        input_json = json.dumps(payload)
        result = subprocess.run(
            [str(ALAYA_BIN), command, "--db", str(self._db_path)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"alaya-bench {command} failed: {result.stderr}")
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def ingest(self, messages: list[Message]) -> None:
        payload = {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "session_id": m.session_id,
                    "timestamp": m.timestamp,
                }
                for m in messages
            ]
        }
        self._run("ingest", payload)

    def query(self, question: str, llm_call: callable) -> str:
        result = self._run("query", {"question": question})
        context = result.get("context", "")
        prompt = (
            f"Based on the following memories, answer the question.\n\n"
            f"Memories:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return llm_call(prompt)

    @property
    def name(self) -> str:
        return "Alaya"
```

**Step 2: Commit**

```bash
git add bench/adapters/alaya.py
git commit -m "feat(bench): Alaya subprocess adapter with JSON bridge"
```

---

### Task 9: LLM Client and Judge

**Files:**
- Create: `bench/judge/__init__.py`
- Create: `bench/judge/llm_judge.py`
- Create: `bench/tests/test_judge.py`

**Step 1: Write the failing test**

Create `bench/tests/test_judge.py`:

```python
"""Tests for LLM client and judge scoring."""

from judge.llm_judge import llm_call, score_answer


def test_llm_call_uses_mock():
    """Test that llm_call works with a mock backend."""
    # This test requires OPENAI_API_KEY, skip if not available
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        import pytest
        pytest.skip("OPENAI_API_KEY not set")
    result = llm_call("Say 'hello'", model="gpt-4o-mini", max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0


def test_score_answer_deterministic():
    """Test deterministic scoring for exact matches."""
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
```

**Step 2: Run test to verify it fails**

Run: `cd bench && python -m pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `bench/judge/__init__.py` (empty).

Create `bench/judge/llm_judge.py`:

```python
"""LLM client and judge scoring for benchmark evaluation."""

import os
from functools import partial

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


def make_llm_call(model: str | None = None) -> callable:
    """Create an llm_call partial with a specific model."""
    return partial(llm_call, model=model)


def score_answer(
    question: str,
    gold: str,
    prediction: str,
    judge_fn: callable | None = None,
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
```

**Step 4: Run test to verify it passes**

Run: `cd bench && python -m pytest tests/test_judge.py -v -k "not test_llm_call_uses_mock"`
Expected: 2 PASS (skip the live API test)

**Step 5: Commit**

```bash
git add bench/judge/ bench/tests/test_judge.py
git commit -m "feat(bench): LLM client and judge scorer with litellm"
```

---

### Task 10: LoCoMo Runner

**Files:**
- Create: `bench/runners/__init__.py`
- Create: `bench/runners/locomo.py`

**Step 1: Write the runner**

Create `bench/runners/__init__.py` (empty).

Create `bench/runners/locomo.py`:

```python
"""LoCoMo benchmark runner.

Loads the LoCoMo dataset, feeds conversations through adapters,
and scores answers using LLM-as-Judge.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from tqdm import tqdm

from adapters.base import MemoryAdapter, Message


@dataclass
class LoCoMoResult:
    system: str
    total_questions: int = 0
    correct: int = 0
    scores_by_category: dict[int, list[float]] = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_questions if self.total_questions else 0.0


def load_locomo(path: Path) -> list[dict]:
    """Load LoCoMo dataset from JSON."""
    with open(path) as f:
        return json.load(f)


def conversation_to_messages(conv: dict) -> list[Message]:
    """Convert a LoCoMo conversation to a list of Message objects."""
    messages = []
    session_idx = 1
    while f"session_{session_idx}" in conv["conversation"]:
        session_key = f"session_{session_idx}"
        timestamp = conv["conversation"].get(f"{session_key}_date_time", f"2024-01-{session_idx:02d}T00:00:00")
        for turn in conv["conversation"][session_key]:
            messages.append(Message(
                role="user" if turn["speaker"] == conv["conversation"]["speaker_a"] else "assistant",
                content=turn["text"],
                session_id=f"session_{session_idx}",
                timestamp=timestamp,
            ))
        session_idx += 1
    return messages


def run_locomo(
    adapter: MemoryAdapter,
    dataset_path: Path,
    judge_fn: callable,
    llm_call: callable,
    limit: int | None = None,
    dry_run: bool = False,
) -> LoCoMoResult:
    """Run LoCoMo benchmark on a single adapter.

    Args:
        adapter: The memory system adapter.
        dataset_path: Path to locomo10.json.
        judge_fn: Callable(question, gold, prediction) -> float.
        llm_call: Callable(prompt) -> str for answer generation.
        limit: Max questions to evaluate (None = all).
        dry_run: If True, count questions and estimate cost without running.
    """
    data = load_locomo(dataset_path)
    result = LoCoMoResult(system=adapter.name)

    # Collect all (conversation, question) pairs
    pairs = []
    for sample in data:
        messages = conversation_to_messages(sample)
        for qa in sample["qa"]:
            pairs.append((messages, qa))

    if limit:
        pairs = pairs[:limit]

    if dry_run:
        result.total_questions = len(pairs)
        avg_tokens = 20000  # approximate LoCoMo conversation length
        print(f"[DRY RUN] {adapter.name}: {len(pairs)} questions, "
              f"~{len(pairs) * avg_tokens:,} input tokens")
        return result

    start = time.time()
    for messages, qa in tqdm(pairs, desc=f"LoCoMo [{adapter.name}]"):
        adapter.reset()
        adapter.ingest(messages)
        prediction = adapter.query(qa["question"], llm_call)

        category = qa.get("category", 0)
        gold = qa.get("answer", "")

        # Skip adversarial (category 5) — no gold answer
        if category == 5:
            continue

        score = judge_fn(qa["question"], gold, prediction)

        result.total_questions += 1
        result.correct += int(score >= 0.5)
        result.scores_by_category.setdefault(category, []).append(score)
        result.per_question.append({
            "question": qa["question"],
            "gold": gold,
            "prediction": prediction,
            "score": score,
            "category": category,
        })

    result.elapsed_seconds = time.time() - start
    return result
```

**Step 2: Commit**

```bash
git add bench/runners/
git commit -m "feat(bench): LoCoMo benchmark runner"
```

---

### Task 11: LongMemEval Runner

**Files:**
- Create: `bench/runners/longmemeval.py`

**Step 1: Write the runner**

Create `bench/runners/longmemeval.py`:

```python
"""LongMemEval benchmark runner.

Loads the LongMemEval dataset, feeds conversation histories through
adapters, and scores answers using LLM-as-Judge.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from adapters.base import MemoryAdapter, Message


@dataclass
class LongMemEvalResult:
    system: str
    total_questions: int = 0
    correct: int = 0
    scores_by_type: dict[str, list[float]] = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_questions if self.total_questions else 0.0


def load_longmemeval(path: Path) -> list[dict]:
    """Load LongMemEval dataset from JSON."""
    with open(path) as f:
        return json.load(f)


def haystack_to_messages(item: dict) -> list[Message]:
    """Convert LongMemEval haystack sessions to Message objects."""
    messages = []
    dates = item.get("haystack_dates", [])
    sessions = item.get("haystack_sessions", [])
    for i, session in enumerate(sessions):
        session_id = item.get("haystack_session_ids", [f"s_{i}"])[i] if i < len(item.get("haystack_session_ids", [])) else f"s_{i}"
        date = dates[i] if i < len(dates) else f"2024-01-{i + 1:02d}"
        for turn in session:
            messages.append(Message(
                role=turn["role"],
                content=turn["content"],
                session_id=session_id,
                timestamp=f"{date}T00:00:00",
            ))
    return messages


def run_longmemeval(
    adapter: MemoryAdapter,
    dataset_path: Path,
    judge_fn: callable,
    llm_call: callable,
    limit: int | None = None,
    dry_run: bool = False,
) -> LongMemEvalResult:
    """Run LongMemEval benchmark on a single adapter.

    Args:
        adapter: The memory system adapter.
        dataset_path: Path to longmemeval_s.json or longmemeval_oracle.json.
        judge_fn: Callable(question, gold, prediction) -> float.
        llm_call: Callable(prompt) -> str for answer generation.
        limit: Max questions to evaluate (None = all).
        dry_run: If True, count questions and estimate cost without running.
    """
    data = load_longmemeval(dataset_path)
    result = LongMemEvalResult(system=adapter.name)

    if limit:
        data = data[:limit]

    if dry_run:
        result.total_questions = len(data)
        avg_tokens = 115000  # LongMemEval-S average
        print(f"[DRY RUN] {adapter.name}: {len(data)} questions, "
              f"~{len(data) * avg_tokens:,} input tokens")
        return result

    start = time.time()
    for item in tqdm(data, desc=f"LongMemEval [{adapter.name}]"):
        adapter.reset()
        messages = haystack_to_messages(item)
        adapter.ingest(messages)

        question = item["question"]
        gold = item["answer"]
        q_type = item.get("question_type", "unknown")

        prediction = adapter.query(question, llm_call)
        score = judge_fn(question, str(gold), prediction)

        result.total_questions += 1
        result.correct += int(score >= 0.5)
        result.scores_by_type.setdefault(q_type, []).append(score)
        result.per_question.append({
            "question_id": item.get("question_id", ""),
            "question": question,
            "gold": str(gold),
            "prediction": prediction,
            "score": score,
            "question_type": q_type,
        })

    result.elapsed_seconds = time.time() - start
    return result
```

**Step 2: Commit**

```bash
git add bench/runners/longmemeval.py
git commit -m "feat(bench): LongMemEval benchmark runner"
```

---

### Task 12: CLI Entrypoint

**Files:**
- Create: `bench/run.py`

**Step 1: Write the CLI**

Create `bench/run.py`:

```python
"""CLI entrypoint for the benchmark harness."""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click

from adapters.base import MemoryAdapter
from judge.llm_judge import make_llm_call, score_answer

RESULTS_DIR = Path(__file__).parent / "results"
DATASETS_DIR = Path(__file__).parent / "datasets"

ADAPTER_REGISTRY: dict[str, type] = {}


def _register_adapters() -> None:
    """Lazily import and register available adapters."""
    from adapters.fullcontext import FullContextAdapter
    from adapters.naive_rag import NaiveRAGAdapter

    ADAPTER_REGISTRY["fullcontext"] = FullContextAdapter
    ADAPTER_REGISTRY["naive_rag"] = NaiveRAGAdapter

    try:
        from adapters.mem0_adapter import Mem0Adapter
        ADAPTER_REGISTRY["mem0"] = Mem0Adapter
    except ImportError:
        pass

    try:
        from adapters.zep_adapter import ZepAdapter
        ADAPTER_REGISTRY["zep"] = ZepAdapter
    except ImportError:
        pass

    try:
        from adapters.alaya import AlayaAdapter
        ADAPTER_REGISTRY["alaya"] = AlayaAdapter
    except (ImportError, FileNotFoundError):
        pass


def _print_table(results: list[dict]) -> None:
    """Print results as a formatted table matching the paper."""
    print("\n" + "=" * 60)
    print(f"{'System':<20} {'Accuracy (%)':<15} {'Correct':<10} {'Total':<10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{r['system']:<20} {r['accuracy'] * 100:>10.2f}%    "
              f"{r['correct']:<10} {r['total']:<10}")
    print("=" * 60)


@click.command()
@click.argument("benchmark", type=click.Choice(["locomo", "longmemeval"]))
@click.option("--systems", "-s", default="fullcontext,naive_rag",
              help="Comma-separated adapter names")
@click.option("--limit", "-n", type=int, default=None,
              help="Limit number of questions")
@click.option("--dry-run", is_flag=True, help="Estimate cost without running")
@click.option("--dataset", type=click.Path(exists=True), default=None,
              help="Override dataset path")
def cli(benchmark: str, systems: str, limit: int | None, dry_run: bool, dataset: str | None):
    """Run memory system benchmarks.

    Examples:
        python run.py locomo --systems fullcontext,naive_rag --limit 10
        python run.py longmemeval --systems alaya,mem0 --dry-run
    """
    _register_adapters()

    system_names = [s.strip() for s in systems.split(",")]
    adapters: list[MemoryAdapter] = []
    for name in system_names:
        if name not in ADAPTER_REGISTRY:
            available = ", ".join(ADAPTER_REGISTRY.keys())
            click.echo(f"Unknown adapter: {name}. Available: {available}", err=True)
            sys.exit(1)
        adapters.append(ADAPTER_REGISTRY[name]())

    # Resolve dataset path
    if dataset:
        dataset_path = Path(dataset)
    elif benchmark == "locomo":
        dataset_path = DATASETS_DIR / "locomo10.json"
    else:
        dataset_path = DATASETS_DIR / "longmemeval_s.json"

    if not dataset_path.exists():
        click.echo(f"Dataset not found: {dataset_path}. Run: python datasets/download.py", err=True)
        sys.exit(1)

    llm_call_fn = make_llm_call()

    def judge_fn(question: str, gold: str, prediction: str) -> float:
        return score_answer(question, gold, prediction)

    all_results = []

    for adapter in adapters:
        click.echo(f"\nRunning {benchmark} on {adapter.name} ...")

        if benchmark == "locomo":
            from runners.locomo import run_locomo
            result = run_locomo(adapter, dataset_path, judge_fn, llm_call_fn,
                                limit=limit, dry_run=dry_run)
        else:
            from runners.longmemeval import run_longmemeval
            result = run_longmemeval(adapter, dataset_path, judge_fn, llm_call_fn,
                                     limit=limit, dry_run=dry_run)

        summary = {
            "system": result.system,
            "accuracy": result.accuracy,
            "correct": result.correct,
            "total": result.total_questions,
            "elapsed_seconds": result.elapsed_seconds,
        }
        all_results.append(summary)

        if not dry_run:
            # Save full results
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = RESULTS_DIR / f"{benchmark}_{adapter.name}_{ts}.json"
            with open(out_path, "w") as f:
                json.dump(asdict(result), f, indent=2)
            click.echo(f"Results saved: {out_path}")

    _print_table(all_results)


if __name__ == "__main__":
    cli()
```

**Step 2: Verify the CLI loads without errors**

Run: `cd bench && python run.py --help`
Expected: Shows usage with `locomo` and `longmemeval` commands.

**Step 3: Commit**

```bash
git add bench/run.py
git commit -m "feat(bench): CLI entrypoint with adapter registry and results table"
```

---

### Task 13: End-to-End Smoke Test

**Files:**
- Create: `bench/tests/test_e2e.py`

**Step 1: Write the smoke test**

Create `bench/tests/test_e2e.py`:

```python
"""End-to-end smoke test using full-context adapter with mock LLM."""

import json
import tempfile
from pathlib import Path

from adapters.base import Message
from adapters.fullcontext import FullContextAdapter
from runners.locomo import conversation_to_messages, run_locomo


MINI_LOCOMO = [
    {
        "sample_id": "test_1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"speaker": "Alice", "dia_id": "1_1", "text": "I just got a golden retriever puppy named Max."},
                {"speaker": "Bob", "dia_id": "1_2", "text": "That's adorable! How old is Max?"},
                {"speaker": "Alice", "dia_id": "1_3", "text": "He's 3 months old."},
            ],
            "session_1_date_time": "2024-01-15 10:30:00",
        },
        "qa": [
            {
                "question": "What is the name of Alice's puppy?",
                "answer": "Max",
                "category": 1,
                "evidence": ["1_1"],
            }
        ],
    }
]


def test_conversation_to_messages():
    msgs = conversation_to_messages(MINI_LOCOMO[0])
    assert len(msgs) == 3
    assert msgs[0].role == "user"
    assert msgs[0].content == "I just got a golden retriever puppy named Max."
    assert msgs[0].session_id == "session_1"


def test_e2e_locomo_with_mock():
    """Run LoCoMo on a tiny dataset with a mock LLM and judge."""
    # Write mini dataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_LOCOMO, f)
        dataset_path = Path(f.name)

    adapter = FullContextAdapter()

    def mock_llm(prompt: str) -> str:
        # The prompt should contain "Max" from the conversation
        if "Max" in prompt:
            return "The puppy's name is Max."
        return "I don't know."

    def mock_judge(question: str, gold: str, prediction: str) -> float:
        return 1.0 if gold.lower() in prediction.lower() else 0.0

    result = run_locomo(adapter, dataset_path, mock_judge, mock_llm)

    assert result.system == "Full-context"
    assert result.total_questions == 1
    assert result.correct == 1
    assert result.accuracy == 1.0

    dataset_path.unlink()


def test_dry_run():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_LOCOMO, f)
        dataset_path = Path(f.name)

    adapter = FullContextAdapter()
    result = run_locomo(adapter, dataset_path, lambda *a: 0.0, lambda p: "", dry_run=True)
    assert result.total_questions == 1
    assert result.correct == 0  # dry run doesn't score

    dataset_path.unlink()
```

**Step 2: Run the tests**

Run: `cd bench && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add bench/tests/test_e2e.py
git commit -m "test(bench): end-to-end smoke test with mock LLM and mini LoCoMo"
```
