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
        self._db_path = Path(self._tmpdir) / "bench.db"
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
