from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResultStore(Protocol):
    def get(self, namespace: str, fingerprint: str) -> str | None: ...

    def put(self, namespace: str, fingerprint: str, payload: str) -> None: ...


class SQLiteCheckpoint:
    def __init__(self, path: str | Path = ".silkloom.db"):
        self.path = Path(path)
        self._ensure_schema()

    def get(self, namespace: str, fingerprint: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM silkloom_results WHERE namespace = ? AND fingerprint = ?",
                (namespace, fingerprint),
            ).fetchone()
        return row[0] if row else None

    def put(self, namespace: str, fingerprint: str, payload: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO silkloom_results (namespace, fingerprint, payload)
                VALUES (?, ?, ?)
                """,
                (namespace, fingerprint, payload),
            )

    def _connect(self) -> sqlite3.Connection:
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS silkloom_results (
                    namespace TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, fingerprint)
                )
                """
            )


@dataclass(frozen=True)
class RunFingerprint:
    model: str
    prompt: str
    system: str | None
    output: Any
    params: Any

    def for_input(self, item: dict[str, Any]) -> str:
        return stable_hash(
            {
                "input": item,
                "model": self.model,
                "prompt": self.prompt,
                "system": self.system,
                "output": self.output,
                "params": self.params,
            }
        )
