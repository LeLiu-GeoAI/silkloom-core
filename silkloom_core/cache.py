from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteCheckpoint:
    def __init__(self, path: str | Path = ".silkloom.db"):
        self.path = Path(path)
        self._ensure_schema()

    def get(self, namespace: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM silkloom_results WHERE namespace = ? AND cache_key = ?",
                (namespace, key),
            ).fetchone()
        return row[0] if row else None

    def set(self, namespace: str, key: str, payload: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO silkloom_results (namespace, cache_key, payload)
                VALUES (?, ?, ?)
                """,
                (namespace, key, payload),
            )

    def _connect(self) -> sqlite3.Connection:
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS silkloom_results (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )


SQLiteCache = SQLiteCheckpoint
hash_input = stable_hash
