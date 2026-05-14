from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def hash_input(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS silkloom_cache (
                    run_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, input_hash)
                )
                """
            )

    def get(self, run_id: str, input_hash: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM silkloom_cache WHERE run_id = ? AND input_hash = ?",
                (run_id, input_hash),
            ).fetchone()
            return row[0] if row else None

    def set(self, run_id: str, input_hash: str, result_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO silkloom_cache (run_id, input_hash, result_json)
                VALUES (?, ?, ?)
                """,
                (run_id, input_hash, result_json),
            )
