from __future__ import annotations

import json
import re
import sqlite3
import traceback
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock, Semaphore
from typing import Any, Callable, Optional, Type

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from .utils import render_template, resolve_mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class NodeResult:
    output: dict[str, Any]
    retries_used: int


class BaseNode(ABC):
    def __init__(self, name: str, max_retries: int = 3, max_workers: Optional[int] = None) -> None:
        if not name:
            raise ValueError("Node name must not be empty.")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1 when provided.")
        self.name = name
        self.max_retries = max_retries
        self.max_workers = max_workers

    @abstractmethod
    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run_with_retry(self, context: dict[str, Any]) -> NodeResult:
        last_attempt = 1
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=8),
            retry=retry_if_exception(self._is_retryable_error),
            reraise=True,
        ):
            with attempt:
                last_attempt = attempt.retry_state.attempt_number
                output = self.process(context)
                if not isinstance(output, dict):
                    raise TypeError(f"Node {self.name} must return a dict.")
                return NodeResult(output=output, retries_used=max(0, last_attempt - 1))

        raise RuntimeError(f"Unexpected retry loop exit for node {self.name}")

    def _is_retryable_error(self, exc: BaseException) -> bool:
        return True


class LLMNode(BaseNode):
    def __init__(
        self,
        name: str,
        prompt_template: str,
        model: str = "gpt-4o-mini",
        response_model: Optional[Type[BaseModel]] = None,
        max_retries: int = 3,
        max_workers: Optional[int] = None,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(name=name, max_retries=max_retries, max_workers=max_workers)
        self.prompt_template = prompt_template
        self.model = model
        self.response_model = response_model
        self.client = client or OpenAI()

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        prompt = render_template(self.prompt_template, context)
        messages = [{"role": "user", "content": prompt}]

        if self.response_model is not None:
            try:
                parsed = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=self.response_model,
                )
                payload = parsed.choices[0].message.parsed
                if payload is None:
                    raise ValueError("LLM structured output parse returned None.")
                if isinstance(payload, BaseModel):
                    return payload.model_dump()
                if isinstance(payload, dict):
                    return payload
                raise TypeError("LLM structured output must be BaseModel or dict.")
            except Exception:
                expected_fields = ", ".join(self.response_model.model_fields.keys())
                fallback_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You must return valid JSON only. Do not include markdown fences or explanations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n\n"
                            f"Return exactly one JSON object with these keys: {expected_fields}."
                        ),
                    },
                ]
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=fallback_messages,
                )
                raw_text = self._extract_text_content(completion.choices[0].message.content)
                try:
                    return self._validate_structured_fallback(raw_text)
                except Exception as validate_exc:
                    raise RuntimeError(
                        "Structured output failed for both parse() and JSON fallback."
                    ) from validate_exc

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        text = self._extract_text_content(completion.choices[0].message.content)
        return {"text": text}

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return content or ""

    def _validate_structured_fallback(self, raw_text: str) -> dict[str, Any]:
        cleaned = self._sanitize_json_text(raw_text)
        payload = self._extract_json_object(cleaned)
        payload = self._normalize_common_aliases(payload)
        return self.response_model.model_validate(payload).model_dump()

    def _sanitize_json_text(self, text: str) -> str:
        value = text.strip().lstrip("\ufeff")
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", value, flags=re.IGNORECASE)
        if fenced:
            value = fenced[0].strip()
        if value.lower().startswith("json\n"):
            value = value[5:].strip()
        return value

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(text[start : end + 1])

        if not isinstance(data, dict):
            raise TypeError("Structured fallback JSON must be an object.")
        return data

    def _normalize_common_aliases(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        expected = set(self.response_model.model_fields.keys())
        aliases = {
            "city_name": "city",
            "location": "city",
            "purpose": "intent",
            "user_intent": "intent",
        }
        for source_key, target_key in aliases.items():
            if source_key in normalized and target_key in expected and target_key not in normalized:
                normalized[target_key] = normalized[source_key]
        return normalized

    def _is_retryable_error(self, exc: BaseException) -> bool:
        return isinstance(exc, (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError))


class FunctionNode(BaseNode):
    def __init__(
        self,
        name: str,
        func: Callable[..., dict[str, Any]],
        kwargs_mapping: Optional[dict[str, str]] = None,
        max_retries: int = 3,
        max_workers: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, max_retries=max_retries, max_workers=max_workers)
        self.func = func
        self.kwargs_mapping = kwargs_mapping or {}

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        kwargs = resolve_mapping(self.kwargs_mapping, context)
        result = self.func(**kwargs)
        if not isinstance(result, dict):
            raise TypeError(f"FunctionNode {self.name} function must return a dict.")
        return result


class Pipeline:
    def __init__(self, db_path: str, execution_mode: str = "depth_first", default_workers: int = 5) -> None:
        if execution_mode not in {"depth_first", "breadth_first"}:
            raise ValueError("execution_mode must be 'depth_first' or 'breadth_first'.")
        if default_workers < 1:
            raise ValueError("default_workers must be >= 1.")
        self.db_path = db_path
        self.execution_mode = execution_mode
        self.default_workers = default_workers
        self.nodes: list[BaseNode] = []
        self._db_lock = Lock()
        self._node_locks: dict[str, Semaphore] = {}
        self._init_db()

    def add_node(self, node: BaseNode) -> "Pipeline":
        if any(existing.name == node.name for existing in self.nodes):
            raise ValueError(f"Duplicate node name: {node.name}")
        self.nodes.append(node)
        return self

    def run(self, inputs: list[dict[str, Any]], run_id: str = None) -> str:
        if not self.nodes:
            raise ValueError("Pipeline has no nodes. Add at least one node before run().")
        if not inputs:
            raise ValueError("inputs must not be empty.")

        resolved_run_id = run_id or str(uuid.uuid4())
        self._ensure_run(resolved_run_id)
        self._prepare_tasks(resolved_run_id, len(inputs))
        self._set_run_status(resolved_run_id, "running")

        try:
            if self.execution_mode == "depth_first":
                self._run_depth_first(resolved_run_id, inputs)
            else:
                self._run_breadth_first(resolved_run_id, inputs)
        except Exception:
            self._set_run_status(resolved_run_id, "paused")
            raise

        self._finalize_run_status(resolved_run_id)
        return resolved_run_id

    def export_results(self, run_id: str, format: str = "json") -> list[dict[str, Any]]:
        if format != "json":
            raise ValueError("Only format='json' is currently supported.")

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT item_index, node_name, status, context_data, error_msg, retries, updated_at
                FROM pipeline_tasks
                WHERE run_id = ?
                ORDER BY item_index, updated_at
                """,
                (run_id,),
            ).fetchall()

        grouped: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(int(row["item_index"]), []).append(row)

        node_order = {node.name: index for index, node in enumerate(self.nodes)}
        results: list[dict[str, Any]] = []
        for item_index, item_rows in sorted(grouped.items(), key=lambda x: x[0]):
            by_node = {r["node_name"]: r for r in item_rows}
            failed_rows = [r for r in item_rows if r["status"] == "failed"]
            success_rows = [r for r in item_rows if r["status"] == "success"]

            if failed_rows:
                final_row = sorted(failed_rows, key=lambda r: r["updated_at"])[-1]
                item_status = "failed"
            elif len(success_rows) == len(self.nodes):
                final_row = by_node[self.nodes[-1].name]
                item_status = "success"
            elif success_rows:
                final_row = sorted(success_rows, key=lambda r: node_order.get(r["node_name"], -1))[-1]
                item_status = "incomplete"
            else:
                final_row = item_rows[0]
                item_status = "pending"

            context = json.loads(final_row["context_data"]) if final_row["context_data"] else {}
            errors = [
                {
                    "node_name": r["node_name"],
                    "error_msg": r["error_msg"],
                    "retries": r["retries"],
                }
                for r in item_rows
                if r["status"] == "failed"
            ]
            results.append(
                {
                    "item_index": item_index,
                    "status": item_status,
                    "last_node": final_row["node_name"],
                    "context": context,
                    "errors": errors,
                }
            )

        return results

    def _run_depth_first(self, run_id: str, inputs: list[dict[str, Any]]) -> None:
        with ThreadPoolExecutor(max_workers=self.default_workers) as executor:
            futures = [
                executor.submit(self._process_item_depth_first, run_id, item_index, item_input)
                for item_index, item_input in enumerate(inputs)
            ]
            for future in as_completed(futures):
                future.result()

    def _run_breadth_first(self, run_id: str, inputs: list[dict[str, Any]]) -> None:
        for node_index, node in enumerate(self.nodes):
            node_workers = node.max_workers or self.default_workers
            with ThreadPoolExecutor(max_workers=node_workers) as executor:
                futures = {}
                for item_index, item_input in enumerate(inputs):
                    task = self._get_task(run_id, item_index, node.name)
                    if task is None:
                        raise RuntimeError(
                            f"Task not found: run_id={run_id}, item={item_index}, node={node.name}"
                        )

                    if task["status"] == "success":
                        continue

                    if task["status"] == "failed" and int(task["retries"]) >= node.max_retries:
                        continue

                    context = self._build_context_for_node(
                        run_id=run_id,
                        item_index=item_index,
                        item_input=item_input,
                        node_index=node_index,
                    )
                    if context is None:
                        continue

                    futures[executor.submit(self._execute_node, run_id, item_index, node, context)] = item_index

                for future in as_completed(futures):
                    future.result()

    def _process_item_depth_first(self, run_id: str, item_index: int, item_input: dict[str, Any]) -> None:
        context: Optional[dict[str, Any]] = {"input": item_input}

        for node_index, node in enumerate(self.nodes):
            task = self._get_task(run_id, item_index, node.name)
            if task is None:
                raise RuntimeError(
                    f"Task not found: run_id={run_id}, item={item_index}, node={node.name}"
                )

            if task["status"] == "success":
                context = self._context_from_task(task, context)
                continue

            if task["status"] == "failed" and int(task["retries"]) >= node.max_retries:
                return

            context = self._build_context_for_node(
                run_id=run_id,
                item_index=item_index,
                item_input=item_input,
                node_index=node_index,
                current_context=context,
            )
            if context is None:
                return

            new_context = self._execute_node(run_id, item_index, node, context)
            if new_context is None:
                return
            context = new_context

    def _execute_node(
        self,
        run_id: str,
        item_index: int,
        node: BaseNode,
        context: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        semaphore = self._get_node_semaphore(node)
        with semaphore:
            self._update_task(
                run_id=run_id,
                item_index=item_index,
                node_name=node.name,
                status="running",
                context_data=context,
                error_msg=None,
                retries=self._get_task_retries(run_id, item_index, node.name),
            )

            try:
                result = node.run_with_retry(context)
                new_context = dict(context)
                new_context[node.name] = result.output
                self._update_task(
                    run_id=run_id,
                    item_index=item_index,
                    node_name=node.name,
                    status="success",
                    context_data=new_context,
                    error_msg=None,
                    retries=result.retries_used,
                )
                return new_context
            except Exception:
                error_text = traceback.format_exc()
                self._update_task(
                    run_id=run_id,
                    item_index=item_index,
                    node_name=node.name,
                    status="failed",
                    context_data=context,
                    error_msg=error_text,
                    retries=node.max_retries,
                )
                return None

    def _build_context_for_node(
        self,
        run_id: str,
        item_index: int,
        item_input: dict[str, Any],
        node_index: int,
        current_context: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if current_context is not None:
            return current_context
        if node_index == 0:
            return {"input": item_input}

        previous_node = self.nodes[node_index - 1]
        previous_task = self._get_task(run_id, item_index, previous_node.name)
        if previous_task is None:
            return None
        if previous_task["status"] != "success":
            return None
        return self._context_from_task(previous_task, {"input": item_input})

    def _context_from_task(
        self,
        task: sqlite3.Row,
        fallback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if task["context_data"]:
            return json.loads(task["context_data"])
        return fallback or {"input": {}}

    def _get_node_semaphore(self, node: BaseNode) -> Semaphore:
        limit = node.max_workers or self.default_workers
        if node.name not in self._node_locks:
            self._node_locks[node.name] = Semaphore(limit)
        return self._node_locks[node.name]

    def _get_task_retries(self, run_id: str, item_index: int, node_name: str) -> int:
        task = self._get_task(run_id, item_index, node_name)
        return int(task["retries"]) if task is not None else 0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    execution_mode TEXT NOT NULL DEFAULT 'depth_first',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    context_data TEXT,
                    error_msg TEXT,
                    retries INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(run_id, item_index, node_name),
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
                )
                """
            )
            self._ensure_runs_schema(conn)

    def _ensure_runs_schema(self, conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()}
        if "execution_mode" not in columns:
            conn.execute("ALTER TABLE pipeline_runs ADD COLUMN execution_mode TEXT NOT NULL DEFAULT 'depth_first'")

    def _ensure_run(self, run_id: str) -> None:
        with self._db_lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT run_id FROM pipeline_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO pipeline_runs(run_id, status, execution_mode, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run_id, "running", self.execution_mode, _utc_now()),
                    )
                else:
                    conn.execute(
                        "UPDATE pipeline_runs SET execution_mode = ? WHERE run_id = ?",
                        (self.execution_mode, run_id),
                    )

    def _set_run_status(self, run_id: str, status: str) -> None:
        with self._db_lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status = ? WHERE run_id = ?",
                    (status, run_id),
                )

    def _prepare_tasks(self, run_id: str, item_count: int) -> None:
        with self._db_lock:
            with self._connect() as conn:
                for item_index in range(item_count):
                    for node in self.nodes:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO pipeline_tasks(
                                task_id, run_id, item_index, node_name,
                                status, context_data, error_msg, retries, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(uuid.uuid4()),
                                run_id,
                                item_index,
                                node.name,
                                "pending",
                                None,
                                None,
                                0,
                                _utc_now(),
                            ),
                        )

    def _get_task(self, run_id: str, item_index: int, node_name: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                """
                SELECT task_id, status, context_data, error_msg, retries
                FROM pipeline_tasks
                WHERE run_id = ? AND item_index = ? AND node_name = ?
                """,
                (run_id, item_index, node_name),
            ).fetchone()

    def _update_task(
        self,
        run_id: str,
        item_index: int,
        node_name: str,
        status: str,
        context_data: dict[str, Any],
        error_msg: Optional[str],
        retries: int,
    ) -> None:
        payload = json.dumps(context_data, ensure_ascii=False)
        with self._db_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE pipeline_tasks
                    SET status = ?, context_data = ?, error_msg = ?, retries = ?, updated_at = ?
                    WHERE run_id = ? AND item_index = ? AND node_name = ?
                    """,
                    (
                        status,
                        payload,
                        error_msg,
                        retries,
                        _utc_now(),
                        run_id,
                        item_index,
                        node_name,
                    ),
                )

    def _finalize_run_status(self, run_id: str) -> None:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM pipeline_tasks WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM pipeline_tasks WHERE run_id = ? AND status = 'failed'",
                (run_id,),
            ).fetchone()[0]
            succeeded = conn.execute(
                "SELECT COUNT(*) FROM pipeline_tasks WHERE run_id = ? AND status = 'success'",
                (run_id,),
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM pipeline_tasks WHERE run_id = ? AND status IN ('pending', 'running')",
                (run_id,),
            ).fetchone()[0]

        if pending > 0:
            status = "paused"
        elif failed == 0 and succeeded == total:
            status = "completed"
        elif failed > 0:
            status = "partial_failed"
        else:
            status = "paused"

        self._set_run_status(run_id, status)
