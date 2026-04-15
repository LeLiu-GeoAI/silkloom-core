from __future__ import annotations

import json
import sqlite3
import traceback
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from time import perf_counter
from threading import Lock
from typing import Any, Callable, Optional

from tqdm import tqdm

from .nodes import BaseNode, CollectNode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self._node_lookup: dict[str, BaseNode] = {}
        self._node_dependencies: dict[str, list[str]] = {}
        self.collect_nodes: list[CollectNode] = []
        self._db_lock = Lock()
        self._progress_lock = Lock()
        self._progress_bar: Optional[tqdm] = None
        self._progress_total = 0
        self._progress_success = 0
        self._progress_failed = 0
        self._progress_callback: Optional[Callable[[dict[str, Any]], None]] = None
        self._active_run_id: Optional[str] = None
        self._init_db()

    def add_node(self, node: BaseNode, depends_on: list[str]) -> "Pipeline":
        if any(existing.name == node.name for existing in self.nodes):
            raise ValueError(f"Duplicate node name: {node.name}")

        for dependency in depends_on:
            if dependency not in self._node_lookup:
                raise ValueError(f"Dependency node not found for {node.name}: {dependency}")

        self.nodes.append(node)
        self._node_lookup[node.name] = node
        self._node_dependencies[node.name] = list(depends_on)
        self._validate_workflow_acyclic()
        return self

    def add_collect_node(
        self,
        name: str,
        func: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]],
        source_node: Optional[str] = None,
        include_failed: bool = False,
    ) -> "Pipeline":
        if any(existing.name == name for existing in self.collect_nodes):
            raise ValueError(f"Duplicate collect node name: {name}")
        if source_node is not None and source_node not in self._node_lookup:
            raise ValueError(f"Collect node source_node not found: {source_node}")
        self.collect_nodes.append(
            CollectNode(
                name=name,
                func=func,
                source_node=source_node,
                include_failed=include_failed,
            )
        )
        return self

    def run(
        self,
        inputs: list[dict[str, Any]],
        run_id: Optional[str] = None,
        show_workflow_prompt: bool = True,
        show_progress: bool = True,
        show_stage_prompt: bool = True,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        if not self.nodes:
            raise ValueError("Pipeline has no nodes. Add at least one node before run().")
        if not inputs:
            raise ValueError("inputs must not be empty.")

        started_at = perf_counter()
        resolved_run_id = run_id or str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._active_run_id = resolved_run_id
        try:
            if show_workflow_prompt:
                print(self._render_workflow_prompt(resolved_run_id, len(inputs)), flush=True)
            if show_stage_prompt:
                print("[SilkLoom] Stage 1/4: preparing run", flush=True)
            self._emit_progress_event(
                {
                    "event": "stage",
                    "run_id": resolved_run_id,
                    "stage": "prepare",
                    "step": 1,
                    "step_total": 4,
                    "inputs": len(inputs),
                }
            )
            self._ensure_run(resolved_run_id)
            self._prepare_tasks(resolved_run_id, len(inputs))
            self._set_run_status(resolved_run_id, "running")

            total_tasks = len(inputs) * len(self.nodes)
            status_counts = self._get_task_status_counts(resolved_run_id)
            self._reset_progress_state(
                total_tasks=total_tasks,
                success_count=status_counts["success"],
                failed_count=status_counts["failed"],
            )
            if show_progress:
                completed_initial = min(total_tasks, status_counts["success"] + status_counts["failed"])
                self._open_progress_bar(total_tasks=total_tasks, completed_initial=completed_initial)

            try:
                if show_stage_prompt:
                    print("[SilkLoom] Stage 2/4: executing workflow nodes", flush=True)
                self._emit_progress_event(
                    {
                        "event": "stage",
                        "run_id": resolved_run_id,
                        "stage": "execute_nodes",
                        "step": 2,
                        "step_total": 4,
                    }
                )
                if self.execution_mode == "depth_first":
                    self._run_depth_first(resolved_run_id, inputs)
                else:
                    self._run_breadth_first(resolved_run_id, inputs)

                if show_stage_prompt and self.collect_nodes:
                    print("[SilkLoom] Stage 3/4: running collect nodes", flush=True)
                self._emit_progress_event(
                    {
                        "event": "stage",
                        "run_id": resolved_run_id,
                        "stage": "collect",
                        "step": 3,
                        "step_total": 4,
                        "collect_nodes": len(self.collect_nodes),
                    }
                )
                self._run_collect_nodes(resolved_run_id, inputs)
            except Exception:
                self._set_run_status(resolved_run_id, "paused")
                raise
            finally:
                self._close_progress_bar()

            if show_stage_prompt:
                print("[SilkLoom] Stage 4/4: finalizing run status", flush=True)
            self._emit_progress_event(
                {
                    "event": "stage",
                    "run_id": resolved_run_id,
                    "stage": "finalize",
                    "step": 4,
                    "step_total": 4,
                }
            )
            self._finalize_run_status(resolved_run_id)
            final_status = self._get_run_status(resolved_run_id)
            latest_counts = self._get_task_status_counts(resolved_run_id)
            elapsed_seconds = perf_counter() - started_at
            if show_stage_prompt:
                print(
                    self._render_run_summary(
                        run_id=resolved_run_id,
                        final_status=final_status,
                        total_tasks=total_tasks,
                        success_count=latest_counts["success"],
                        failed_count=latest_counts["failed"],
                        elapsed_seconds=elapsed_seconds,
                    ),
                    flush=True,
                )
            self._emit_progress_event(
                {
                    "event": "run_finished",
                    "run_id": resolved_run_id,
                    "status": final_status,
                    "total_tasks": total_tasks,
                    "success": latest_counts["success"],
                    "failed": latest_counts["failed"],
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            return resolved_run_id
        finally:
            self._active_run_id = None
            self._progress_callback = None

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

    def get_run_artifacts(self, run_id: str) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT artifact_name, payload
                FROM pipeline_run_artifacts
                WHERE run_id = ?
                ORDER BY artifact_name
                """,
                (run_id,),
            ).fetchall()
        artifacts: dict[str, dict[str, Any]] = {}
        for row in rows:
            artifacts[str(row["artifact_name"])] = json.loads(row["payload"]) if row["payload"] else {}
        return artifacts

    def describe_workflow(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "nodes": [
                {
                    "name": node.name,
                    "depends_on": list(self._node_dependencies.get(node.name, [])),
                }
                for node in self.nodes
            ],
            "collect_nodes": [
                {
                    "name": node.name,
                    "source_node": node.source_node,
                    "include_failed": node.include_failed,
                }
                for node in self.collect_nodes
            ],
        }

    def _render_workflow_prompt(self, run_id: str, input_count: int) -> str:
        lines = [
            "=== SilkLoom Workflow ===",
            f"run_id: {run_id}",
            f"mode: {self.execution_mode}",
            f"inputs: {input_count}",
            "nodes:",
        ]

        for index, node in enumerate(self.nodes, start=1):
            depends_on = self._node_dependencies.get(node.name, [])
            dependency_text = ", ".join(depends_on) if depends_on else "<input>"
            lines.append(f"  {index}. {node.name} <- {dependency_text}")

        if self.collect_nodes:
            lines.append("collect:")
            for collect_node in self.collect_nodes:
                source_text = collect_node.source_node or "<all_nodes>"
                lines.append(f"  - {collect_node.name} <- {source_text}")

        return "\n".join(lines)

    def _run_depth_first(self, run_id: str, inputs: list[dict[str, Any]]) -> None:
        ordered_node_names = [node.name for node in self._topological_nodes()]
        with ThreadPoolExecutor(max_workers=self.default_workers) as executor:
            futures = [
                executor.submit(
                    self._process_item_depth_first_names,
                    run_id,
                    item_index,
                    item_input,
                    ordered_node_names,
                )
                for item_index, item_input in enumerate(inputs)
            ]
            for future in as_completed(futures):
                future.result()

    def _run_breadth_first(self, run_id: str, inputs: list[dict[str, Any]]) -> None:
        levels = [[node.name for node in level] for level in self._topological_levels()]
        for level in levels:
            with ThreadPoolExecutor(max_workers=self.default_workers) as executor:
                futures = [
                    executor.submit(
                        self._process_single_node,
                        run_id,
                        item_index,
                        item_input,
                        node_name,
                    )
                    for node_name in level
                    for item_index, item_input in enumerate(inputs)
                ]
                for future in as_completed(futures):
                    future.result()

    def _process_item_depth_first_names(
        self,
        run_id: str,
        item_index: int,
        item_input: dict[str, Any],
        ordered_node_names: list[str],
    ) -> None:
        for node_name in ordered_node_names:
            self._process_single_node(
                run_id=run_id,
                item_index=item_index,
                item_input=item_input,
                node_name=node_name,
            )

    def _process_single_node(
        self,
        run_id: str,
        item_index: int,
        item_input: dict[str, Any],
        node_name: str,
    ) -> None:
        node = self._node_lookup[node_name]
        task = self._get_task(run_id, item_index, node.name)
        if task is None:
            raise RuntimeError(
                f"Task not found: run_id={run_id}, item={item_index}, node={node.name}"
            )

        previous_status = str(task["status"])

        if previous_status == "success":
            return

        if previous_status == "failed" and int(task["retries"]) >= node.max_retries:
            return

        context = self._build_context_for_named_node(
            run_id=run_id,
            item_index=item_index,
            item_input=item_input,
            node_name=node.name,
        )
        if context is None:
            return

        self._execute_node(run_id, item_index, node, context, previous_status=previous_status)

    def _execute_node(
        self,
        run_id: str,
        item_index: int,
        node: BaseNode,
        context: dict[str, Any],
        previous_status: str,
    ) -> Optional[dict[str, Any]]:
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
            self._on_task_settled(previous_status=previous_status, new_status="success", node_name=node.name)
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
            self._on_task_settled(previous_status=previous_status, new_status="failed", node_name=node.name)
            return None

    def _build_context_for_named_node(
        self,
        run_id: str,
        item_index: int,
        item_input: dict[str, Any],
        node_name: str,
    ) -> Optional[dict[str, Any]]:
        dependencies = self._node_dependencies.get(node_name, [])
        if not dependencies:
            return {"input": item_input}

        context: dict[str, Any] = {"input": item_input}
        for dependency in dependencies:
            dependency_task = self._get_task(run_id, item_index, dependency)
            if dependency_task is None:
                return None
            if dependency_task["status"] != "success":
                return None
            dependency_context = self._context_from_task(dependency_task, {"input": item_input})
            for key, value in dependency_context.items():
                if key == "input":
                    continue
                context[key] = value
        return context

    def _context_from_task(
        self,
        task: sqlite3.Row,
        fallback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if task["context_data"]:
            return json.loads(task["context_data"])
        return fallback or {"input": {}}

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_run_artifacts (
                    run_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, artifact_name),
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
                )
                """
            )

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

    def _get_run_status(self, run_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM pipeline_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return "unknown"
        return str(row[0])

    def _get_task_status_counts(self, run_id: str) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM pipeline_tasks
                WHERE run_id = ?
                GROUP BY status
                """,
                (run_id,),
            ).fetchall()

        counts = {"pending": 0, "running": 0, "success": 0, "failed": 0}
        for status, count in rows:
            key = str(status)
            if key in counts:
                counts[key] = int(count)
        return counts

    def _reset_progress_state(self, total_tasks: int, success_count: int, failed_count: int) -> None:
        with self._progress_lock:
            self._progress_total = total_tasks
            self._progress_success = max(0, success_count)
            self._progress_failed = max(0, failed_count)

    def _open_progress_bar(self, total_tasks: int, completed_initial: int) -> None:
        with self._progress_lock:
            self._progress_bar = tqdm(
                total=total_tasks,
                initial=max(0, min(completed_initial, total_tasks)),
                desc="SilkLoom",
                unit="task",
                dynamic_ncols=True,
                leave=True,
                ascii=True,
            )
            self._progress_bar.set_postfix(
                success=self._progress_success,
                failed=self._progress_failed,
            )

    def _close_progress_bar(self) -> None:
        with self._progress_lock:
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None

    def _on_task_settled(self, previous_status: str, new_status: str, node_name: str) -> None:
        terminal_status = {"success", "failed"}
        with self._progress_lock:
            if previous_status in terminal_status:
                if previous_status == "success":
                    self._progress_success = max(0, self._progress_success - 1)
                elif previous_status == "failed":
                    self._progress_failed = max(0, self._progress_failed - 1)

            if new_status == "success":
                self._progress_success += 1
            elif new_status == "failed":
                self._progress_failed += 1

            if self._progress_bar is not None:
                if previous_status not in terminal_status and new_status in terminal_status:
                    self._progress_bar.update(1)
                self._progress_bar.set_postfix(
                    success=self._progress_success,
                    failed=self._progress_failed,
                    node=node_name,
                )
            completed = self._progress_success + self._progress_failed
            event_payload = {
                "event": "task_settled",
                "run_id": self._active_run_id,
                "node": node_name,
                "status": new_status,
                "success": self._progress_success,
                "failed": self._progress_failed,
                "completed": completed,
                "total": self._progress_total,
            }
        self._emit_progress_event(event_payload)

    def _emit_progress_event(self, event: dict[str, Any]) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            # Progress callback should never break pipeline execution.
            return

    def _render_run_summary(
        self,
        run_id: str,
        final_status: str,
        total_tasks: int,
        success_count: int,
        failed_count: int,
        elapsed_seconds: float,
    ) -> str:
        return (
            "[SilkLoom] Run finished "
            f"run_id={run_id} "
            f"status={final_status} "
            f"success={success_count}/{total_tasks} "
            f"failed={failed_count} "
            f"elapsed={elapsed_seconds:.2f}s"
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

    def _run_collect_nodes(self, run_id: str, inputs: list[dict[str, Any]]) -> None:
        if not self.collect_nodes:
            return

        results = self.export_results(run_id)
        for collect_node in self.collect_nodes:
            items_for_collect: list[dict[str, Any]] = []
            for item in results:
                status = str(item.get("status", "pending"))
                if status != "success" and not collect_node.include_failed:
                    continue

                context = item.get("context", {})
                if not isinstance(context, dict):
                    continue

                value: Any = context
                if collect_node.source_node is not None:
                    if collect_node.source_node not in context:
                        continue
                    value = context[collect_node.source_node]

                items_for_collect.append(
                    {
                        "item_index": item.get("item_index"),
                        "status": status,
                        "value": value,
                        "context": context,
                    }
                )

            metadata = {
                "run_id": run_id,
                "input_count": len(inputs),
                "included_items": len(items_for_collect),
                "source_node": collect_node.source_node,
            }
            payload = collect_node.func(items_for_collect, metadata)
            if not isinstance(payload, dict):
                raise TypeError(f"Collect node {collect_node.name} must return a dict.")
            self._upsert_run_artifact(run_id, collect_node.name, payload)

    def _upsert_run_artifact(self, run_id: str, artifact_name: str, payload: dict[str, Any]) -> None:
        now = _utc_now()
        serialized = json.dumps(payload, ensure_ascii=False)
        with self._db_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_run_artifacts(run_id, artifact_name, payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, artifact_name)
                    DO UPDATE SET payload = excluded.payload,
                                  updated_at = excluded.updated_at
                    """,
                    (run_id, artifact_name, serialized, now, now),
                )

    def _validate_workflow_acyclic(self) -> None:
        indegree, followers = self._build_graph_state()

        queue = deque(name for name, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for nxt in followers.get(current, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if visited != len(self.nodes):
            raise ValueError("Workflow dependencies contain a cycle.")

    def _topological_nodes(self) -> list[BaseNode]:
        indegree, followers = self._build_graph_state()

        queue = deque(name for name, degree in indegree.items() if degree == 0)
        ordered_names: list[str] = []
        while queue:
            current = queue.popleft()
            ordered_names.append(current)
            for nxt in followers.get(current, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(ordered_names) != len(self.nodes):
            raise ValueError("Workflow dependencies contain a cycle.")

        return [self._node_lookup[name] for name in ordered_names]

    def _topological_levels(self) -> list[list[BaseNode]]:
        indegree, followers = self._build_graph_state()

        current_level = [name for name, degree in indegree.items() if degree == 0]
        levels: list[list[BaseNode]] = []
        visited = 0

        while current_level:
            current_level_nodes = [self._node_lookup[name] for name in current_level]
            levels.append(current_level_nodes)
            visited += len(current_level)

            next_level: list[str] = []
            for node_name in current_level:
                for follower in followers.get(node_name, []):
                    indegree[follower] -= 1
                    if indegree[follower] == 0:
                        next_level.append(follower)
            current_level = next_level

        if visited != len(self.nodes):
            raise ValueError("Workflow dependencies contain a cycle.")

        return levels

    def _build_graph_state(self) -> tuple[dict[str, int], dict[str, list[str]]]:
        indegree: dict[str, int] = {node.name: 0 for node in self.nodes}
        followers: dict[str, list[str]] = defaultdict(list)
        for node_name, dependencies in self._node_dependencies.items():
            for dependency in dependencies:
                followers[dependency].append(node_name)
                indegree[node_name] += 1
        return indegree, followers
