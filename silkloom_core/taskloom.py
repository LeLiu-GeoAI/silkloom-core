from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .checkpoint import ResultStore, RunFingerprint, SQLiteCheckpoint, stable_hash
from .clients import ChatClient, OpenAIChat, OpenAICompatible
from .json_utils import extract_reasoning, parse_json_payload
from .message_builder import MessageBuilder


STATUS_COLUMNS = ("_loom_ok", "_loom_error", "_loom_checkpoint", "_loom_attempts")
OPTIONAL_STATUS_COLUMNS = ("_loom_output", "_loom_reasoning")


class Loom:
    def __init__(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        client: ChatClient | Any | None = None,
        checkpoint: str | Path | ResultStore | None = ".silkloom.db",
        retries: int = 2,
        repair_json: bool = True,
        **params: Any,
    ):
        self.model = model
        self.prompt = prompt
        self.system = system
        self.params = params
        self.retries = max(0, retries)
        self.repair_json = repair_json
        self.messages = MessageBuilder(prompt, system)
        self.client: ChatClient = self._client(client)
        self.checkpoint = self._checkpoint(checkpoint)

    def __call__(
        self,
        data: Any,
        *,
        input: str | Sequence[str],
        images: str | Sequence[str] | None = None,
        resume: str | None = None,
        concurrency: int = 5,
        prefix: str | None = None,
        status: bool = True,
        include_output: bool = False,
        include_reasoning: bool = False,
    ):
        pd = self._pandas()
        frame = self._frame(data)
        input_columns = [input] if isinstance(input, str) else list(input)
        image_columns = [] if images is None else ([images] if isinstance(images, str) else list(images))
        self._require_columns(frame, input_columns + image_columns)

        records = [self._record(row, input_columns, image_columns) for _, row in frame.iterrows()]
        context = self._run_context(
            resume=resume,
            input_columns=input_columns,
            image_columns=image_columns,
            concurrency=concurrency,
        )
        states = self._run_rows(records, context=context, concurrency=concurrency)

        generated = pd.DataFrame([state["result"]["value"] if state["result"]["ok"] else {} for state in states], index=frame.index)
        generated = self._prepare_output_columns(frame, generated, prefix)

        parts = [frame.copy()]
        if not generated.empty:
            parts.append(generated)
        if status:
            parts.append(
                self._status_frame(
                    states,
                    index=frame.index,
                    include_output=include_output,
                    include_reasoning=include_reasoning,
                )
            )
        return pd.concat(parts, axis=1)

    def close(self) -> None:
        self.client.close()

    async def aclose(self) -> None:
        await self.client.aclose()

    def __enter__(self) -> Loom:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> Loom:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.aclose()
        return False

    def _run_rows(
        self,
        records: list[dict[str, Any]],
        *,
        context: dict[str, Any],
        concurrency: int,
    ) -> list[dict[str, Any]]:
        loaded: dict[int, dict[str, Any]] = {}
        pending: list[tuple[int, dict[str, Any], str]] = []
        fingerprint = self._fingerprint()
        resume = context["namespace"]

        for index, record in enumerate(records):
            key = fingerprint.for_input(record)
            payload = self.checkpoint.get(resume, key) if self.checkpoint and resume else None
            if payload is None:
                pending.append((index, record, key))
                continue
            state = json.loads(payload)
            state["result"]["checkpoint"] = True
            loaded[index] = state

        states: list[dict[str, Any] | None] = [loaded.get(index) for index in range(len(records))]
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(self._run_row, record, key, context): (index, key) for index, record, key in pending}
            for future in as_completed(futures):
                index, key = futures[future]
                state = future.result()
                if self.checkpoint and resume and state["result"]["ok"]:
                    self.checkpoint.put(resume, key, json.dumps(state, ensure_ascii=False, default=str))
                states[index] = state

        return [state for state in states if state is not None]

    def _run_row(self, record: dict[str, Any], fingerprint: str, context: dict[str, Any]) -> dict[str, Any]:
        last_error: str | None = None
        last_output: str | None = None
        last_reasoning: str | None = None
        messages = self.messages.build_messages(record)

        for attempt in range(1, self.retries + 2):
            try:
                output = self.client.complete(
                    model=self.model,
                    messages=messages,
                    params=self.params,
                )
                text, reasoning = extract_reasoning(output)
                value = parse_json_payload(text, auto_repair_json=self.repair_json)
                if not isinstance(value, dict):
                    raise ValueError("SilkLoom expects each model response to be a JSON object.")
                return self._state(
                    context=context,
                    fingerprint=fingerprint,
                    record=record,
                    messages=messages,
                    ok=True,
                    value=value,
                    error=None,
                    output=output,
                    reasoning=reasoning,
                    attempts=attempt,
                )
            except Exception:
                last_error = traceback.format_exc()
                last_output = locals().get("output")
                last_reasoning = locals().get("reasoning")

        return self._state(
            context=context,
            fingerprint=fingerprint,
            record=record,
            messages=messages,
            ok=False,
            value={},
            error=last_error,
            output=last_output,
            reasoning=last_reasoning,
            attempts=self.retries + 1,
        )

    def _record(self, row: Any, input_columns: Sequence[str], image_columns: Sequence[str]) -> dict[str, Any]:
        record = {column: row[column] for column in input_columns}
        image_refs: list[str] = []
        for column in image_columns:
            value = row[column]
            if self._is_missing(value):
                continue
            if isinstance(value, (list, tuple)):
                image_refs.extend(str(item) for item in value if not self._is_missing(item))
            else:
                image_refs.append(str(value))
        if image_refs:
            record["images"] = image_refs
        return record

    def _status_frame(
        self,
        states: list[dict[str, Any]],
        *,
        index: Any,
        include_output: bool,
        include_reasoning: bool,
    ):
        rows = []
        for state in states:
            result = state["result"]
            row = {
                "_loom_ok": result["ok"],
                "_loom_error": result["error"],
                "_loom_checkpoint": result["checkpoint"],
                "_loom_attempts": result["attempts"],
            }
            if include_output:
                row["_loom_output"] = result["output"]
            if include_reasoning:
                row["_loom_reasoning"] = result["reasoning"]
            rows.append(row)
        return self._pandas().DataFrame(rows, index=index)

    def _run_context(
        self,
        *,
        resume: str | None,
        input_columns: Sequence[str],
        image_columns: Sequence[str],
        concurrency: int,
    ) -> dict[str, Any]:
        return {
            "silkloom_version": __version__,
            "namespace": resume,
            "model": self.model,
            "prompt": self.prompt,
            "system": self.system,
            "params": self._safe_params(),
            "retries": self.retries,
            "repair_json": self.repair_json,
            "input_columns": list(input_columns),
            "image_columns": list(image_columns),
            "concurrency": concurrency,
            "output": "json-object",
        }

    def _state(
        self,
        *,
        context: dict[str, Any],
        fingerprint: str,
        record: dict[str, Any],
        messages: list[dict[str, Any]],
        ok: bool,
        value: dict[str, Any],
        error: str | None,
        output: str | None,
        reasoning: str | None,
        attempts: int,
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "context": context,
            "fingerprint": fingerprint,
            "input": record,
            "messages": messages,
            "result": {
                "ok": ok,
                "value": value,
                "error": error,
                "output": output,
                "reasoning": reasoning,
                "checkpoint": False,
                "attempts": attempts,
            },
        }

    def _prepare_output_columns(self, frame: Any, generated: Any, prefix: str | None):
        if generated.empty:
            return generated
        if prefix is not None:
            return generated.rename(columns={column: f"{prefix}{column}" for column in generated.columns})

        reserved = set(STATUS_COLUMNS + OPTIONAL_STATUS_COLUMNS)
        collisions = [column for column in generated.columns if column in frame.columns or column in reserved]
        if collisions:
            raise ValueError(
                "Model output columns conflict with input or reserved columns. "
                f"Conflicts: {collisions}. Pass prefix='llm_' to keep both."
            )
        return generated

    def _fingerprint(self) -> RunFingerprint:
        return RunFingerprint(
            model=self.model,
            prompt=self.prompt,
            system=self.system,
            output="json-object",
            params=self._safe_params(),
        )

    def _safe_params(self) -> Any:
        safe: dict[str, Any] = {}
        for key, value in sorted(self.params.items()):
            if callable(value):
                continue
            try:
                json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                safe[key] = value
            except Exception:
                safe[key] = str(value)
        return json.loads(json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str))

    def _frame(self, data: Any):
        pd = self._pandas()
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, Mapping):
            return pd.DataFrame(data)
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            return pd.DataFrame(list(data))
        raise TypeError("Loom expects a pandas DataFrame, a mapping of columns, or a sequence of row mappings.")

    def _require_columns(self, frame: Any, columns: Sequence[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Input columns are missing from data: {missing}")

    def _client(self, client: ChatClient | Any | None) -> ChatClient:
        if client is None:
            return OpenAIChat()
        if all(hasattr(client, attr) for attr in ("complete", "acomplete", "close", "aclose")):
            return client
        return OpenAICompatible(client)

    def _checkpoint(self, checkpoint: str | Path | ResultStore | None) -> ResultStore | None:
        if checkpoint is None:
            return None
        if isinstance(checkpoint, (str, Path)):
            return SQLiteCheckpoint(checkpoint)
        return checkpoint

    def _is_missing(self, value: Any) -> bool:
        try:
            return bool(self._pandas().isna(value))
        except (TypeError, ValueError):
            return False

    def _pandas(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SilkLoom requires pandas.") from exc
        return pd
