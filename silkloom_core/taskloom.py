from __future__ import annotations

import asyncio
import inspect
import json
import traceback
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Generic, Iterable, TypeVar, Dict, Tuple

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .cache import SQLiteCache, hash_input
from .json_utils import extract_reasoning, parse_json_payload
from .message_builder import MessageBuilder
from .models import BatchResult, TaskResult

T = TypeVar("T")
BatchProgressCallback = Callable[[int, int, dict[str, Any], TaskResult[T] | None, str], Any]


class TaskLoom(Generic[T]):
    def __init__(
        self,
        model: str,
        prompt_template: str,
        system_prompt: str | None = None,
        response_model: type[BaseModel] | type[dict] | None = None,
        auto_repair_json: bool = True,
        max_retries: int = 3,
        client: Any | None = None,
        db_path: str = ".silkloom.db",
        **llm_kwargs: Any,
    ):
        self.model = model
        self.response_model = response_model
        self.auto_repair_json = auto_repair_json
        self.max_retries = max_retries
        self.llm_kwargs = llm_kwargs
        self._message_builder = MessageBuilder(prompt_template, system_prompt)
        self.db_path = db_path

        self._sync_client = client or OpenAI()
        self._async_client = AsyncOpenAI() if client is None else client

    def process(self, data: str | dict) -> TaskResult[T]:
        input_data = self._normalize_input(data)
        return self._process_with_retries(input_data)

    async def aprocess(self, data: str | dict) -> TaskResult[T]:
        input_data = self._normalize_input(data)
        return await self._aprocess_with_retries(input_data)

    def map(
        self,
        sequence: Iterable[str | dict],
        task_name: str | None = None,
        max_workers: int = 5,
        show_progress: bool = False,
        progress_desc: str = "TaskLoom map",
        progress_callback: BatchProgressCallback[T] | None = None,
    ) -> BatchResult[T]:
        inputs = [self._normalize_input(item) for item in sequence]
        results: list[TaskResult[T] | None] = [None] * len(inputs)

        cache = SQLiteCache(self.db_path) if task_name else None
        pending: list[tuple[int, dict[str, Any], str]] = []

        progress_context = self._progress_context(len(inputs), show_progress, progress_desc)
        with progress_context as progress_bar:
            for idx, input_data in enumerate(inputs):
                input_key = hash_input(input_data)
                if cache and task_name:
                    cached = cache.get(task_name, input_key)
                    if cached is not None:
                        task_result = self._deserialize_task_result(cached)
                        task_result.cached = True
                        results[idx] = task_result
                        completed_count = sum(item is not None for item in results)
                        status_text = self._build_progress_text(completed_count, len(inputs), input_data, task_result)
                        self._emit_progress_callback(
                            progress_callback,
                            completed_count,
                            len(inputs),
                            input_data,
                            task_result,
                            status_text,
                        )
                        if progress_bar is not None:
                            progress_bar.update(1)
                        continue
                pending.append((idx, input_data, input_key))

            if pending:
                with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
                    future_map = {
                        executor.submit(self._process_with_retries, input_data): (idx, input_key, input_data)
                        for idx, input_data, input_key in pending
                    }

                    for future in as_completed(future_map):
                        idx, input_key, _ = future_map[future]
                        task_result = future.result()
                        results[idx] = task_result
                        completed_count = sum(item is not None for item in results)
                        status_text = self._build_progress_text(completed_count, len(inputs), inputs[idx], task_result)

                        if cache and task_name and task_result.is_success:
                            cache.set(task_name, input_key, task_result.model_dump_json())
                        self._emit_progress_callback(
                            progress_callback,
                            completed_count,
                            len(inputs),
                            inputs[idx],
                            task_result,
                            status_text,
                        )
                        if progress_bar is not None:
                            progress_bar.update(1)

        return BatchResult([item for item in results if item is not None])

    async def amap(
        self,
        sequence: Iterable[str | dict],
        task_name: str | None = None,
        max_concurrent: int = 5,
        show_progress: bool = False,
        progress_desc: str = "TaskLoom amap",
        progress_callback: BatchProgressCallback[T] | None = None,
    ) -> BatchResult[T]:
        inputs = [self._normalize_input(item) for item in sequence]
        results: list[TaskResult[T] | None] = [None] * len(inputs)
        cache = SQLiteCache(self.db_path) if task_name else None
        pending: list[tuple[int, dict[str, Any], str]] = []

        progress_context = self._progress_context(len(inputs), show_progress, progress_desc)
        with progress_context as progress_bar:
            for idx, input_data in enumerate(inputs):
                input_key = hash_input(input_data)
                if cache and task_name:
                    cached = cache.get(task_name, input_key)
                    if cached is not None:
                        task_result = self._deserialize_task_result(cached)
                        task_result.cached = True
                        results[idx] = task_result
                        completed_count = sum(item is not None for item in results)
                        status_text = self._build_progress_text(completed_count, len(inputs), input_data, task_result)
                        self._emit_progress_callback(
                            progress_callback,
                            completed_count,
                            len(inputs),
                            input_data,
                            task_result,
                            status_text,
                        )
                        if progress_bar is not None:
                            progress_bar.update(1)
                        continue
                pending.append((idx, input_data, input_key))

            semaphore = asyncio.Semaphore(max(1, max_concurrent))

            async def _run_one(idx: int, input_data: dict[str, Any], input_key: str) -> None:
                async with semaphore:
                    task_result = await self._aprocess_with_retries(input_data)
                results[idx] = task_result
                completed_count = sum(item is not None for item in results)
                status_text = self._build_progress_text(completed_count, len(inputs), input_data, task_result)
                if cache and task_name and task_result.is_success:
                    cache.set(task_name, input_key, task_result.model_dump_json())
                self._emit_progress_callback(
                    progress_callback,
                    completed_count,
                    len(inputs),
                    input_data,
                    task_result,
                    status_text,
                )
                if progress_bar is not None:
                    progress_bar.update(1)

            await asyncio.gather(*[_run_one(idx, inp, key) for idx, inp, key in pending])

        return BatchResult([item for item in results if item is not None])

    def stream(
        self,
        sequence: Iterable[str | dict],
        task_name: str | None = None,
        max_workers: int = 5,
        ordered: bool = False,
    ) -> Iterable[TaskResult[T]]:
        inputs = [self._normalize_input(item) for item in sequence]
        cache = SQLiteCache(self.db_path) if task_name else None

        pending: list[Tuple[int, dict[str, Any], str]] = []
        for idx, input_data in enumerate(inputs):
            input_key = hash_input(input_data)
            if cache and task_name:
                cached = cache.get(task_name, input_key)
                if cached is not None:
                    task_result = self._deserialize_task_result(cached)
                    task_result.cached = True
                    yield task_result
                    continue
            pending.append((idx, input_data, input_key))

        if not pending:
            return

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            future_map: Dict[Any, Tuple[int, str]] = {}
            for idx, input_data, input_key in pending:
                fut = executor.submit(self._process_with_retries, input_data)
                future_map[fut] = (idx, input_key)

            if not ordered:
                for fut in as_completed(future_map):
                    idx, input_key = future_map[fut]
                    task_result = fut.result()
                    if cache and task_name and task_result.is_success:
                        cache.set(task_name, input_key, task_result.model_dump_json())
                    yield task_result
            else:
                buffer: Dict[int, TaskResult[T]] = {}
                next_idx = 0
                for fut in as_completed(future_map):
                    idx, input_key = future_map[fut]
                    task_result = fut.result()
                    if cache and task_name and task_result.is_success:
                        cache.set(task_name, input_key, task_result.model_dump_json())
                    buffer[idx] = task_result
                    while next_idx in buffer:
                        yield buffer.pop(next_idx)
                        next_idx += 1

    async def astream(
        self,
        sequence: Iterable[str | dict],
        task_name: str | None = None,
        max_workers: int = 5,
        ordered: bool = False,
    ) -> AsyncGenerator[TaskResult[T], None]:
        from asyncio import Queue

        inputs = [self._normalize_input(item) for item in sequence]
        cache = SQLiteCache(self.db_path) if task_name else None

        pending: list[Tuple[int, dict[str, Any], str]] = []
        for idx, input_data in enumerate(inputs):
            input_key = hash_input(input_data)
            if cache and task_name:
                cached = cache.get(task_name, input_key)
                if cached is not None:
                    task_result = self._deserialize_task_result(cached)
                    task_result.cached = True
                    yield task_result
                    continue
            pending.append((idx, input_data, input_key))

        if not pending:
            return

        semaphore = asyncio.Semaphore(max(1, max_workers))
        q: Queue = Queue()

        async def worker(idx: int, input_data: dict[str, Any], input_key: str):
            async with semaphore:
                res = await self._aprocess_with_retries(input_data)
            if cache and task_name and res.is_success:
                cache.set(task_name, input_key, res.model_dump_json())
            await q.put((idx, res))

        tasks = [asyncio.create_task(worker(idx, inp, key)) for idx, inp, key in pending]

        if not ordered:
            finished = 0
            while finished < len(tasks):
                idx, res = await q.get()
                finished += 1
                yield res
        else:
            buffer: Dict[int, TaskResult[T]] = {}
            next_idx = 0
            finished = 0
            while finished < len(tasks):
                idx, res = await q.get()
                finished += 1
                buffer[idx] = res
                while next_idx in buffer:
                    yield buffer.pop(next_idx)
                    next_idx += 1


    def _normalize_input(self, data: str | dict) -> dict[str, Any]:
        if isinstance(data, str):
            return {"text": data}
        if isinstance(data, dict):
            return data
        raise TypeError("Input data must be str or dict")

    def _progress_context(self, total: int, enabled: bool, desc: str):
        if not enabled:
            return nullcontext(None)

        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError("tqdm is required for progress bars, install silkloom-core[progress]") from exc

        return tqdm(total=total, desc=desc)

    def _build_progress_text(
        self,
        completed: int,
        total: int,
        input_data: dict[str, Any],
        task_result: TaskResult[T],
    ) -> str:
        prefix = "处理完成" if task_result.is_success else "处理失败"
        text = input_data.get("text")
        if isinstance(text, str) and text.strip():
            preview = text.strip().replace("\n", " ")[:48]
            if len(text.strip()) > 48:
                preview += "..."
            return f"{prefix} 第 {completed}/{total} 篇：{preview}"
        return f"{prefix} 第 {completed}/{total} 项"

    def _emit_progress_callback(
        self,
        progress_callback: BatchProgressCallback[T] | None,
        completed: int,
        total: int,
        input_data: dict[str, Any],
        task_result: TaskResult[T],
        status_text: str,
    ) -> None:
        if progress_callback is None:
            return

        progress_callback(completed, total, input_data, task_result, status_text)

    def _process_with_retries(self, input_data: dict[str, Any]) -> TaskResult[T]:
        last_error: str | None = None
        last_raw: str | None = None
        last_reasoning: str | None = None
        input_hash = hash_input(input_data)

        for _ in range(self.max_retries + 1):
            try:
                raw_output = self._call_llm_sync(input_data)
                cleaned_output, reasoning = extract_reasoning(raw_output)
                parsed = self._parse_output(cleaned_output, raw_output)
                return TaskResult[T](
                    task_id=input_hash,
                    is_success=True,
                    data=parsed,
                    error=None,
                    input_data=input_data,
                    raw_output=raw_output,
                    reasoning=reasoning,
                    cached=False,
                )
            except Exception:
                last_error = traceback.format_exc()
                if "raw_output" in locals():
                    last_raw = raw_output
                if "reasoning" in locals():
                    last_reasoning = reasoning

        return TaskResult[T](
            task_id=input_hash,
            is_success=False,
            data=None,
            error=last_error,
            input_data=input_data,
            raw_output=last_raw,
            reasoning=last_reasoning,
            cached=False,
        )

    async def _aprocess_with_retries(self, input_data: dict[str, Any]) -> TaskResult[T]:
        last_error: str | None = None
        last_raw: str | None = None
        last_reasoning: str | None = None
        input_hash = hash_input(input_data)

        for _ in range(self.max_retries + 1):
            try:
                raw_output = await self._call_llm_async(input_data)
                cleaned_output, reasoning = extract_reasoning(raw_output)
                parsed = self._parse_output(cleaned_output, raw_output)
                return TaskResult[T](
                    task_id=input_hash,
                    is_success=True,
                    data=parsed,
                    error=None,
                    input_data=input_data,
                    raw_output=raw_output,
                    reasoning=reasoning,
                    cached=False,
                )
            except Exception:
                last_error = traceback.format_exc()
                if "raw_output" in locals():
                    last_raw = raw_output
                if "reasoning" in locals():
                    last_reasoning = reasoning

        return TaskResult[T](
            task_id=input_hash,
            is_success=False,
            data=None,
            error=last_error,
            input_data=input_data,
            raw_output=last_raw,
            reasoning=last_reasoning,
            cached=False,
        )

    def _call_llm_sync(self, input_data: dict[str, Any]) -> str:
        messages = self._message_builder.build_messages(input_data)
        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.llm_kwargs,
        )
        if inspect.isawaitable(response):
            raise TypeError("Sync process received an async client response")
        return self._extract_content(response)

    async def _call_llm_async(self, input_data: dict[str, Any]) -> str:
        messages = self._message_builder.build_messages(input_data)
        create_func = self._async_client.chat.completions.create
        maybe_awaitable = create_func(model=self.model, messages=messages, **self.llm_kwargs)

        if inspect.isawaitable(maybe_awaitable):
            response = await maybe_awaitable
        else:
            # User may pass a sync-only client; run it in thread to keep async API usable.
            response = await asyncio.to_thread(
                create_func,
                model=self.model,
                messages=messages,
                **self.llm_kwargs,
            )

        return self._extract_content(response)

    def _extract_content(self, response: Any) -> str:
        message = response.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(text)
            return "\n".join(parts)

        return str(content)

    def _parse_output(self, cleaned_output: str, raw_output: str) -> T:
        if self.response_model is None:
            return raw_output  # type: ignore[return-value]

        parsed = parse_json_payload(cleaned_output, auto_repair_json=self.auto_repair_json)

        if self.response_model is dict:
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object for response_model=dict")
            return parsed  # type: ignore[return-value]

        if isinstance(self.response_model, type) and issubclass(self.response_model, BaseModel):
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object for Pydantic response_model")
            return self.response_model.model_validate(parsed)  # type: ignore[return-value]

        raise TypeError("response_model must be BaseModel subclass, dict, or None")

    def _deserialize_task_result(self, payload: str) -> TaskResult[T]:
        data = json.loads(payload)
        parsed_data = data.get("data")

        if (
            self.response_model
            and self.response_model is not dict
            and isinstance(self.response_model, type)
            and issubclass(self.response_model, BaseModel)
            and isinstance(parsed_data, dict)
        ):
            parsed_data = self.response_model.model_validate(parsed_data)

        task_id = data.get("task_id")
        if not task_id:
            # fallback to hashing input_data if available
            input_info = data.get("input_data") or {}
            try:
                task_id = hash_input(input_info)
            except Exception:
                task_id = ""

        return TaskResult[T](
            task_id=task_id,
            is_success=bool(data.get("is_success")),
            data=parsed_data,
            error=data.get("error"),
            input_data=data.get("input_data") or {},
            raw_output=data.get("raw_output"),
            reasoning=data.get("reasoning"),
            cached=bool(data.get("cached", False)),
        )
