from __future__ import annotations

import asyncio
import inspect
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Generic, Iterable, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .cache import SQLiteCache, hash_input
from .json_utils import extract_reasoning, parse_json_payload
from .message_builder import MessageBuilder
from .models import BatchResult, TaskResult

T = TypeVar("T")


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
        **llm_kwargs: Any,
    ):
        self.model = model
        self.response_model = response_model
        self.auto_repair_json = auto_repair_json
        self.max_retries = max_retries
        self.llm_kwargs = llm_kwargs
        self._message_builder = MessageBuilder(prompt_template, system_prompt)

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
        db_path: str = ".silkloom.db",
        run_id: str | None = None,
        workers: int = 5,
    ) -> BatchResult[T]:
        inputs = [self._normalize_input(item) for item in sequence]
        results: list[TaskResult[T] | None] = [None] * len(inputs)

        cache = SQLiteCache(db_path) if run_id else None
        pending: list[tuple[int, dict[str, Any], str]] = []

        for idx, input_data in enumerate(inputs):
            input_key = hash_input(input_data)
            if cache and run_id:
                cached = cache.get(run_id, input_key)
                if cached is not None:
                    results[idx] = self._deserialize_task_result(cached)
                    continue
            pending.append((idx, input_data, input_key))

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                future_map = {
                    executor.submit(self._process_with_retries, input_data): (idx, input_key, input_data)
                    for idx, input_data, input_key in pending
                }

                for future in as_completed(future_map):
                    idx, input_key, _ = future_map[future]
                    task_result = future.result()
                    results[idx] = task_result

                    if cache and run_id and task_result.is_success:
                        cache.set(run_id, input_key, task_result.model_dump_json())

        return BatchResult([item for item in results if item is not None])

    async def amap(
        self,
        sequence: Iterable[str | dict],
        db_path: str = ".silkloom.db",
        run_id: str | None = None,
        max_concurrent: int = 5,
    ) -> BatchResult[T]:
        inputs = [self._normalize_input(item) for item in sequence]
        results: list[TaskResult[T] | None] = [None] * len(inputs)

        cache = SQLiteCache(db_path) if run_id else None
        pending: list[tuple[int, dict[str, Any], str]] = []

        for idx, input_data in enumerate(inputs):
            input_key = hash_input(input_data)
            if cache and run_id:
                cached = cache.get(run_id, input_key)
                if cached is not None:
                    results[idx] = self._deserialize_task_result(cached)
                    continue
            pending.append((idx, input_data, input_key))

        semaphore = asyncio.Semaphore(max(1, max_concurrent))

        async def _run_one(idx: int, input_data: dict[str, Any], input_key: str) -> None:
            async with semaphore:
                task_result = await self._aprocess_with_retries(input_data)
            results[idx] = task_result
            if cache and run_id and task_result.is_success:
                cache.set(run_id, input_key, task_result.model_dump_json())

        await asyncio.gather(*[_run_one(idx, inp, key) for idx, inp, key in pending])

        return BatchResult([item for item in results if item is not None])

    def _normalize_input(self, data: str | dict) -> dict[str, Any]:
        if isinstance(data, str):
            return {"text": data}
        if isinstance(data, dict):
            return data
        raise TypeError("Input data must be str or dict")

    def _process_with_retries(self, input_data: dict[str, Any]) -> TaskResult[T]:
        last_error: str | None = None
        last_raw: str | None = None
        last_reasoning: str | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_output = self._call_llm_sync(input_data)
                cleaned_output, reasoning = extract_reasoning(raw_output)
                parsed = self._parse_output(cleaned_output, raw_output)
                return TaskResult[T](
                    is_success=True,
                    data=parsed,
                    error=None,
                    input_data=input_data,
                    raw_output=raw_output,
                    reasoning=reasoning,
                )
            except Exception:
                last_error = traceback.format_exc()
                if "raw_output" in locals():
                    last_raw = raw_output
                if "reasoning" in locals():
                    last_reasoning = reasoning

        return TaskResult[T](
            is_success=False,
            data=None,
            error=last_error,
            input_data=input_data,
            raw_output=last_raw,
            reasoning=last_reasoning,
        )

    async def _aprocess_with_retries(self, input_data: dict[str, Any]) -> TaskResult[T]:
        last_error: str | None = None
        last_raw: str | None = None
        last_reasoning: str | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_output = await self._call_llm_async(input_data)
                cleaned_output, reasoning = extract_reasoning(raw_output)
                parsed = self._parse_output(cleaned_output, raw_output)
                return TaskResult[T](
                    is_success=True,
                    data=parsed,
                    error=None,
                    input_data=input_data,
                    raw_output=raw_output,
                    reasoning=reasoning,
                )
            except Exception:
                last_error = traceback.format_exc()
                if "raw_output" in locals():
                    last_raw = raw_output
                if "reasoning" in locals():
                    last_reasoning = reasoning

        return TaskResult[T](
            is_success=False,
            data=None,
            error=last_error,
            input_data=input_data,
            raw_output=last_raw,
            reasoning=last_reasoning,
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

        return TaskResult[T](
            is_success=bool(data.get("is_success")),
            data=parsed_data,
            error=data.get("error"),
            input_data=data.get("input_data") or {},
            raw_output=data.get("raw_output"),
            reasoning=data.get("reasoning"),
        )
