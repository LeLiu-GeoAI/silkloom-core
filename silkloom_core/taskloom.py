from __future__ import annotations

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, AsyncIterator, Generic, Iterable, Iterator, TypeVar

from pydantic import BaseModel

from .cache import SQLiteCheckpoint, stable_hash
from .clients import ChatClient, OpenAIChat, OpenAICompatible
from .message_builder import MessageBuilder
from .models import Batch, Result
from .output import OutputParser

T = TypeVar("T")


class Loom(Generic[T]):
    def __init__(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        output: type[BaseModel] | type[dict] | None = None,
        client: ChatClient | Any | None = None,
        cache: str | SQLiteCheckpoint | None = ".silkloom.db",
        retries: int = 2,
        repair_json: bool = True,
        **params: Any,
    ):
        self.model = model
        self.prompt = prompt
        self.system = system
        self.params = params
        self.retries = max(0, retries)
        self.messages = MessageBuilder(prompt, system)
        self.parser: OutputParser[T] = OutputParser(output, repair_json=repair_json)
        self.client: ChatClient = self._adapt_client(client)
        self.cache = self._adapt_cache(cache)

    def run(self, data: str | dict[str, Any]) -> Result[T]:
        item = self._normalize(data)
        return self._run_uncached(item)

    async def arun(self, data: str | dict[str, Any]) -> Result[T]:
        item = self._normalize(data)
        return await self._arun_uncached(item)

    def batch(
        self,
        items: Iterable[str | dict[str, Any]],
        *,
        name: str | None = None,
        concurrency: int = 5,
    ) -> Batch[T]:
        return Batch(list(self.each(items, name=name, concurrency=concurrency, ordered=True)))

    async def abatch(
        self,
        items: Iterable[str | dict[str, Any]],
        *,
        name: str | None = None,
        concurrency: int = 5,
    ) -> Batch[T]:
        results = [item async for item in self.aeach(items, name=name, concurrency=concurrency, ordered=True)]
        return Batch(results)

    def each(
        self,
        items: Iterable[str | dict[str, Any]],
        *,
        name: str | None = None,
        concurrency: int = 5,
        ordered: bool = False,
    ) -> Iterator[Result[T]]:
        normalized = [self._normalize(item) for item in items]
        cached, pending = self._split_cache(normalized, name)

        if ordered:
            yield from self._each_ordered(normalized, cached, pending, name, concurrency)
            return

        for result in cached.values():
            yield result

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {
                pool.submit(self._run_uncached, item): (index, item, key)
                for index, item, key in pending
            }
            for future in as_completed(futures):
                _, _, key = futures[future]
                result = future.result()
                self._store(name, key, result)
                yield result

    async def aeach(
        self,
        items: Iterable[str | dict[str, Any]],
        *,
        name: str | None = None,
        concurrency: int = 5,
        ordered: bool = False,
    ) -> AsyncIterator[Result[T]]:
        normalized = [self._normalize(item) for item in items]
        cached, pending = self._split_cache(normalized, name)

        if ordered:
            async for result in self._aeach_ordered(normalized, cached, pending, name, concurrency):
                yield result
            return

        for result in cached.values():
            yield result

        queue: asyncio.Queue[tuple[str, Result[T]]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def worker(item: dict[str, Any], key: str) -> None:
            async with semaphore:
                result = await self._arun_uncached(item)
            self._store(name, key, result)
            await queue.put((key, result))

        tasks = [asyncio.create_task(worker(item, key)) for _, item, key in pending]
        try:
            for _ in tasks:
                _, result = await queue.get()
                yield result
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

    def close(self) -> None:
        self.client.close()

    async def aclose(self) -> None:
        await self.client.aclose()

    def __enter__(self) -> Loom[T]:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> Loom[T]:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.aclose()
        return False

    def _each_ordered(
        self,
        normalized: list[dict[str, Any]],
        cached: dict[int, Result[T]],
        pending: list[tuple[int, dict[str, Any], str]],
        name: str | None,
        concurrency: int,
    ) -> Iterator[Result[T]]:
        buffer = dict(cached)
        next_index = 0
        while next_index in buffer:
            yield buffer.pop(next_index)
            next_index += 1

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {
                pool.submit(self._run_uncached, item): (index, key)
                for index, item, key in pending
            }
            for future in as_completed(futures):
                index, key = futures[future]
                result = future.result()
                self._store(name, key, result)
                buffer[index] = result
                while next_index in buffer:
                    yield buffer.pop(next_index)
                    next_index += 1

    async def _aeach_ordered(
        self,
        normalized: list[dict[str, Any]],
        cached: dict[int, Result[T]],
        pending: list[tuple[int, dict[str, Any], str]],
        name: str | None,
        concurrency: int,
    ) -> AsyncIterator[Result[T]]:
        buffer = dict(cached)
        next_index = 0
        while next_index in buffer:
            yield buffer.pop(next_index)
            next_index += 1

        queue: asyncio.Queue[tuple[int, Result[T]]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def worker(index: int, item: dict[str, Any], key: str) -> None:
            async with semaphore:
                result = await self._arun_uncached(item)
            self._store(name, key, result)
            await queue.put((index, result))

        tasks = [asyncio.create_task(worker(index, item, key)) for index, item, key in pending]
        try:
            for _ in tasks:
                index, result = await queue.get()
                buffer[index] = result
                while next_index in buffer:
                    yield buffer.pop(next_index)
                    next_index += 1
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _run_uncached(self, item: dict[str, Any]) -> Result[T]:
        last_error: str | None = None
        last_output: str | None = None
        last_reasoning: str | None = None
        run_id = stable_hash(item)

        for attempt in range(1, self.retries + 2):
            try:
                output = self.client.complete(
                    model=self.model,
                    messages=self.messages.build_messages(item),
                    params=self.params,
                )
                value, reasoning = self.parser.parse(output)
                return Result(
                    id=run_id,
                    ok=True,
                    value=value,
                    input=item,
                    output=output,
                    reasoning=reasoning,
                    attempts=attempt,
                )
            except Exception:
                last_error = traceback.format_exc()
                last_output = locals().get("output")
                last_reasoning = locals().get("reasoning")

        return Result(
            id=run_id,
            ok=False,
            error=last_error,
            input=item,
            output=last_output,
            reasoning=last_reasoning,
            attempts=self.retries + 1,
        )

    async def _arun_uncached(self, item: dict[str, Any]) -> Result[T]:
        last_error: str | None = None
        last_output: str | None = None
        last_reasoning: str | None = None
        run_id = stable_hash(item)

        for attempt in range(1, self.retries + 2):
            try:
                output = await self.client.acomplete(
                    model=self.model,
                    messages=self.messages.build_messages(item),
                    params=self.params,
                )
                value, reasoning = self.parser.parse(output)
                return Result(
                    id=run_id,
                    ok=True,
                    value=value,
                    input=item,
                    output=output,
                    reasoning=reasoning,
                    attempts=attempt,
                )
            except Exception:
                last_error = traceback.format_exc()
                last_output = locals().get("output")
                last_reasoning = locals().get("reasoning")

        return Result(
            id=run_id,
            ok=False,
            error=last_error,
            input=item,
            output=last_output,
            reasoning=last_reasoning,
            attempts=self.retries + 1,
        )

    def _split_cache(
        self,
        items: list[dict[str, Any]],
        name: str | None,
    ) -> tuple[dict[int, Result[T]], list[tuple[int, dict[str, Any], str]]]:
        cached: dict[int, Result[T]] = {}
        pending: list[tuple[int, dict[str, Any], str]] = []

        for index, item in enumerate(items):
            key = self._cache_key(item)
            payload = self.cache.get(name, key) if self.cache and name else None
            if payload is None:
                pending.append((index, item, key))
                continue
            result = self._deserialize(payload)
            result.cache_hit = True
            cached[index] = result
        return cached, pending

    def _store(self, name: str | None, key: str, result: Result[T]) -> None:
        if self.cache and name and result.ok:
            self.cache.set(name, key, result.model_dump_json())

    def _cache_key(self, item: dict[str, Any]) -> str:
        return stable_hash(
            {
                "input": item,
                "model": self.model,
                "prompt": self.prompt,
                "system": self.system,
                "output": self.parser.fingerprint(),
                "params": self._safe_params(),
            }
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

    def _deserialize(self, payload: str) -> Result[T]:
        data = json.loads(payload)
        result: Result[T] = Result.model_validate(data)
        if result.value is not None and isinstance(self.parser.schema, type) and issubclass(self.parser.schema, BaseModel):
            result.value = self.parser.schema.model_validate(result.value)
        return result

    def _normalize(self, data: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, str):
            return {"text": data}
        if isinstance(data, dict):
            return data
        raise TypeError("SilkLoom input must be a string or a dict.")

    def _adapt_client(self, client: ChatClient | Any | None) -> ChatClient:
        if client is None:
            return OpenAIChat()
        if all(hasattr(client, attr) for attr in ("complete", "acomplete", "close", "aclose")):
            return client
        return OpenAICompatible(client)

    def _adapt_cache(self, cache: str | SQLiteCheckpoint | None) -> SQLiteCheckpoint | None:
        if cache is None:
            return None
        if isinstance(cache, SQLiteCheckpoint):
            return cache
        return SQLiteCheckpoint(cache)
