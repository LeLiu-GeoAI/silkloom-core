from __future__ import annotations

import asyncio
import inspect
from typing import Any, Protocol

from openai import AsyncOpenAI, OpenAI


class ChatClient(Protocol):
    def complete(self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str: ...

    async def acomplete(self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str: ...

    def close(self) -> None: ...

    async def aclose(self) -> None: ...


class OpenAIChat:
    def __init__(self, sync: Any | None = None, async_: Any | None = None):
        self.sync = sync or OpenAI()
        self.async_ = async_ or AsyncOpenAI()

    def complete(self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
        response = self.sync.chat.completions.create(model=model, messages=messages, **params)
        if inspect.isawaitable(response):
            raise TypeError("Synchronous run received an awaitable OpenAI response.")
        return extract_text(response)

    async def acomplete(self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
        create = self.async_.chat.completions.create
        response = create(model=model, messages=messages, **params)
        if inspect.isawaitable(response):
            return extract_text(await response)
        return extract_text(response)

    def close(self) -> None:
        close_once(self.sync)
        if self.async_ is not self.sync:
            close_once(self.async_)

    async def aclose(self) -> None:
        await aclose_once(self.async_)
        if self.sync is not self.async_:
            close_once(self.sync)


class OpenAICompatible(OpenAIChat):
    def __init__(self, client: Any):
        super().__init__(sync=client, async_=client)

    async def acomplete(self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
        create = self.async_.chat.completions.create
        response = create(model=model, messages=messages, **params)
        if inspect.isawaitable(response):
            return extract_text(await response)
        return extract_text(response)


def extract_text(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def close_once(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return

    aclose = getattr(client, "aclose", None)
    if callable(aclose):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(aclose())
        else:
            loop.create_task(aclose())


async def aclose_once(client: Any) -> None:
    aclose = getattr(client, "aclose", None)
    if callable(aclose):
        await aclose()
        return
    close_once(client)
