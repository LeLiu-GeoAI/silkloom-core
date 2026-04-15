from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from .utils import render_template, resolve_mapping


@dataclass
class NodeResult:
    output: dict[str, Any]
    retries_used: int


@dataclass
class CollectNode:
    name: str
    func: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]]
    source_node: Optional[str] = None
    include_failed: bool = False


class BaseNode(ABC):
    def __init__(
        self,
        name: str,
        max_retries: int = 3,
    ) -> None:
        if not name:
            raise ValueError("Node name must not be empty.")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        self.name = name
        self.max_retries = max_retries

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
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(name=name, max_retries=max_retries)
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
    ) -> None:
        super().__init__(name=name, max_retries=max_retries)
        self.func = func
        self.kwargs_mapping = kwargs_mapping or {}

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        kwargs = resolve_mapping(self.kwargs_mapping, context)
        result = self.func(**kwargs)
        if not isinstance(result, dict):
            raise TypeError(f"FunctionNode {self.name} function must return a dict.")
        return result
