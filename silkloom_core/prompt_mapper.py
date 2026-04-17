from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Optional, Protocol, TypeVar, overload

import jinja2
from openai import OpenAI
from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd

from .results import ResultSet
from .types import TaskResult

TModel = TypeVar("TModel", bound=BaseModel)
TOutput = TypeVar("TOutput")

PromptContext = dict[str, Any]


class _ChatCompletionsProtocol(Protocol):
    def create(self, *, model: str, messages: list[dict[str, Any]]) -> Any:
        ...


class _ChatProtocol(Protocol):
    completions: _ChatCompletionsProtocol


class _BetaChatCompletionsProtocol(Protocol):
    def parse(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_format: type[BaseModel],
    ) -> Any:
        ...


class _BetaChatProtocol(Protocol):
    completions: _BetaChatCompletionsProtocol


class _BetaProtocol(Protocol):
    chat: _BetaChatProtocol


class OpenAIClientProtocol(Protocol):
    chat: _ChatProtocol
    beta: _BetaProtocol


class PromptMapper(Generic[TOutput]):
    @overload
    def __init__(
        self: "PromptMapper[str]",
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response_model: None = None,
        max_retries: int = 3,
        client: Optional[OpenAIClientProtocol] = None,
    ):
        ...

    @overload
    def __init__(
        self: "PromptMapper[TModel]",
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response_model: type[TModel] = ...,
        max_retries: int = 3,
        client: Optional[OpenAIClientProtocol] = None,
    ):
        ...

    def __init__(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response_model: Optional[type[BaseModel]] = None,
        max_retries: int = 3,
        client: Optional[OpenAIClientProtocol] = None,
    ):
        """Create a prompt mapper for single or batch LLM inference.

        user_prompt and system_prompt are strict Jinja2 templates. Each item
        passed to map() becomes the template context, so template variables must
        match the keys in the input dict, for example {{ text }} or {{ images }}.
        If map() receives a pandas DataFrame, each row is treated as one input
        dict and the column names become the available template variables.
        """
        self.model = model
        self.response_model = response_model
        self.max_retries = max_retries
        # 默认使用官方客户端，也可注入第三方 (如 GLM, Ollama)
        self.client = client or OpenAI()

        # 预编译 Jinja2 模板，缺失变量时直接报错，避免静默渲染为空字符串
        self._template_env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self._user_prompt_source = user_prompt
        self._system_prompt_source = system_prompt
        self.user_template = self._template_env.from_string(user_prompt)
        self.system_template = self._template_env.from_string(system_prompt) if system_prompt else None

    @staticmethod
    def _normalize_raw_output(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _extract_reasoning(completion: Any) -> Optional[str]:
        try:
            message = completion.choices[0].message
        except Exception:
            return None

        # 部分 OpenAI 兼容模型会直接提供 reasoning/reasoning_content 字段
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message, attr, None)
            if value:
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

        content = getattr(message, "content", None)
        if isinstance(content, str):
            # 常见 think 模型会把推理包在 <think>...</think> 中
            match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                return reasoning or None
            return None

        if isinstance(content, list):
            reasoning_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"reasoning", "thinking", "reasoning_text"}:
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str) and text.strip():
                        reasoning_parts.append(text.strip())
            if reasoning_parts:
                return "\n".join(reasoning_parts)

        return None

    def _render_messages(self, item: PromptContext) -> list[dict[str, Any]]:
        """将输入字典渲染为 OpenAI API 需要的 messages 列表"""
        messages: list[dict[str, Any]] = []
        if self.system_template:
            messages.append({"role": "system", "content": self.system_template.render(**item)})

        user_content = self.user_template.render(**item)
        
        # 多模态处理：如果存在 images 字段，转化为数组结构
        images = item.get("images", [])
        if images:
            content_list = [{"type": "text", "text": user_content}]
            for img in images:
                # 这里简单处理，假设已经是 url 或 base64 data URI
                content_list.append({"type": "image_url", "image_url": {"url": img}})
            messages.append({"role": "user", "content": content_list})
        else:
            messages.append({"role": "user", "content": user_content})
            
        return messages

    def _execute_single(self, item: PromptContext) -> TaskResult[TOutput]:
        """执行单次 API 调用，供 engine.py 在线程池中调用"""
        messages = self._render_messages(item)
        last_error = None
        last_raw_output: Any = None

        for attempt in range(self.max_retries):
            try:
                # 结构化输出模式 (Pydantic)
                if self.response_model:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=self.response_model,
                    )
                    raw_output = self._normalize_raw_output(completion)
                    return TaskResult(
                        is_success=True,
                        data=completion.choices[0].message.parsed,
                        usage=completion.usage.model_dump() if completion.usage else None,
                        input_data=item,
                        raw_output=raw_output,
                        reasoning=self._extract_reasoning(completion),
                    ) # pyright: ignore[reportReturnType]
                # 纯文本模式
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # type: ignore
                )
                raw_output = self._normalize_raw_output(completion)
                return TaskResult(
                    is_success=True,
                    data=completion.choices[0].message.content,
                    usage=completion.usage.model_dump() if completion.usage else None,
                    input_data=item,
                    raw_output=raw_output,
                    reasoning=self._extract_reasoning(completion),
                ) # pyright: ignore[reportReturnType]
            except Exception as e:
                last_error = str(e)
                response_obj = getattr(e, "response", None)
                response_body = getattr(e, "body", None)
                raw_candidate = response_body if response_body is not None else response_obj
                if raw_candidate is None:
                    raw_candidate = {"exception": type(e).__name__, "message": str(e)}
                last_raw_output = self._normalize_raw_output(raw_candidate)

        return TaskResult(
            is_success=False,
            error=last_error,
            input_data=item,
            raw_output=last_raw_output,
        )

    def _normalize_single_input(self, item: str | Mapping[str, Any]) -> PromptContext:
        if isinstance(item, str):
            return {"text": item}
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError("Each input item must be str or mapping[str, Any].")

    def _normalize_batch_inputs(self, sequence: Any) -> list[PromptContext]:
        # Pandas DataFrame: each row is one context dictionary
        if hasattr(sequence, "to_dict") and hasattr(sequence, "columns"):
            records = sequence.to_dict(orient="records")
            return [self._normalize_single_input(record) for record in records]

        # Keep legacy behavior for list[str] / tuple[str]
        if isinstance(sequence, (list, tuple)) and (len(sequence) == 0 or isinstance(sequence[0], str)):
            return [self._normalize_single_input(item) for item in sequence]

        return [self._normalize_single_input(item) for item in sequence]

    def _build_run_id(self) -> str:
        prompt_source = self._user_prompt_source
        if self._system_prompt_source:
            prompt_source = f"{self._system_prompt_source}\n{prompt_source}"
        sig = f"{self.model}_{prompt_source}".encode("utf-8")
        return f"auto_{hashlib.md5(sig).hexdigest()[:8]}"

    @overload
    def run_one(self, item: str) -> TaskResult[TOutput]:
        ...

    @overload
    def run_one(self, item: Mapping[str, Any]) -> TaskResult[TOutput]:
        ...

    def run_one(self, item: str | Mapping[str, Any]) -> TaskResult[TOutput]:
        """执行单条输入并返回底层 TaskResult。"""
        input_item = self._normalize_single_input(item)
        return self._execute_single(input_item)

    @overload
    def map(
        self,
        sequence: Sequence[str],
        db_path: str = ".silkloom_cache.db",
        run_id: Optional[str] = None,
        workers: int = 5,
    ) -> ResultSet[TOutput]:
        ...

    @overload
    def map(
        self,
        sequence: Iterable[Mapping[str, Any]],
        db_path: str = ".silkloom_cache.db",
        run_id: Optional[str] = None,
        workers: int = 5,
    ) -> ResultSet[TOutput]:
        ...

    @overload
    def map(
        self,
        sequence: "pd.DataFrame",
        db_path: str = ".silkloom_cache.db",
        run_id: Optional[str] = None,
        workers: int = 5,
    ) -> ResultSet[TOutput]:
        ...

    def map(
        self,
        sequence: Any,
        db_path: str = ".silkloom_cache.db",
        run_id: Optional[str] = None,
        workers: int = 5,
    ) -> ResultSet[TOutput]:
        """将任务映射到数据序列，执行并发处理。

        支持三类常见输入：
        - pandas DataFrame: 每行转换为一个输入字典，列名作为模板变量
        - 字符串列表: 自动包装为 {"text": ...}
        - 字典序列: 原样作为模板上下文
        """
        
        # 1. 统一输入归一化（DataFrame / 字符串序列 / 字典序列）
        inputs = self._normalize_batch_inputs(sequence)
            
        # 2. 自动生成 run_id
        if not run_id:
            run_id = self._build_run_id()
            
        # 3. 驱动底层并发与数据库状态机
        from .engine import _run_batch_engine
        raw_results = _run_batch_engine(self, inputs, db_path, run_id, workers)

        return ResultSet(raw_results, run_id)
