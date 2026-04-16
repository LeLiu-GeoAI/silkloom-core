from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Generic, Optional, Protocol, TypeVar, overload

import jinja2
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

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


class LLMTask(Generic[TOutput]):
    @overload
    def __init__(
        self: "LLMTask[str]",
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
        self: "LLMTask[TModel]",
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
        """Create an LLM batch task.

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

        for attempt in range(self.max_retries):
            try:
                # 结构化输出模式 (Pydantic)
                if self.response_model:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=self.response_model,
                    )
                    return TaskResult(
                        is_success=True,
                        data=completion.choices[0].message.parsed,
                        usage=completion.usage.model_dump() if completion.usage else None,
                        input_data=item,
                    )
                # 纯文本模式
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                return TaskResult(
                    is_success=True,
                    data=completion.choices[0].message.content,
                    usage=completion.usage.model_dump() if completion.usage else None,
                    input_data=item,
                )
            except Exception as e:
                last_error = str(e)

        return TaskResult(is_success=False, error=last_error, input_data=item)

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
        sequence: pd.DataFrame,
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
        
        # 1. 鸭子类型检测：统一转为 List[dict]
        if isinstance(sequence, pd.DataFrame):
            inputs: list[PromptContext] = sequence.to_dict(orient="records") # Pandas DataFrame
        elif isinstance(sequence, (list, tuple)) and len(sequence) > 0 and isinstance(sequence[0], str):
            inputs = [{"text": item} for item in sequence] # 纯字符串列表
        else:
            inputs = [dict(item) for item in sequence] # 已经是字典列表
            
        # 2. 自动生成 run_id
        if not run_id:
            prompt_source = self._user_prompt_source
            if self._system_prompt_source:
                prompt_source = f"{self._system_prompt_source}\n{prompt_source}"
            sig = f"{self.model}_{prompt_source}".encode("utf-8")
            run_id = f"auto_{hashlib.md5(sig).hexdigest()[:8]}"
            
        # 3. 驱动底层并发与数据库状态机
        from .engine import _run_batch_engine
        raw_results = _run_batch_engine(self, inputs, db_path, run_id, workers)

        return ResultSet(raw_results, run_id)
