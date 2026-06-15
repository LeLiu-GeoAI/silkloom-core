from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import json_repair
import pandas as pd
from jinja2 import StrictUndefined, Template
from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = "Please output valid JSON only."


def encode_image_to_base64(path: str | Path) -> str:
    with Path(path).open("rb") as image:
        return base64.b64encode(image.read()).decode("ascii")


def image_to_data_url(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime, _ = mimetypes.guess_type(image_path.as_posix())
    return f"data:{mime or 'application/octet-stream'};base64,{encode_image_to_base64(image_path)}"


class SQLiteCache:
    def __init__(self, path: str | Path = ".llm_cache.db"):
        self.path = Path(path)
        self._ensure_schema()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE cache_key = ? AND ok = 1",
                (key,),
            ).fetchone()
        return row[0] if row else None

    def put(
        self,
        key: str,
        *,
        request: dict[str, Any],
        response: str | None = None,
        parsed: dict[str, Any] | None = None,
        error: str | None = None,
        ok: bool = False,
        attempts: int = 0,
    ) -> None:
        params = {name: value for name, value in request.items() if name not in {"model", "messages"}}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache (
                    cache_key,
                    ok,
                    model,
                    messages_json,
                    params_json,
                    request_json,
                    response,
                    parsed_json,
                    error,
                    attempts,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(cache_key) DO UPDATE SET
                    ok = excluded.ok,
                    model = excluded.model,
                    messages_json = excluded.messages_json,
                    params_json = excluded.params_json,
                    request_json = excluded.request_json,
                    response = excluded.response,
                    parsed_json = excluded.parsed_json,
                    error = excluded.error,
                    attempts = excluded.attempts,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    key,
                    int(ok),
                    request.get("model"),
                    self._json(request.get("messages")),
                    self._json(params),
                    self._json(request),
                    response or "",
                    self._json(parsed),
                    error,
                    attempts,
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    ok INTEGER NOT NULL DEFAULT 1,
                    model TEXT,
                    messages_json TEXT,
                    params_json TEXT,
                    request_json TEXT,
                    parsed_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            existing = {row[1] for row in conn.execute("PRAGMA table_info(cache)")}
            columns = {
                "ok": "INTEGER NOT NULL DEFAULT 1",
                "model": "TEXT",
                "messages_json": "TEXT",
                "params_json": "TEXT",
                "request_json": "TEXT",
                "parsed_json": "TEXT",
                "error": "TEXT",
                "attempts": "INTEGER NOT NULL DEFAULT 0",
                "updated_at": "TEXT",
            }
            for name, definition in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE cache ADD COLUMN {name} {definition}")

    def _json(self, value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


@pd.api.extensions.register_dataframe_accessor("llm")
class PandasLLMAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj
        self._client: Any | None = None
        self._cache = SQLiteCache()
        self._cancel_event = threading.Event()

    def setup(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        cache_path: str | Path = ".llm_cache.db",
        client: Any | None = None,
        **client_options: Any,
    ) -> PandasLLMAccessor:
        self._client = client or OpenAI(api_key=api_key, base_url=base_url, **client_options)
        self._cache = SQLiteCache(cache_path)
        return self

    def cancel(self) -> None:
        self._cancel_event.set()

    def extract(
        self,
        prompt_template: str,
        *,
        image_column: str | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        model: str = "gpt-3.5-turbo",
        max_workers: int = 4,
        json_mode: bool = False,
        max_retries: int = 2,
        progress_callback: Callable[[int, int], None] | None = None,
        verbose: bool = True,
        **request_options: Any,
    ) -> pd.DataFrame:
        if self._client is None:
            self._client = OpenAI()

        if image_column is not None and image_column not in self._obj.columns:
            raise KeyError(f"Image column not found: {image_column}")

        template = Template(prompt_template, undefined=StrictUndefined)

        self._cancel_event.clear()
        total = len(self._obj)
        completed = 0
        results: dict[Any, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = {
                pool.submit(
                    self._process_row,
                    row,
                    template,
                    image_column,
                    system_prompt,
                    model,
                    json_mode,
                    max_retries,
                    request_options,
                ): index
                for index, row in self._obj.iterrows()
            }

            iterator = as_completed(futures)
            bar = self._progress_bar(verbose, total)
            try:
                for future in iterator:
                    if self._cancel_event.is_set():
                        pool.shutdown(wait=False, cancel_futures=True)
                        break

                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        results[index] = {"_llm_error": f"System Error: {exc}"}

                    completed += 1
                    if bar is not None:
                        bar.update(1)
                    if progress_callback is not None:
                        progress_callback(completed, total)
            except KeyboardInterrupt:
                self._cancel_event.set()
                pool.shutdown(wait=False, cancel_futures=True)
            finally:
                if bar is not None:
                    bar.close()

        return pd.DataFrame.from_dict(results, orient="index").reindex(self._obj.index)

    def _process_row(
        self,
        row: pd.Series,
        prompt_template: Template,
        image_column: str | None,
        system_prompt: str | None,
        model: str,
        json_mode: bool,
        max_retries: int,
        request_options: dict[str, Any],
    ) -> dict[str, Any]:
        if self._cancel_event.is_set():
            return {"_llm_error": "Cancelled by user"}

        try:
            user_prompt = prompt_template.render(**row.to_dict())
            messages = self._messages(row, user_prompt, image_column, system_prompt)
        except Exception as exc:
            return {"_llm_error": str(exc)}

        request = {"model": model, "messages": messages, **request_options}
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        cache_key = self._cache_key(request)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._parse_content(cached)

        last_error: str | None = None
        for attempt in range(max_retries + 1):
            if self._cancel_event.is_set():
                return {"_llm_error": "Cancelled by user"}

            try:
                response = self._client.chat.completions.create(**request)
                raw_content = self._response_text(response)
                parsed = self._parse_content(raw_content)
                self._cache.put(
                    cache_key,
                    request=request,
                    response=raw_content,
                    parsed=parsed,
                    error=parsed.get("_llm_error"),
                    ok="_llm_error" not in parsed,
                    attempts=attempt + 1,
                )
                return parsed
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(2**attempt)

        error_result = {"_llm_error": f"Failed after {max_retries} retries. Error: {last_error}"}
        self._cache.put(
            cache_key,
            request=request,
            parsed=error_result,
            error=error_result["_llm_error"],
            ok=False,
            attempts=max_retries + 1,
        )
        return error_result

    def _messages(
        self,
        row: pd.Series,
        user_prompt: str,
        image_column: str | None,
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        content: str | list[dict[str, Any]] = user_prompt
        if image_column is not None and not self._is_missing(row[image_column]):
            content = [{"type": "text", "text": user_prompt}]
            for image in self._image_values(row[image_column]):
                content.append({"type": "image_url", "image_url": {"url": self._normalize_image(image)}})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _image_values(self, value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if not self._is_missing(item)]
        return [str(value)]

    def _normalize_image(self, value: str) -> str:
        lower = value.lower()
        if lower.startswith(("http://", "https://", "data:image/")):
            return value
        return image_to_data_url(value)

    def _parse_content(self, raw_content: str) -> dict[str, Any]:
        try:
            parsed = json_repair.loads(raw_content)
        except Exception as exc:
            return {"_llm_error": f"Parse Error: {exc}", "_raw_content": raw_content}

        if isinstance(parsed, dict):
            return parsed
        if parsed == "" and raw_content.strip():
            return {"_llm_raw": raw_content}
        return {"_llm_raw": parsed}

    def _cache_key(self, request: dict[str, Any]) -> str:
        payload = json.dumps(request, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _response_text(self, response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _progress_bar(self, verbose: bool, total: int) -> Any | None:
        if not verbose:
            return None
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        return tqdm(total=total, desc="LLM Inference")

    def _is_missing(self, value: Any) -> bool:
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False
