from __future__ import annotations

from typing import Any, Generic, Iterator, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class TaskResult(BaseModel, Generic[T]):
    is_success: bool
    data: T | None
    error: str | None
    input_data: dict[str, Any]
    raw_output: str | None
    reasoning: str | None


class BatchResult(Generic[T]):
    def __init__(self, items: list[TaskResult[T]]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[TaskResult[T]]:
        return iter(self._items)

    def __getitem__(self, index: int) -> TaskResult[T]:
        return self._items[index]

    def successful(self) -> list[TaskResult[T]]:
        return [item for item in self._items if item.is_success]

    def failed(self) -> list[TaskResult[T]]:
        return [item for item in self._items if not item.is_success]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self._items]

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required for to_pandas(), install silkloom-core[data]") from exc
        return pd.DataFrame(self.to_dicts())


class _CacheRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    input_hash: str
    result_json: str
