from __future__ import annotations

from typing import Any, Generic, Iterator, List, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class TaskResult(BaseModel, Generic[T]):
    task_id: str
    is_success: bool
    data: T | None
    error: str | None
    input_data: dict[str, Any]
    raw_output: str | None
    reasoning: str | None
    cached: bool = False


class BatchResult(Generic[T]):
    def __init__(self, results: List[TaskResult[T]]):
        self.results = results

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[TaskResult[T]]:
        return iter(self.results)

    def __getitem__(self, index: int) -> TaskResult[T]:
        return self.results[index]

    def successful(self) -> List[TaskResult[T]]:
        return [item for item in self.results if item.is_success]

    def failed(self) -> List[TaskResult[T]]:
        return [item for item in self.results if not item.is_success]

    def to_dicts(self) -> List[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.results]

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
