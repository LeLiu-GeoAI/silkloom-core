from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Iterator, Sequence, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    id: str
    ok: bool
    value: T | None = None
    error: str | None = None
    input: dict[str, Any]
    output: str | None = None
    reasoning: str | None = None
    cache_hit: bool = False
    attempts: int = 1

    @property
    def failed(self) -> bool:
        return not self.ok

    def unwrap(self) -> T:
        if not self.ok:
            raise RuntimeError(self.error or "SilkLoom run failed")
        return self.value  # type: ignore[return-value]


@dataclass(frozen=True)
class Batch(Generic[T]):
    results: Sequence[Result[T]]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[Result[T]]:
        return iter(self.results)

    def __getitem__(self, index: int) -> Result[T]:
        return self.results[index]

    def values(self) -> list[T]:
        return [item.unwrap() for item in self.results]

    def successful(self) -> list[Result[T]]:
        return [item for item in self.results if item.ok]

    def failed(self) -> list[Result[T]]:
        return [item for item in self.results if not item.ok]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.results]

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install silkloom-core[data] to use Batch.to_pandas().") from exc
        return pd.DataFrame(self.to_dicts())


@dataclass(frozen=True)
class RunConfig:
    namespace: str | None = None
    concurrency: int = 5
    ordered: bool = True
