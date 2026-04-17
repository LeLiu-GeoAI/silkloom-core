from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

TTaskData = TypeVar("TTaskData")


@dataclass
class TaskResult(Generic[TTaskData]):
    """单次大模型任务执行的结果容器"""

    is_success: bool
    data: Optional[TTaskData] = None
    error: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    input_data: Optional[dict[str, Any]] = None
    raw_output: Optional[Any] = None
    reasoning: Optional[str] = None
