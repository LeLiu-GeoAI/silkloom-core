from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class TaskResult:
    """单次大模型任务执行的结果容器"""

    is_success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    input_data: Optional[Dict[str, Any]] = None
