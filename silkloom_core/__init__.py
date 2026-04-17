from __future__ import annotations

"""
SilkLoom Core V4.1
极简、带状态持久化的大模型批处理引擎。
"""

from .prompt_mapper import PromptMapper
from .results import ResultSet
from .types import TaskResult

__all__ = [
	"PromptMapper",
	"ResultSet",
	"TaskResult",
]
