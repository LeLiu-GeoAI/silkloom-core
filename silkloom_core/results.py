from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, overload

from .types import TaskResult

TResultData = TypeVar("TResultData")


class ResultSet(Sequence[TaskResult[TResultData]], Generic[TResultData]):
    def __init__(self, raw_results: list[TaskResult[TResultData]], run_id: str):
        self._raw = raw_results
        self._run_id = run_id

    @overload
    def __getitem__(self, index: int) -> TaskResult[TResultData]:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[TaskResult[TResultData]]:
        ...

    def __getitem__(self, index: int | slice) -> TaskResult[TResultData] | list[TaskResult[TResultData]]:
        return self._raw[index]

    def __len__(self) -> int:
        return len(self._raw)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def total_tokens(self) -> int:
        return sum(r.usage.get("total_tokens", 0) for r in self._raw if r.usage)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self._raw if r.is_success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self._raw if not r.is_success)

    @property
    def errors(self) -> list[str]:
        return [r.error for r in self._raw if not r.is_success and r.error]

    @property
    def results(self) -> list[TaskResult[TResultData]]:
        """返回任务级结果列表（每一项包含 data / raw_output / error 等字段）。"""
        return list(self._raw)

    @property
    def outputs(self) -> list[TResultData | None]:
        """返回与输入严格对齐的解析结果列表（成功为 data，失败为 None）。"""
        return [res.data if res.is_success else None for res in self._raw]

    def successful(self) -> list[TaskResult[TResultData]]:
        """返回所有成功任务的结果对象。"""
        return [res for res in self._raw if res.is_success]

    def failed(self) -> list[TaskResult[TResultData]]:
        """返回所有失败任务的结果对象。"""
        return [res for res in self._raw if not res.is_success]

    @property
    def raw_outputs(self) -> list[Any | None]:
        """返回每条输入对应的原始模型输出（成功/失败都保留）"""
        return [r.raw_output for r in self._raw]

    @property
    def reasonings(self) -> list[str | None]:
        """返回每条输入对应的推理内容（若模型提供）"""
        return [r.reasoning for r in self._raw]

    def export_jsonl(self, path: str) -> None:
        """导出为 JSONL 文件"""
        with open(path, "w", encoding="utf-8") as f:
            for res in self._raw:
                if res.is_success:
                    # 如果是 Pydantic 模型，调用 model_dump；如果是字典，直接存
                    data = res.data.model_dump() if hasattr(res.data, "model_dump") else res.data
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def export_csv(self, path: str, flatten: bool = False, include_usage: bool = True) -> None:
        """导出为 CSV 文件，支持一变多数据的打平"""
        rows = []
        for index, res in enumerate(self._raw):
            base_row = {"_input_index": index, "status": "success" if res.is_success else "failed"}
            if include_usage:
                base_row["total_tokens"] = res.usage.get("total_tokens", 0) if res.usage else 0

            if not res.is_success:
                base_row["error"] = res.error
                rows.append(base_row)
                continue

            data = res.data.model_dump() if hasattr(res.data, "model_dump") else res.data

            # 打平逻辑 (Flatten)
            if flatten and isinstance(data, list):
                for item in data:
                    row = base_row.copy()
                    if isinstance(item, dict):
                        row.update(item)
                    else:
                        row["value"] = item
                    rows.append(row)
            else:
                row = base_row.copy()
                if isinstance(data, dict):
                    row.update(data)
                else:
                    row["value"] = data
                rows.append(row)

        if not rows:
            return

        # 提取所有出现过的列名作为表头
        fieldnames = list({key for row in rows for key in row.keys()})
        # 将元数据列排在前面
        meta_keys = ["_input_index", "status", "total_tokens", "error"]
        fieldnames = [k for k in meta_keys if k in fieldnames] + [k for k in fieldnames if k not in meta_keys]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
