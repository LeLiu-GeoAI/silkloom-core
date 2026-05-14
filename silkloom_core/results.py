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

    def filter(self, predicate: callable) -> "ResultSet[TResultData]":
        """按条件过滤结果集，返回新的 ResultSet。
        
        Args:
            predicate: 接收 TaskResult 的函数，返回布尔值
            
        Returns:
            包含符合条件的任务的新 ResultSet
            
        示例:
            successful_results = rs.filter(lambda r: r.is_success)
            high_token_results = rs.filter(lambda r: r.usage and r.usage.get('total_tokens', 0) > 1000)
        """
        filtered = [r for r in self._raw if predicate(r)]
        return ResultSet(filtered, f"{self._run_id}_filtered")

    def map(self, fn: callable) -> list:
        """对每个结果应用函数，返回结果列表。
        
        Args:
            fn: 接收 TaskResult 的函数，返回任意值
            
        Returns:
            函数应用结果的列表
            
        示例:
            tokens_list = rs.map(lambda r: r.usage['total_tokens'] if r.usage else 0)
            success_count = len(rs.map(lambda r: r.data if r.is_success else None))
        """
        return [fn(r) for r in self._raw]

    def transform(self, fn: callable) -> "ResultSet[TResultData]":
        """对结果集中的 data 字段进行转换，返回新的 ResultSet。
        
        Args:
            fn: 接收 data 的函数，返回转换后的值
            
        Returns:
            包含转换后数据的新 ResultSet
            
        示例:
            uppercase_results = rs.transform(lambda data: data.upper() if isinstance(data, str) else data)
        """
        transformed = []
        for r in self._raw:
            if r.is_success:
                try:
                    new_data = fn(r.data)
                    transformed.append(TaskResult(
                        is_success=True,
                        data=new_data,
                        usage=r.usage,
                        input_data=r.input_data,
                        raw_output=r.raw_output,
                        reasoning=r.reasoning
                    ))
                except Exception as e:
                    transformed.append(TaskResult(
                        is_success=False,
                        error=f"Transform error: {e}",
                        input_data=r.input_data
                    ))
            else:
                transformed.append(r)
        return ResultSet(transformed, f"{self._run_id}_transformed")

    def to_dicts(self, merge_input: bool = True) -> list[dict]:
        """将结果转换为纯字典列表，不依赖任何第三方库。
        
        Args:
            merge_input: 是否合并原始输入数据与输出数据到同一行。
        
        Returns:
            字典列表，按输入顺序对齐。成功结果包含 .model_dump() 的内容；失败结果包含 error 字段。
        """
        rows = []
        for task in self._raw:
            row = {}
            
            # 1. 合并输入数据（可选）
            if merge_input and isinstance(task.input_data, dict):
                row.update(task.input_data)
            
            # 2. 合并输出数据
            if task.is_success:
                if hasattr(task.data, "model_dump"):  # Pydantic V2
                    row.update(task.data.model_dump())
                elif hasattr(task.data, "dict"):      # Pydantic V1 兼容
                    row.update(task.data.dict())
                elif isinstance(task.data, dict):
                    row.update(task.data)
                else:
                    row["output"] = task.data  # 纯文本或其他类型
            else:
                row["error"] = str(task.error) if task.error else "Unknown error"
            
            # 3. 状态标志
            row["is_success"] = task.is_success
            rows.append(row)
        
        return rows

    def to_pandas(self, merge_input: bool = True):
        """导出为 Pandas DataFrame。
        
        调用此方法需要用户环境中已安装 pandas。
        如果未安装，会抛出有帮助的错误提示。
        
        Args:
            merge_input: 是否合并原始输入数据与输出数据到同列。
        
        Returns:
            pd.DataFrame，每行对应一条输入，列包括输入字段、输出字段、is_success。
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "调用 to_pandas() 需要安装 pandas。请运行: pip install pandas"
            )
        
        dicts = self.to_dicts(merge_input=merge_input)
        return pd.DataFrame(dicts)

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
