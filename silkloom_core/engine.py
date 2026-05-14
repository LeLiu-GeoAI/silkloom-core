from __future__ import annotations

import sqlite3
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Protocol, TypeVar

from .types import TaskResult

TOutput = TypeVar("TOutput")


class CacheManager:
    """用于管理 SilkLoom 批处理缓存的工具类。
    
    支持查看缓存状态、清空缓存、回滚已完成的任务等操作。
    """
    
    def __init__(self, db_path: str = ".silkloom_cache.db"):
        self.db_path = db_path
    
    def inspect(self, run_id: Optional[str] = None) -> dict[str, Any]:
        """查看缓存状态。
        
        Args:
            run_id: 运行ID，若为 None 则列出所有运行的统计信息
            
        Returns:
            包含任务数、成功/失败统计、最后更新时间等信息的字典
        """
        try:
            # 确保数据库已初始化
            _init_db(self.db_path)
            
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            if run_id:
                # 查看特定 run_id 的统计
                cursor.execute(
                    "SELECT COUNT(*), COUNT(CASE WHEN status='success' THEN 1 END), "
                    "COUNT(CASE WHEN status='failed' THEN 1 END) FROM tasks WHERE run_id=?",
                    (run_id,)
                )
                total, success, failed = cursor.fetchone() or (0, 0, 0)
                return {
                    "run_id": run_id,
                    "total_tasks": total,
                    "successful": success,
                    "failed": failed,
                    "pending": total - success - failed
                }
            else:
                # 列出所有 run_id 的统计
                cursor.execute("SELECT DISTINCT run_id FROM tasks ORDER BY run_id DESC")
                runs = cursor.fetchall()
                summary = []
                for (rid,) in runs:
                    cursor.execute(
                        "SELECT COUNT(*), COUNT(CASE WHEN status='success' THEN 1 END), "
                        "COUNT(CASE WHEN status='failed' THEN 1 END) FROM tasks WHERE run_id=?",
                        (rid,)
                    )
                    total, success, failed = cursor.fetchone() or (0, 0, 0)
                    summary.append({
                        "run_id": rid,
                        "total_tasks": total,
                        "successful": success,
                        "failed": failed,
                        "pending": total - success - failed
                    })
                return {"run_summaries": summary}
        finally:
            conn.close()
    
    def clear(self, run_id: Optional[str] = None, confirm: bool = False) -> None:
        """清空缓存。
        
        Args:
            run_id: 若指定则仅删除该运行的任务，若为 None 则清空整个数据库
            confirm: 安全确认，必须为 True 才会执行清空
        """
        if not confirm:
            raise ValueError("缓存清空操作需要显式确认: clear(run_id=..., confirm=True)")
        
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            if run_id:
                conn.execute("DELETE FROM tasks WHERE run_id=?", (run_id,))
                print(f"✓ 已清空 run_id='{run_id}' 的缓存")
            else:
                conn.execute("DELETE FROM tasks")
                print("✓ 已清空所有缓存")
            conn.commit()
        finally:
            conn.close()
    
    def rollback(self, run_id: str, confirm: bool = False) -> None:
        """将已完成的任务标记为待处理状态，用于重新运行。
        
        Args:
            run_id: 目标运行ID
            confirm: 安全确认
        """
        if not confirm:
            raise ValueError("回滚操作需要显式确认: rollback(run_id=..., confirm=True)")
        
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute(
                "UPDATE tasks SET status='pending', result_data=NULL, error_msg=NULL "
                "WHERE run_id=? AND status IN ('success', 'failed')",
                (run_id,)
            )
            conn.commit()
            affected = conn.total_changes
            print(f"✓ 已回滚 {affected} 个任务为待处理状态")
        finally:
            conn.close()


class _TaskInstanceProtocol(Protocol[TOutput]):
    response_model: Any

    def _execute_single(self, item: dict[str, Any]) -> TaskResult[TOutput]:
        ...
    
    async def _execute_single_async(self, item: dict[str, Any]) -> TaskResult[TOutput]:
        ...


# 使用锁保证 SQLite 多线程写入安全
_db_lock = threading.Lock()


def _init_db(db_path: str):
    """初始化 SQLite 数据库表结构"""
    with _db_lock, sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                item_index INTEGER,
                status TEXT,
                input_data TEXT,
                result_data TEXT,
                error_msg TEXT,
                usage_data TEXT,
                raw_output_data TEXT,
                reasoning_data TEXT,
                UNIQUE(run_id, item_index)
            )
            """
        )

        # 为老版本数据库补齐新增字段，保证向后兼容
        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "raw_output_data" not in existing_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN raw_output_data TEXT")
        if "reasoning_data" not in existing_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN reasoning_data TEXT")


def _run_batch_engine(
    task_instance: _TaskInstanceProtocol[TOutput], 
    inputs: list[dict[str, Any]], 
    db_path: str,
    run_id: str, 
    workers: int,
    show_progress: bool = True,
) -> list[TaskResult[TOutput]]:
    
    _init_db(db_path)
    
    # 建立数据库连接 (多线程共享连接时需开启 check_same_thread=False)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # 1. 同步全量任务到数据库，并获取当前状态
    pending_tasks = []  # 记录需要跑的任务：(index, item)
    results_map = {}    # 记录每个 index 最终的 TaskResult

    with _db_lock:
        for idx, item in enumerate(inputs):
            cursor = conn.execute(
                "SELECT status, result_data, error_msg, usage_data, raw_output_data, reasoning_data FROM tasks WHERE run_id=? AND item_index=?", 
                (run_id, idx),
            )
            row = cursor.fetchone()
            
            if row:
                status, res_data, err_msg, usage_data, raw_output_data, reasoning_data = row
                if status == "success":
                    # 已经成功的数据，直接反序列化恢复，跳过执行
                    parsed_data = json.loads(res_data) if res_data is not None else None
                    # 如果配置了 response_model，需将其转回 Pydantic 实例
                    if task_instance.response_model and isinstance(parsed_data, dict):
                        parsed_data = task_instance.response_model.model_validate(parsed_data)
                        
                    results_map[idx] = TaskResult(
                        is_success=True, 
                        data=parsed_data, 
                        usage=json.loads(usage_data) if usage_data else None,
                        input_data=item,
                        raw_output=json.loads(raw_output_data) if raw_output_data else None,
                        reasoning=reasoning_data,
                    )
                else:
                    # 历史失败/中断任务在新一轮运行时显式重置为 pending，确保会重试
                    conn.execute(
                        "UPDATE tasks SET status=?, result_data=?, error_msg=?, usage_data=?, raw_output_data=?, reasoning_data=? WHERE run_id=? AND item_index=?",
                        ("pending", None, None, None, None, None, run_id, idx),
                    )
                    pending_tasks.append((idx, item))
            else:
                # 数据库没有记录，插入一条 pending 记录
                conn.execute(
                    "INSERT INTO tasks (run_id, item_index, status, input_data) VALUES (?, ?, ?, ?)",
                    (run_id, idx, "pending", json.dumps(item, ensure_ascii=False))
                )
                pending_tasks.append((idx, item))
        conn.commit()

    def _serialize_result_data(data: Any) -> str | None:
        """Serialize successful result data while preserving falsy values like '', 0, []."""
        if data is None:
            return None
        payload = data.model_dump() if hasattr(data, "model_dump") else data
        return json.dumps(payload, ensure_ascii=False)

    # 2. 并发执行待处理的任务
    def worker(idx: int, item: dict[str, Any]):
        # 调用 PromptMapper 内部的方法发起网络请求
        res = task_instance._execute_single(item)
        
        # 将结果原子化写入数据库
        status = "success" if res.is_success else "failed"
        res_json = _serialize_result_data(res.data)
        usage_json = json.dumps(res.usage) if res.usage else None
        raw_output_json = json.dumps(res.raw_output, ensure_ascii=False) if res.raw_output is not None else None

        with _db_lock:
            conn.execute(
                "UPDATE tasks SET status=?, result_data=?, error_msg=?, usage_data=?, raw_output_data=?, reasoning_data=? WHERE run_id=? AND item_index=?",
                (status, res_json, res.error, usage_json, raw_output_json, res.reasoning, run_id, idx)
            )
            conn.commit()
            
        return idx, res

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, idx, item): idx for idx, item in pending_tasks}
        
        # 尝试使用 tqdm 显示进度条，如果未安装则无声降级
        iterator = as_completed(futures)
        if show_progress and pending_tasks:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(pending_tasks), desc="Processing", unit="item")
            except ImportError:
                pass  # tqdm 未安装，继续使用普通迭代器
        
        for future in iterator:
            idx, res = future.result()
            results_map[idx] = res

    conn.close()

    # 3. 按原始输入的顺序重新组合列表并返回
    final_results = [results_map[i] for i in range(len(inputs))]
    return final_results


async def _run_batch_engine_async(
    task_instance: _TaskInstanceProtocol[TOutput], 
    inputs: list[dict[str, Any]], 
    db_path: str,
    run_id: str, 
    max_concurrent: int = 5,
    show_progress: bool = True,
) -> list[TaskResult[TOutput]]:
    """异步批处理引擎，支持原生 asyncio 集成"""
    def _serialize_result_data(data: Any) -> str | None:
        """Serialize successful result data while preserving falsy values like '', 0, []."""
        if data is None:
            return None
        payload = data.model_dump() if hasattr(data, "model_dump") else data
        return json.dumps(payload, ensure_ascii=False)

    
    _init_db(db_path)
    
    # 建立数据库连接（在单线程 asyncio 模式下使用，需要小心处理）
    loop = asyncio.get_event_loop()
    
    def _get_db_row(idx: int, item: dict[str, Any]) -> tuple:
        """同步函数：从 DB 读取缓存"""
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.execute(
            "SELECT status, result_data, error_msg, usage_data, raw_output_data, reasoning_data FROM tasks WHERE run_id=? AND item_index=?",
            (run_id, idx),
        )
        row = cursor.fetchone()
        conn.close()
        return row
    
    def _save_db(idx: int, status: str, res_json: str, res_error: str, usage_json: str, raw_output_json: str, reasoning: str):
        """同步函数：保存结果到 DB"""
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            "INSERT OR REPLACE INTO tasks (run_id, item_index, status, result_data, error_msg, usage_data, raw_output_data, reasoning_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, idx, status, res_json, res_error, usage_json, raw_output_json, reasoning),
        )
        conn.commit()
        conn.close()
    
    pending_tasks = []  # (index, item, row_data)
    results_map = {}
    
    # 1. 检查缓存（用 executor 在线程池中执行 DB 操作）
    for idx, item in enumerate(inputs):
        row = await loop.run_in_executor(None, _get_db_row, idx, item)
        
        if row:
            status, res_data, err_msg, usage_data, raw_output_data, reasoning_data = row
            if status == "success":
                parsed_data = json.loads(res_data) if res_data is not None else None
                if task_instance.response_model and isinstance(parsed_data, dict):
                    parsed_data = task_instance.response_model.model_validate(parsed_data)
                
                results_map[idx] = TaskResult(
                    is_success=True,
                    data=parsed_data,
                    usage=json.loads(usage_data) if usage_data else None,
                    input_data=item,
                    raw_output=json.loads(raw_output_data) if raw_output_data else None,
                    reasoning=reasoning_data,
                )
                continue
        
        # 需要处理的任务
        pending_tasks.append((idx, item))
    
    # 2. 异步执行待处理任务（用信号量控制并发数）
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def worker_async(idx: int, item: dict[str, Any]):
        async with semaphore:
            res = await task_instance._execute_single_async(item)
            
            # 保存到 DB（用 executor）
            status = "success" if res.is_success else "failed"
            res_json = _serialize_result_data(res.data)
            usage_json = json.dumps(res.usage) if res.usage else None
            raw_output_json = json.dumps(res.raw_output, ensure_ascii=False) if res.raw_output is not None else None
            
            await loop.run_in_executor(
                None,
                _save_db,
                idx,
                status,
                res_json,
                res.error,
                usage_json,
                raw_output_json,
                res.reasoning,
            )
            
            return idx, res
    
    # 3. 并发执行任务并收集结果
    if pending_tasks:
        tasks_coro = [worker_async(idx, item) for idx, item in pending_tasks]
        
        if show_progress:
            try:
                from tqdm.asyncio import tqdm
                results_iter = await tqdm.gather(*tasks_coro, desc="Processing (async)", unit="item")
            except ImportError:
                results_iter = await asyncio.gather(*tasks_coro)
        else:
            results_iter = await asyncio.gather(*tasks_coro)
        
        for idx, res in results_iter:
            results_map[idx] = res
    
    # 4. 按原始顺序返回
    final_results = [results_map[i] for i in range(len(inputs))]
    return final_results
