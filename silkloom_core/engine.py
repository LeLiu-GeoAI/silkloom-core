from __future__ import annotations

import sqlite3
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from .types import TaskResult


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
                UNIQUE(run_id, item_index)
            )
            """
        )


def _run_batch_engine(
    task_instance: Any, 
    inputs: List[Dict[str, Any]], 
    db_path: str,
    run_id: str, 
    workers: int
) -> List[TaskResult]:
    
    _init_db(db_path)
    
    # 建立数据库连接 (多线程共享连接时需开启 check_same_thread=False)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # 1. 同步全量任务到数据库，并获取当前状态
    pending_tasks = []  # 记录需要跑的任务：(index, item)
    results_map = {}    # 记录每个 index 最终的 TaskResult

    with _db_lock:
        for idx, item in enumerate(inputs):
            cursor = conn.execute(
                "SELECT status, result_data, error_msg, usage_data FROM tasks WHERE run_id=? AND item_index=?", 
                (run_id, idx),
            )
            row = cursor.fetchone()
            
            if row:
                status, res_data, err_msg, usage_data = row
                if status == "success":
                    # 已经成功的数据，直接反序列化恢复，跳过执行
                    parsed_data = json.loads(res_data)
                    # 如果配置了 response_model，需将其转回 Pydantic 实例
                    if task_instance.response_model and isinstance(parsed_data, dict):
                        parsed_data = task_instance.response_model.model_validate(parsed_data)
                        
                    results_map[idx] = TaskResult(
                        is_success=True, 
                        data=parsed_data, 
                        usage=json.loads(usage_data) if usage_data else None,
                        input_data=item,
                    )
                else:
                    # 历史失败/中断任务在新一轮运行时显式重置为 pending，确保会重试
                    conn.execute(
                        "UPDATE tasks SET status=?, result_data=?, error_msg=?, usage_data=? WHERE run_id=? AND item_index=?",
                        ("pending", None, None, None, run_id, idx),
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

    # 2. 并发执行待处理的任务
    def worker(idx: int, item: dict):
        # 调用 LLMTask 内部的方法发起网络请求
        res = task_instance._execute_single(item)
        
        # 将结果原子化写入数据库
        status = "success" if res.is_success else "failed"
        res_json = json.dumps(res.data.model_dump() if hasattr(res.data, "model_dump") else res.data, ensure_ascii=False) if res.data else None
        usage_json = json.dumps(res.usage) if res.usage else None

        with _db_lock:
            conn.execute(
                "UPDATE tasks SET status=?, result_data=?, error_msg=?, usage_data=? WHERE run_id=? AND item_index=?",
                (status, res_json, res.error, usage_json, run_id, idx)
            )
            conn.commit()
            
        return idx, res

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, idx, item): idx for idx, item in pending_tasks}
        for future in as_completed(futures):
            idx, res = future.result()
            results_map[idx] = res

    conn.close()

    # 3. 按原始输入的顺序重新组合列表并返回
    final_results = [results_map[i] for i in range(len(inputs))]
    return final_results
