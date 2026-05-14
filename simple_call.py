from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from silkloom_core import TaskLoom


def load_env_file(path: str = ".env") -> None:
    """Minimal .env loader to avoid extra dependencies."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


class TextAnalysis(BaseModel):
    sentiment: str
    keywords: list[str]


def print_section(title: str) -> None:
    print("\n" + "=" * 18 + f" {title} " + "=" * 18)


def print_task_result(tag: str, result: Any) -> None:
    print(f"[{tag}] is_success={result.is_success}")
    if result.is_success:
        print(f"[{tag}] data={result.data}")
    else:
        print(f"[{tag}] error={result.error}")
    print(f"[{tag}] raw_output={result.raw_output}")
    print(f"[{tag}] reasoning={result.reasoning}")


def test_process_raw(client: OpenAI) -> None:
    print_section("process: raw text")
    loom = TaskLoom(
        model="glm-4-flash",
        prompt_template="请将这句话改写为更正式的学术语气：{{ text }}",
        client=client,
        temperature=0.3,
    )
    result = loom.process("我们发现这个方法挺好用的。")
    print_task_result("process_raw", result)


def test_process_dict_json(client: OpenAI) -> None:
    print_section("process: dict JSON")
    loom = TaskLoom(
        model="glm-4-flash",
        prompt_template=(
            "请分析文本并严格返回 JSON 对象，键必须是 sentiment 和 keywords。"
            "不要输出任何额外文字。文本：{{ text }}"
        ),
        response_model=dict,
        client=client,
        temperature=0.1,
    )
    result = loom.process({"text": "这篇论文结构清晰，但实验部分略显不足。"})
    print_task_result("process_dict_json", result)


def test_process_pydantic(client: OpenAI) -> None:
    print_section("process: pydantic")
    loom = TaskLoom(
        model="glm-4-flash",
        prompt_template=(
            "请分析文本并严格返回 JSON 对象，字段 sentiment 为字符串，"
            "keywords 为字符串数组。不要输出任何额外文字。文本：{{ text }}"
        ),
        response_model=TextAnalysis,
        client=client,
        temperature=0.1,
    )
    result = loom.process({"text": "方法有效，但计算成本较高，部署受限。"})
    print_task_result("process_pydantic", result)


async def test_aprocess_raw(client: OpenAI) -> None:
    print_section("aprocess: raw text")
    loom = TaskLoom(
        model="glm-4-flash",
        prompt_template="请把这句话改写成更规范的书面语：{{ text }}",
        client=client,
        temperature=0.3,
    )
    result = await loom.aprocess("该方案挺稳的，效果也不错。")
    print_task_result("aprocess_raw", result)


def test_map_and_cache(client: OpenAI) -> None:
    print_section("map + cache")
    db_path = ".silkloom.test.db"
    run_id = "map_interface_test_v1"
    sequence = [
        "第一条：请改写为正式语气。",
        "第二条：请改写为正式语气。",
        {"text": "第三条：请改写为正式语气。"},
    ]

    loom = TaskLoom(
        model="glm-4-flash",
        prompt_template="{{ text }}",
        client=client,
        temperature=0.2,
    )

    t1 = time.perf_counter()
    first = loom.map(sequence, db_path=db_path, run_id=run_id, workers=3)
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    second = loom.map(sequence, db_path=db_path, run_id=run_id, workers=3)
    t4 = time.perf_counter()

    print(f"[map] first_run_count={len(first)}, success={len(first.successful())}, failed={len(first.failed())}")
    print(f"[map] second_run_count={len(second)}, success={len(second.successful())}, failed={len(second.failed())}")
    print(f"[map] first_run_seconds={t2 - t1:.2f}, second_run_seconds={t4 - t3:.2f}")
    print(f"[map] to_dicts_sample={first.to_dicts()[0] if len(first) else None}")

    try:
        df = first.to_pandas()
        print(f"[map] to_pandas_shape={df.shape}")
    except Exception as exc:
        print(f"[map] to_pandas skipped: {exc}")


async def test_amap_and_cache(client: OpenAI) -> None:
    print_section("amap + cache")
    db_path = ".silkloom.test.async.db"
    run_id = "amap_interface_test_v1"
    sequence = [
        "A1: 转为更正式语气",
        "A2: 转为更正式语气",
        {"text": "A3: 转为更正式语气"},
    ]

    loom = TaskLoom(
        model="glm-4-flash",
        prompt_template="{{ text }}",
        client=client,
        temperature=0.2,
    )

    t1 = time.perf_counter()
    first = await loom.amap(sequence, db_path=db_path, run_id=run_id, max_concurrent=3)
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    second = await loom.amap(sequence, db_path=db_path, run_id=run_id, max_concurrent=3)
    t4 = time.perf_counter()

    print(f"[amap] first_run_count={len(first)}, success={len(first.successful())}, failed={len(first.failed())}")
    print(f"[amap] second_run_count={len(second)}, success={len(second.successful())}, failed={len(second.failed())}")
    print(f"[amap] first_run_seconds={t2 - t1:.2f}, second_run_seconds={t4 - t3:.2f}")


def test_image_protocol(client: OpenAI) -> None:
    print_section("process: image protocol (best effort)")
    loom = TaskLoom(
        model="glm-4v-flash",
        prompt_template="请简要描述图片内容。要求：{{ instruction }}",
        response_model=dict,
        client=client,
        temperature=0.1,
    )
    result = loom.process(
        {
            "instruction": "返回 JSON，包含 scene 字段。",
            "images": [
                "https://images.unsplash.com/photo-1469474968028-56623f02e42e",
            ],
        }
    )
    print_task_result("process_image", result)


async def run_all_tests(client: OpenAI) -> None:
    test_process_raw(client)
    test_process_dict_json(client)
    test_process_pydantic(client)
    await test_aprocess_raw(client)
    test_map_and_cache(client)
    await test_amap_and_cache(client)

    # 图像测试受模型和供应商能力影响较大，失败不影响其余接口验证。
    try:
        test_image_protocol(client)
    except Exception as exc:
        print(f"[process_image] skipped or failed: {exc}")


def main() -> None:
    load_env_file()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Please set it in .env")
    if not base_url:
        raise RuntimeError("Missing BASE_URL. Please set it in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)
    asyncio.run(run_all_tests(client))


if __name__ == "__main__":
    main()
