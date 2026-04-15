# SilkLoom Core

[中文](README.zh-CN.md) | [English](README.md)

SilkLoom Core 是一个轻量、可恢复的工作流执行工具，适用于可重复执行的 LLM 与函数流水线。

## 简介

SilkLoom Core 本质上是一个通用的批处理执行层，适合把同一流程反复跑在大量输入上，并保证失败可重试、任务可续跑。

核心能力：

- 节点化流程编排（`LLMNode`、`FunctionNode`、自定义 `BaseNode`）
- 支持 `add_node(..., depends_on=[...])` 显式 DAG 分支与汇合
- 支持 `add_collect_node(...)` 在同一 run 内做跨输入聚合
- 并发执行
- 内置 Python 编排能力，减少依赖并便于个人维护
- 自动重试（指数退避）
- SQLite 持久化与断点续跑（`run_id`）
- 支持通过 `get_run_artifacts` 读取聚合产物
- 内置 `tqdm` 进度条与阶段提示
- 结构化输出（Pydantic）

设计理念：

- 重点是可重复执行，而不是智能调度
- 工作流逻辑尽量显式、可控、可复现
- 面向长批次任务的可恢复与可观测
- 内部保持紧凑与显式，优先长期可读性

## 安装

```bash
pip install silkloom-core
```

源码安装：

```bash
git clone https://github.com/your-org/silkloom-core.git
cd silkloom-core
pip install -e .
```

开发依赖：

```bash
pip install -e ".[dev]"
```

## 快速开始

```python
from silkloom_core import Pipeline, LLMNode, FunctionNode


def score_text(text: str) -> dict:
    score = min(len(text) / 100, 1.0)
    return {"score": round(score, 3)}


pipeline = Pipeline(db_path="pipeline.db", execution_mode="depth_first", default_workers=4)

pipeline.add_node(
    LLMNode(
        name="summarize",
        prompt_template="请用一句话总结: {input.text}",
        model="gpt-4o-mini",
    ),
    depends_on=[],
)

pipeline.add_node(
    FunctionNode(
        name="score",
        func=score_text,
        kwargs_mapping={"text": "{summarize.text}"},
    ),
    depends_on=["summarize"],
)

run_id = pipeline.run([
    {"text": "SilkLoom Core 可做科研文本的批处理。"},
    {"text": "它支持 SQLite 持久化和 run_id 续跑。"},
])

print(pipeline.export_results(run_id))
```

## OpenAI 兼容接口示例

`LLMNode` 支持注入自定义 OpenAI 客户端：

```python
LLMNode(..., client=your_openai_client)
```

只要服务兼容 OpenAI Chat Completions，就可以接入。

### 1) 官方 OpenAI

```python
from silkloom_core import LLMNode

node = LLMNode(
    name="extract",
    prompt_template="提取关键信息: {input.note}",
    model="gpt-4o-mini",
)
```

```bash
export OPENAI_API_KEY="your_openai_key"
# PowerShell:
# $env:OPENAI_API_KEY="your_openai_key"
```

### 2) GLM-4-Flash（OpenAI 兼容）

```python
import os
from openai import OpenAI
from silkloom_core import LLMNode

glm_client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

node = LLMNode(
    name="extract_geo",
    prompt_template="从文本中提取城市、主题和坐标: {input.note}",
    model="glm-4-flash",
    client=glm_client,
)
```

```bash
export ZHIPUAI_API_KEY="your_glm_key"
# PowerShell:
# $env:ZHIPUAI_API_KEY="your_glm_key"
```

### 3) 本地 Ollama（OpenAI 兼容）

先启动 Ollama 并拉取模型（示例）：

```bash
ollama pull qwen2.5:7b
ollama serve
```

在 SilkLoom Core 中接入：

```python
from openai import OpenAI
from silkloom_core import LLMNode

ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

node = LLMNode(
    name="local_summary",
    prompt_template="总结以下调研文本: {input.note}",
    model="qwen2.5:7b",
    client=ollama_client,
)
```

说明：本地模型结构化输出能力存在差异。若使用 `response_model`，建议在提示词中明确“仅输出 JSON 对象”。

## 示例脚本

仓库里的示例当前以 GIS/城市研究为案例，但 SilkLoom Core 本身不限定领域，可替换为任意业务场景。

```bash
python examples/quickstart.py
python examples/structured_output.py
python examples/resume_with_run_id.py
python examples/trajectory_od_commute.py
```

- quickstart.py：总结笔记并打主题标签
- structured_output.py：抽取结构化属性并生成 GeoJSON-like 要素
- resume_with_run_id.py：模拟可重复瓦片处理与续跑
- trajectory_od_commute.py：OD 抽取 + 距离/时间分段 + 流线输出

## 核心概念

### 0. 编排边界

- SilkLoom Core 直接负责任务编排与并发调度
- 对外 API 保持紧凑：节点 API、SQLite 持久化、run_id 续跑与导出接口
- 默认本地运行，不依赖外部编排服务

调用 `run()` 时，默认会在控制台打印工作流提示并显示 `tqdm` 进度条。

- `show_workflow_prompt=False`：关闭工作流结构提示
- `show_progress=False`：关闭进度条
- `show_stage_prompt=False`：关闭阶段提示与收尾摘要
- `progress_callback=callable`：订阅结构化运行事件

`progress_callback(event)` 会收到字典事件：

- `event="stage"`：阶段更新（`prepare`、`execute_nodes`、`collect`、`finalize`）
- `event="task_settled"`：单任务完成更新，包含 `node`、`status`、`completed`、`total`
- `event="run_finished"`：最终摘要，包含 `status`、`success`、`failed`、`elapsed_seconds`

### 1. Pipeline 模式

- `depth_first`：按输入条目端到端推进
- `breadth_first`：按节点分批推进

### 2. 上下文流转

- 初始上下文：`{"input": ...}`
- 每个节点成功后写入：`context[node_name] = output_dict`

### 3. 重试与恢复

- 失败自动重试（指数退避）
- 使用同一个 `run_id` 可以续跑未完成任务

### 4. DAG 分支与汇合

- 通过 `add_node(node, depends_on=[...])` 显式声明依赖关系

```python
pipeline.add_node(FunctionNode(name="extract_od", func=extract_od), depends_on=[])
pipeline.add_node(FunctionNode(name="estimate_time", func=estimate_time), depends_on=["extract_od"])
pipeline.add_node(FunctionNode(name="estimate_cost", func=estimate_cost), depends_on=["extract_od"])
pipeline.add_node(
    FunctionNode(name="join_report", func=join_report),
    depends_on=["estimate_time", "estimate_cost"],
)
```

### 5. 跨输入 Collect/Reduce

```python
def merge_geojson(items, meta):
    features = [item["value"]["feature"] for item in items if "feature" in item["value"]]
    return {"type": "FeatureCollection", "features": features, "run_id": meta["run_id"]}

pipeline.add_collect_node(
    name="merge_geojson",
    func=merge_geojson,
    source_node="build_feature",
)
```

读取聚合结果：

```python
artifacts = pipeline.get_run_artifacts(run_id)
print(artifacts["merge_geojson"])
```

## API 概览

- `Pipeline.add_node(node, depends_on) -> Pipeline`
- `Pipeline.add_collect_node(name, func, source_node=None, include_failed=False) -> Pipeline`
- `Pipeline.run(inputs, run_id=None, show_workflow_prompt=True, show_progress=True, show_stage_prompt=True, progress_callback=None) -> str`
- `Pipeline.export_results(run_id, format="json") -> list[dict]`
- `Pipeline.get_run_artifacts(run_id) -> dict[str, dict]`
- `Pipeline.describe_workflow() -> dict`

## 许可证

MIT
