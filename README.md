# SilkLoom Core

[中文](README.zh-CN.md) | [English](README.md)

SilkLoom Core is a lightweight, resilient batch pipeline for repeatable workflows. It is a general-purpose execution layer for running the same process over many inputs, with retries and resumability built in.

## Overview

Key capabilities:

- Node-based workflow composition (`LLMNode`, `FunctionNode`, custom `BaseNode`)
- Concurrent execution
- Retry with exponential backoff
- SQLite persistence and resumability with `run_id`
- Structured output with Pydantic

Design philosophy:

- Focus on repeatable execution, not intelligent scheduling
- Keep workflow logic explicit and deterministic
- Make long-running batch jobs restartable and observable

## Installation

```bash
pip install silkloom-core
```

Install from source:

```bash
git clone https://github.com/your-org/silkloom-core.git
cd silkloom-core
pip install -e .
```

Dev extras:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from silkloom_core import Pipeline, LLMNode, FunctionNode


def score_text(text: str) -> dict:
    score = min(len(text) / 100, 1.0)
    return {"score": round(score, 3)}


pipeline = Pipeline(db_path="pipeline.db", execution_mode="depth_first", default_workers=4)

pipeline.add_node(
    LLMNode(
        name="summarize",
        prompt_template="Summarize in one sentence: {input.text}",
        model="gpt-4o-mini",
    )
)

pipeline.add_node(
    FunctionNode(
        name="score",
        func=score_text,
        kwargs_mapping={"text": "{summarize.text}"},
    )
)

run_id = pipeline.run([
    {"text": "SilkLoom Core supports repeatable LLM batch processing."},
    {"text": "It persists progress in SQLite and can resume by run_id."},
])

print(pipeline.export_results(run_id))
```

## OpenAI-Compatible Endpoints

`LLMNode` supports custom OpenAI clients via:

```python
LLMNode(..., client=your_openai_client)
```

So any endpoint compatible with OpenAI Chat Completions can be used.

### 1) Official OpenAI

```python
from silkloom_core import LLMNode

node = LLMNode(
    name="extract",
    prompt_template="Extract key facts: {input.note}",
    model="gpt-4o-mini",
)
```

```bash
export OPENAI_API_KEY="your_openai_key"
# PowerShell:
# $env:OPENAI_API_KEY="your_openai_key"
```

### 2) GLM-4-Flash (OpenAI-compatible)

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
    prompt_template="Extract city, topic, and coordinates: {input.note}",
    model="glm-4-flash",
    client=glm_client,
)
```

```bash
export ZHIPUAI_API_KEY="your_glm_key"
# PowerShell:
# $env:ZHIPUAI_API_KEY="your_glm_key"
```

### 3) Local Ollama (OpenAI-compatible)

Start Ollama and pull a model (example):

```bash
ollama pull qwen2.5:7b
ollama serve
```

Use it in SilkLoom Core:

```python
from openai import OpenAI
from silkloom_core import LLMNode

ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

node = LLMNode(
    name="local_summary",
    prompt_template="Summarize this note: {input.note}",
    model="qwen2.5:7b",
    client=ollama_client,
)
```

Note: local models vary in structured-output quality. If you use `response_model`, explicitly require strict JSON-only output in the prompt.

## Example Scripts

The provided examples use GIS/urban research as one domain case, but SilkLoom Core itself is domain-agnostic.

```bash
python examples/quickstart.py
python examples/structured_output.py
python examples/resume_with_run_id.py
python examples/trajectory_od_commute.py
```

- quickstart.py: summarize notes and tag themes
- structured_output.py: extract structured attributes and build GeoJSON-like features
- resume_with_run_id.py: simulate repeatable tile processing with resume
- trajectory_od_commute.py: OD extraction + distance/time segmentation + flowline output

## Core Concepts

### 1. Pipeline Modes

- `depth_first`: per-item end-to-end progression
- `breadth_first`: stage-by-stage progression across items

### 2. Context Flow

- Initial context: `{"input": ...}`
- Node output storage: `context[node_name] = output_dict`

### 3. Retry and Resume

- Automatic retries with exponential backoff
- Resume unfinished tasks by reusing the same `run_id`

## API Summary

- `Pipeline.add_node(node) -> Pipeline`
- `Pipeline.run(inputs, run_id=None) -> str`
- `Pipeline.export_results(run_id, format="json") -> list[dict]`

## License

MIT
