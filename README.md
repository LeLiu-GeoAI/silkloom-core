# SilkLoom Core

[中文](README.zh-CN.md) | [English](README.md)

SilkLoom Core is a minimal, stateful LLM batch engine.

The public surface is intentionally small:

- PromptMapper
- ResultSet

This README is split into two parts:

1. User Guide: installation, input formats, prompt rules, and examples.
2. API Reference: constructor arguments, method signatures, and returned objects.

Prompt templates use strict Jinja2 syntax. `user_prompt` and `system_prompt` render against each input item, so template variables must match the keys in that item's dictionary. Missing variables raise an error instead of rendering as empty text. For a plain string list, SilkLoom wraps each item as `{"text": "..."}`.

## Install

```bash
pip install silkloom-core
```

From source:

```bash
git clone https://github.com/LeLiu-GeoAI/silkloom-core.git
cd silkloom-core
pip install -e .
```

## User Guide

### Quick Start

```python
from openai import OpenAI
from silkloom_core import PromptMapper

client = OpenAI(api_key="your_key")

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="Translate into English: {{ text }}",
    client=client,
)

results = mapper.map(["你好", "今天天气不错"])
print(results[0].data)
print(results.success_count, results.failed_count)
```

### Input Formats

PromptMapper.map() accepts three common input shapes:

- list[str]: each string is wrapped as `{"text": ...}`
- list[dict]: each dict becomes one prompt context
- pandas.DataFrame: optional; each row becomes one prompt context and the column names become template variables

If you want to pass a DataFrame, install pandas separately. It is not required for normal usage.

Dictionary list example:

```python
from silkloom_core import PromptMapper

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="Extract name and intent from text: {{ text }}",
)

results = mapper.map([
    {"text": "My name is Alice. I want a refund."},
    {"text": "Bob asks about delivery."},
])
```

### Pandas DataFrame

Each DataFrame row is treated as one input item, and the column names are available as template variables.

```python
import pandas as pd
from silkloom_core import PromptMapper

df = pd.DataFrame(
    [
        {"text": "Urban heat island is intensifying.", "lang": "en"},
        {"text": "城市更新需要兼顾公平。", "lang": "zh"},
    ]
)

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="Rewrite the following {{ lang }} text: {{ text }}",
)

results = mapper.map(df)
```

### Prompt Template Rules

Template variables must match the keys in the input context.

```python
mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="Rewrite the following {{ lang }} text: {{ text }}",
)
```

For a DataFrame, this row exposes `text` and `lang` to the template:

```python
{"text": "Urban heat is rising.", "lang": "en"}
```

### Structured Output

```python
from pydantic import BaseModel
from silkloom_core import PromptMapper


class ExtractInfo(BaseModel):
    name: str
    intent: str


mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="Extract name and intent from text: {{ text }}",
    response_model=ExtractInfo,
)

results = mapper.map([
    {"text": "My name is Alice. I want a refund."},
    {"text": "Bob asks about delivery."},
])

print(results[0].data.name)
```

Note: if a model returns JSON wrapped in a ```json ... ``` code fence, SilkLoom will strip the fence and extract valid JSON before `response_model` validation.

### GLM and Ollama

#### GLM-4-Flash

```python
import os
from openai import OpenAI
from silkloom_core import PromptMapper

glm_client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

mapper = PromptMapper(
    model="glm-4-flash",
    user_prompt="Summarize this text: {{ text }}",
    client=glm_client,
)

results = mapper.map(["Urban renewal should balance efficiency and equity."])
```

#### Ollama

```python
from openai import OpenAI
from silkloom_core import PromptMapper

ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

mapper = PromptMapper(
    model="qwen2.5:7b",
    user_prompt="Rewrite in academic tone: {{ text }}",
    client=ollama_client,
)

results = mapper.map(["Traffic is usually worst in evening peak."])
```

### Multimodal Input

Pass image sources in `images` (supports local path, URL, or base64/data URI):

```python
from silkloom_core import PromptMapper

mapper = PromptMapper(
    model="gpt-4o",
    user_prompt="Describe these images and answer: {{ text }}",
)

results = mapper.map([
    {
        "text": "What is shown?",
        "images": ["./pic1.jpg", "https://example.com/pic2.png"],
    }
])
```

### Resumability

`map` supports resumability with SQLite via `db_path` + `run_id`:

```python
results = mapper.map(
    [{"text": "a"}, {"text": "b"}],
    db_path="my_run.db",
    run_id="demo_001",
    workers=5,
)
```

Running again with the same `run_id` reuses successful records.

### Single Item Execution

Use `run_one` when you only need one input:

```python
from silkloom_core import PromptMapper

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="Summarize in one sentence: {{ text }}",
)

result = mapper.run_one({"text": "Cities need compact and equitable transit systems."})
print(result.is_success, result.data)
```

### Exporting Results

`ResultSet` supports in-memory access and file export:

```python
results.run_id
results.success_count
results.failed_count
results.total_tokens
results.errors
results.outputs          # parsed output for each input (success or None on failure)
results.results          # full TaskResult list
results.successful()     # only successful TaskResult entries
results.failed()         # only failed TaskResult entries
results.raw_outputs      # raw model payload for each input (success and failure)
results.reasonings       # model reasoning/think text if provided by the backend
results[0]
results.export_jsonl("out.jsonl")
results.export_csv("out.csv", flatten=True)
```

## API Reference

### PromptMapper

Constructor:

```python
PromptMapper(
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    response_model: type[BaseModel] | None = None,
    max_retries: int = 3,
    client: Any | None = None,
)
```

Arguments:

- model: target model name, such as `gpt-4o-mini`
- user_prompt: required Jinja2 template for the user message
- system_prompt: optional Jinja2 template for the system message
- response_model: optional Pydantic model for structured output parsing
- max_retries: number of attempts for one item
- client: optional OpenAI-compatible client; defaults to the official client

Method:

```python
run_one(item: str | dict[str, Any]) -> TaskResult
map(sequence, db_path=".silkloom_cache.db", run_id=None, workers=5) -> ResultSet
```

Validation rules:

- `max_retries` must be >= 1
- `workers` must be >= 1
- `map` does not accept a single string (use `run_one("...")` or `map(["..."])`)

Accepted inputs:

- list[str]
- list[dict]
- pandas.DataFrame

### ResultSet

`ResultSet` behaves like a sequence aligned with the input order.

Properties:

- run_id
- success_count
- failed_count
- total_tokens
- errors
- outputs
- results
- raw_outputs
- reasonings

Methods:

- `results[0]`: returns the TaskResult at the same index as the input
- `successful()`: returns successful TaskResult entries
- `failed()`: returns failed TaskResult entries
- `export_jsonl(path)`: write successful results to JSONL
- `export_csv(path, flatten=False, include_usage=True)`: write a CSV export

### TaskResult

Each raw task result contains:

- is_success
- data
- error
- usage
- input_data
- raw_output
- reasoning

Note: in typical usage, you do not need to instantiate `TaskResult` manually.
You only read it from `run_one(...)`, `results[0]`, or `results.results`.

### Access Raw Output and Think Content

SilkLoom stores raw model output for every item, including failed items:

```python
for i, task_result in enumerate(results.results):
    print(i, task_result.is_success)
    print("raw:", task_result.raw_output)
    print("error:", task_result.error)
```

For think/reasoning models, SilkLoom tries to extract reasoning from common fields
(`reasoning`, `reasoning_content`) and from `<think>...</think>` blocks. If the model
or provider does not expose reasoning, `reasoning` will be `None`.

## License

MIT

