# SilkLoom Core

[中文](README.zh-CN.md) | [English](README.md)

SilkLoom Core 是一个极简、带状态持久化的大模型批处理引擎。
对外暴露的公共 API 很小：

- PromptMapper
- ResultSet


本文档分成两部分：

1. 用户指南：安装、输入类型、提示词规则和示例。
2. API 文档：构造参数、方法签名和返回对象。

提示词模板使用严格的 Jinja2 语法。`user_prompt` 和 `system_prompt` 会针对每条输入数据渲染，模板里的变量名必须与该条输入字典的键一致。缺失变量会直接报错，不会静默渲染为空字符串。对于纯字符串列表，SilkLoom 会自动包装成 `{"text": "..."}`。

## 安装

```bash
pip install silkloom-core
```

源码安装：

```bash
git clone https://github.com/LeLiu-GeoAI/silkloom-core.git
cd silkloom-core
pip install -e .
```

## 用户指南

### 快速开始

```python
from openai import OpenAI
from silkloom_core import PromptMapper

client = OpenAI(api_key="your_key")

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="请翻译成英文：{{ text }}",
    client=client,
)

results = mapper.map(["你好", "今天天气不错"])
print(results[0].data)
print(results.success_count, results.failed_count)
```

### 输入类型

PromptMapper.map() 支持三种常见输入：

- list[str]：每个字符串会自动包装成 `{"text": ...}`
- list[dict]：每个字典会作为一条输入上下文
- pandas.DataFrame：可选，每一行会作为一条输入上下文，列名会直接成为模板变量

如果你要传 DataFrame，需要单独安装 pandas。普通用法不需要它。

字典列表示例：

```python
from silkloom_core import PromptMapper

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="从文本中提取姓名和意图：{{ text }}",
)

results = mapper.map([
    {"text": "我叫 Alice，我想退款"},
    {"text": "Bob 在咨询物流"},
])
```

### Pandas DataFrame

DataFrame 的每一行会被当作一条输入数据，列名会直接成为模板变量。

```python
import pandas as pd
from silkloom_core import PromptMapper

df = pd.DataFrame(
    [
        {"text": "城市热岛正在加剧。", "lang": "zh"},
        {"text": "Urban renewal should balance efficiency and equity.", "lang": "en"},
    ]
)

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="请改写以下 {{ lang }} 文本：{{ text }}",
)

results = mapper.map(df)
```

### 提示词模板规则

模板里的变量名必须和输入上下文中的键一致。

```python
mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="请改写以下 {{ lang }} 文本：{{ text }}",
)
```

如果输入是 DataFrame 的一行，下面这些列名就会直接暴露给模板：

```python
{"text": "城市热岛正在加剧。", "lang": "zh"}
```

### 结构化输出

```python
from pydantic import BaseModel
from silkloom_core import PromptMapper


class ExtractInfo(BaseModel):
    name: str
    intent: str


mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="从文本中提取姓名和意图：{{ text }}",
    response_model=ExtractInfo,
)

results = mapper.map([
    {"text": "我叫 Alice，我想退款"},
    {"text": "Bob 在咨询物流"},
])

print(results[0].data.name)
```

说明：如果模型返回的是 ```json ... ``` 代码块，SilkLoom 会自动去掉围栏并提取 JSON 后再做 `response_model` 校验。

### 其他模型（GLM / Ollama）

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
    user_prompt="请总结这段文本：{{ text }}",
    client=glm_client,
)

results = mapper.map(["城市更新要兼顾效率和公平。"])
```

#### Ollama（本地）

```python
from openai import OpenAI
from silkloom_core import PromptMapper

ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

mapper = PromptMapper(
    model="qwen2.5:7b",
    user_prompt="请将这句话改写为学术表达：{{ text }}",
    client=ollama_client,
)

results = mapper.map(["晚高峰交通拥堵最明显。"])
```

### 多模态输入

在输入中使用 `images` 字段（支持本地路径、URL、base64/data URI）：

```python
from silkloom_core import PromptMapper

mapper = PromptMapper(
    model="gpt-4o",
    user_prompt="请分析图片并回答：{{ text }}",
)

results = mapper.map([
    {
        "text": "图里主要内容是什么？",
        "images": ["./pic1.jpg", "https://example.com/pic2.png"],
    }
])
```

### 断点续跑

`map` 通过 `db_path` + `run_id` 提供 SQLite 级别断点续跑：

```python
results = mapper.map(
    [{"text": "a"}, {"text": "b"}],
    db_path="my_run.db",
    run_id="demo_001",
    workers=5,
)
```

再次用同一个 `run_id` 运行时，会复用已成功结果。

### 单条执行

当你只需要处理一条输入时，使用 `run_one`：

```python
from silkloom_core import PromptMapper

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="请用一句话总结：{{ text }}",
)

result = mapper.run_one({"text": "城市交通系统需要兼顾效率、可达性与公平性。"})
print(result.is_success, result.data)
```

### 结果导出

`ResultSet` 既可以直接索引，也可以导出文件：

```python
results.run_id
results.success_count
results.failed_count
results.total_tokens
results.errors
results.outputs          # 每条输入对应的解析结果（成功为 data，失败为 None）
results.results          # 完整 TaskResult 列表
results.successful()     # 仅成功任务
results.failed()         # 仅失败任务
results.raw_outputs      # 每条输入对应的原始模型输出（成功/失败都保留）
results.reasonings       # 若后端返回推理字段，则这里可拿到 think/reasoning 文本
results[0]
results.export_jsonl("out.jsonl")
results.export_csv("out.csv", flatten=True)
```

## API 文档

### PromptMapper

构造函数：

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

参数说明：

- model：目标模型名，例如 `gpt-4o-mini`
- user_prompt：必填的用户提示词模板，使用 Jinja2 语法
- system_prompt：可选的系统提示词模板，使用 Jinja2 语法
- response_model：可选的 Pydantic 模型，用于结构化输出解析
- max_retries：单条输入的最大重试次数
- client：可选的 OpenAI 兼容客户端；不传则使用官方客户端

方法：

```python
run_one(item: str | dict[str, Any]) -> TaskResult
map(sequence, db_path=".silkloom_cache.db", run_id=None, workers=5) -> ResultSet
```

参数约束：

- `max_retries` 必须 >= 1
- `workers` 必须 >= 1
- `map` 不接受单个字符串（请使用 `run_one("...")` 或 `map(["..."])`）

支持的输入类型：

- list[str]
- list[dict]
- pandas.DataFrame

### ResultSet

`ResultSet` 的顺序与输入顺序严格对齐。

属性：

- run_id
- success_count
- failed_count
- total_tokens
- errors
- outputs
- results
- raw_outputs
- reasonings

方法：

- `results[0]`：返回和输入同索引的 TaskResult
- `successful()`：返回成功任务列表
- `failed()`：返回失败任务列表
- `export_jsonl(path)`：导出成功结果到 JSONL
- `export_csv(path, flatten=False, include_usage=True)`：导出 CSV

### TaskResult

每条底层任务结果包含：

- is_success
- data
- error
- usage
- input_data
- raw_output
- reasoning

说明：普通使用中你不需要手动构造 `TaskResult`，只需要读取 `run_one(...)`、`results[0]` 或 `results.results` 返回的对象即可。

### 获取原始输出与 Think 内容

SilkLoom 会为每条输入保留原始模型输出，包括失败项：

```python
for i, task_result in enumerate(results.results):
    print(i, task_result.is_success)
    print("raw:", task_result.raw_output)
    print("error:", task_result.error)
```

对于 think/reasoning 模型，SilkLoom 会优先尝试从常见字段（`reasoning`、`reasoning_content`）提取，
并兼容 `<think>...</think>` 文本块。如果模型或服务商本身不返回推理内容，则 `reasoning` 为 `None`。

## 许可证

MIT

