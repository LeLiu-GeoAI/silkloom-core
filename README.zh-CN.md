# SilkLoom Core

[中文](README.zh-CN.md) | [English](README.md)

SilkLoom Core 是一个极简、带状态持久化的大模型批处理引擎。对外暴露的公共 API 极度克制，仅包含 PromptMapper 和 ResultSet，却能覆盖大多数 LLM 批处理场景。

**✨ 特性摘要**

- **极简 API**：一行代码完成单条/批量处理，心智负担低。
- **状态持久化**：内置 SQLite 断点续跑，中断后可继续。
- **零依赖数据集成**：支持 to_dicts()；安装 pandas 后可一键 to_pandas()。
- **全异步支持**：提供 asyncio API，便于接入 FastAPI/Gradio/Streamlit。
- **强格式校验**：结合 Pydantic 做结构化校验，并自动提取 JSON。
- **推理文本提取**：自动提取 reasoning 或 <think> 推理过程（若模型提供）。

## 1. 安装

```bash
pip install silkloom-core
```

注：核心功能不依赖 pandas。只有在使用 to_pandas() 时，才需要额外安装 pandas。

可选功能依赖安装：

```bash
# DataFrame 导出支持
pip install "silkloom-core[data]"

# 进度条支持
pip install "silkloom-core[progress]"

# 一次安装所有可选能力
pip install "silkloom-core[full]"
```

源码安装：

```bash
git clone https://github.com/LeLiu-GeoAI/silkloom-core.git
cd silkloom-core
pip install -e .
```

## 2. 基础教程

### 2.1 快速开始

```python
from openai import OpenAI
from silkloom_core import PromptMapper

client = OpenAI(api_key="your_key")

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="请翻译成英文：{{ text }}",
    client=client,
    temperature=0.5,  # 额外参数会透传给 LLM
)

results = mapper.map(["你好", "今天天气不错"])
# 默认显示进度条（需安装 tqdm），可通过 show_progress=False 关闭

print(results[0].data)
print(f"成功: {results.success_count}, 失败: {results.failed_count}")
```

### 2.2 输入类型

模板使用严格 Jinja2 语法。模板变量必须与输入键一致，缺失变量会直接报错。

map() 支持三类输入：

1. **list[str]**：自动包装为 {"text": ...}
2. **list[dict]**：最常用，字典 key 直接作为模板变量
3. **pandas.DataFrame**：每行作为一条上下文，列名就是模板变量

```python
mapper = PromptMapper(model="gpt-4o-mini", user_prompt="提取意图：{{ text }}")
results = mapper.map([
    {"text": "我叫 Alice，我想退款", "id": 1},
    {"text": "Bob 在咨询物流", "id": 2},
])
```

### 2.3 结果访问与导出

ResultSet 提供任务级访问、统计信息和导出能力。

```python
# 1) DataFrame（需要 pandas）
df = results.to_pandas(merge_input=True)

# 2) 字典列表（零依赖）
dicts = results.to_dicts(merge_input=True)

# 3) 文件导出
results.export_jsonl("out.jsonl")
results.export_csv("out.csv", flatten=True)

# 4) 常用访问
print(results.successful())
print(results.raw_outputs)
print(results.reasonings)
```

### 2.4 单条执行

```python
result = mapper.run_one({"text": "城市交通系统需要兼顾效率与公平。"})
print(result.is_success, result.data)
```

## 3. 进阶特性

### 3.1 结构化输出校验（Pydantic）

传入 response_model 后，SilkLoom 会自动提取 JSON（包括 markdown 围栏中的 JSON），并执行 Pydantic 校验。

```python
from pydantic import BaseModel

class ExtractInfo(BaseModel):
    name: str
    intent: str

mapper = PromptMapper(
    model="gpt-4o-mini",
    user_prompt="提取姓名和意图：{{ text }}",
    response_model=ExtractInfo,
)

results = mapper.map([{"text": "我叫 Alice，想退款"}])
print(results[0].data.name)
```

### 3.2 异步并发处理（Asyncio）

适合集成到 FastAPI、Streamlit、Gradio 等框架。底层使用 Semaphore 控制并发。

```python
import asyncio
from silkloom_core import PromptMapper

async def demo():
    mapper = PromptMapper(model="gpt-4o-mini", user_prompt="请总结：{{ text }}")

    results = await mapper.amap(
        [{"text": "城市热岛效应加剧。"}, {"text": "全球变暖趋势。"}],
        max_concurrent=5,
        show_progress=True,
    )
    print(results.success_count, results.failed_count)

asyncio.run(demo())
```

### 3.3 状态持久化（断点续跑）

传入 db_path + run_id 即启用 SQLite 缓存。再次使用相同 run_id 时，会跳过已成功任务。

```python
results = mapper.map(
    large_dataset,
    db_path="my_run.db",
    run_id="experiment_v1",
    workers=5,
)
```

### 3.4 多模态输入（图片）

在输入字典中包含 images 字段（支持本地路径、URL、base64/data URI）。

```python
results = mapper.map([
    {
        "text": "图里主要内容是什么？",
        "images": ["./pic1.jpg", "https://example.com/pic2.png"],
    }
])
```

## 4. 生态接入

SilkLoom 兼容 OpenAI 风格客户端，便于接入本地或第三方服务。

### 4.1 Ollama（本地）

```python
from openai import OpenAI
from silkloom_core import PromptMapper

ollama_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
mapper = PromptMapper(model="qwen2.5:7b", user_prompt="{{ text }}", client=ollama_client)
```

### 4.2 ZhipuAI / GLM

```python
from openai import OpenAI
from silkloom_core import PromptMapper

glm_client = OpenAI(api_key="your_key", base_url="https://open.bigmodel.cn/api/paas/v4/")
mapper = PromptMapper(model="glm-4-flash", user_prompt="{{ text }}", client=glm_client)
```

## 5. API 手册

### 5.1 PromptMapper

**构造函数**

```python
PromptMapper(
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    response_model: type[BaseModel] | None = None,
    max_retries: int = 3,
    client: Any | None = None,
    **llm_kwargs,  # 如 temperature, top_p, max_tokens
)
```

**核心方法**

- run_one(item: str | dict) -> TaskResult：同步单条执行
- map(sequence, db_path=".silkloom_cache.db", run_id=None, workers=5, show_progress=True) -> ResultSet：同步批处理
- arun_one(item: str | dict) -> TaskResult：异步单条执行
- amap(sequence, db_path=".silkloom_cache.db", run_id=None, max_concurrent=5, show_progress=True) -> ResultSet：异步批处理

**参数约束**

- max_retries >= 1
- workers >= 1
- max_concurrent >= 1
- map() 不接受单个字符串，请使用 run_one("...") 或 map(["..."])

### 5.2 ResultSet

严格按输入顺序对齐的结果集合。

**常用属性**

- run_id
- success_count / failed_count
- total_tokens
- results（完整 TaskResult 列表）
- raw_outputs（原始输出）
- reasonings（推理文本）

**常用方法**

- successful() / failed()
- to_dicts(merge_input=True)
- to_pandas(merge_input=True)
- export_jsonl(path)
- export_csv(path, flatten=False, include_usage=True)

### 5.3 TaskResult

单条任务结果对象，核心字段包括：

- is_success：是否成功
- data：解析后的结果
- error：失败错误信息
- input_data：原始输入
- raw_output：模型原始输出
- reasoning：推理文本（若有）

### 5.4 泛型与 IDE 类型提示

SilkLoom 已在 PromptMapper、ResultSet、TaskResult 中引入泛型，能够提升 IDE 对 data 字段的补全能力。

```python
from pydantic import BaseModel
from silkloom_core import PromptMapper

class ExtractInfo(BaseModel):
    name: str
    intent: str

mapper = PromptMapper(response_model=ExtractInfo, model="gpt-4o-mini", user_prompt="{{ text }}")
results = mapper.map([{"text": "我叫 Alice，想退款"}])

first = results[0]
if first.data is not None:
    print(first.data.name)    # IDE 可推导为 ExtractInfo
    print(first.data.intent)
```

当未设置 response_model 时，data 通常为字符串（或模型原始文本内容）。

### 5.5 自定义异常

SilkLoom 提供了可直接导入的异常类型，便于业务层做精细化错误处理。

- ConfigurationError：参数配置不合法（如 max_retries < 1）
- InvalidInputError：输入类型不符合要求（如把单个字符串传给 map）
- AsyncClientNotConfiguredError：调用异步方法但未配置 async 客户端
- TemplateRenderError：Jinja2 模板渲染失败
- ResponseParseError：结构化输出解析失败
- LLMRequestError：底层请求失败（会在 TaskResult.error 中带类型前缀）

```python
from silkloom_core import PromptMapper, InvalidInputError, ConfigurationError

try:
    mapper = PromptMapper(model="gpt-4o-mini", user_prompt="{{ text }}", max_retries=0)
except ConfigurationError as e:
    print("配置错误:", e)

try:
    mapper = PromptMapper(model="gpt-4o-mini", user_prompt="{{ text }}")
    mapper.map("单个字符串")
except InvalidInputError as e:
    print("输入错误:", e)
```

## 许可证

MIT
