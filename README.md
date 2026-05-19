# SilkLoom Core 2.0 API 设计文档

## 1. 设计哲学 (Design Philosophy)

SilkLoom Core 2.0 定位于极简的结构化多模态批处理引擎。
对外，它坚持“一行代码跑通”的低心智负担体验，仅保留一套核心 API；对内，它通过高度解耦的流水线，默默为你处理并发调度、断点续跑、图片转换组装以及脏 JSON 自动修复（`json_repair`）等大模型工程痛点。

核心边界：本引擎的输入范围严格收敛于纯文本与图片。

---

## 2. 核心 API 参考

### 2.1 主引擎：`TaskLoom`

系统的唯一入口。负责模板管理、输出模式定义与任务调度。

**签名**

```python
class TaskLoom(Generic[T]):
    def __init__(
        self,
        model: str,
        prompt_template: str,
        system_prompt: str | None = None,
        response_model: type[BaseModel] | type[dict] | None = None,
        auto_repair_json: bool = True,
        max_retries: int = 3,
        client: Any | None = None,
        **llm_kwargs: Any
    ): ...
```

**参数说明**

* `model`: 模型名称（如 `"gpt-4o-mini"`、`"qwen-vl-plus"`）。
* `prompt_template`: Jinja2 语法的用户提示词模板（如 `"分析这张图：{{ text }}"`）。仅渲染文本变量。
* `system_prompt`: 可选的系统提示词。
* `response_model`: 输出结构。
  * `Pydantic BaseModel`：执行 JSON 提取与修复，返回强类型对象。
  * `dict`：执行 JSON 提取与修复，返回纯字典。
  * `None`：返回模型原始文本。
* `auto_repair_json`: 默认 `True`，通过 `json_repair` 自动挽救模型输出的非标准或残缺 JSON。
* `max_retries`: 单条任务失败时的最大重试次数。
* `client`: 兼容 OpenAI API 规范的客户端实例。
* `llm_kwargs`: 透传给底层大模型的参数（如 `temperature`、`top_p`）。

### 2.2 多模态输入契约 (Input Protocol)

`TaskLoom` 的执行方法（`process` / `map`）接受的数据源为 `dict` 或 `dict` 的集合。

引擎内部约定了特殊保留字段 `images`：

* 文本变量：字典中除 `images` 外的键，均视为 Jinja2 变量，参与 `prompt_template` 的渲染。
* 图片变量：字典中的 `images: list[str]` 字段。内部 `MessageBuilder` 会自动拦截此字段，支持传入本地路径（自动转 Base64）、URL 或 Data URI，并自动组装为大模型支持的多模态消息结构。

### 2.3 执行方法 (Execution)

* `process(data: str | dict) -> TaskResult[T]`
  同步单条执行。输入纯字符串会自动等价于 `{"text": ...}`。
* `aprocess(data: str | dict) -> TaskResult[T]`
  异步单条执行。
* `map(sequence: Iterable[str | dict], db_path: str = ".silkloom.db", run_id: str | None = None, workers: int = 5, show_progress: bool = False, progress_desc: str = "TaskLoom map", progress_callback: Callable[[int, int, dict[str, Any], TaskResult[T] | None, str], Any] | None = None) -> BatchResult[T]`
  同步批处理。传入 `run_id` 即启用 SQLite 缓存，支持断点续跑，自动跳过已成功任务。
* `amap(sequence: Iterable[str | dict], db_path: str = ".silkloom.db", run_id: str | None = None, max_concurrent: int = 5, show_progress: bool = False, progress_desc: str = "TaskLoom amap", progress_callback: Callable[[int, int, dict[str, Any], TaskResult[T] | None, str], Any] | None = None) -> BatchResult[T]`
  异步批处理。

如果你希望在批处理中显示进度条，先安装可选依赖：

```bash
pip install silkloom-core[progress]
```

然后在调用时显式开启：

```python
results = loom.map(
    items,
    run_id="cv_parse_v1",
    show_progress=True,
    progress_desc="解析简历",
)
```

如果你在 Gradio 中想显示更细的状态文案，可以用 `progress_callback` 自定义：

```python
def on_progress(completed, total, input_data, result, status_text):
    print(status_text)

results = loom.map(
    items,
    progress_callback=on_progress,
)
```

### 2.4 数据模型 (Data Models)

**`TaskResult[T]`**（单条任务快照）

```python
class TaskResult(BaseModel, Generic[T]):
    is_success: bool          # 是否成功
    data: T | None            # 解析后的核心数据（Pydantic 对象 / Dict / Str）
    error: str | None         # 错误信息堆栈
    input_data: dict          # 原始输入
    raw_output: str | None    # 大模型返回的原始文本
    reasoning: str | None     # 推理过程（如 DeepSeek/Qwen 的  Witticism 标签内容）
```

**`BatchResult[T]`**（批处理结果集，按输入顺序对齐）

* 提供 `.successful()` / `.failed()` 快捷过滤。
* 提供 `.to_dicts()` / `.to_pandas()` 快捷导出。

---

## 3. 典型应用范例

### 场景一：纯文本 + Pydantic 严谨抽取

文本数据通过 Jinja2 渲染，底层利用 `json_repair` 兜底，最终输出结构化对象。

```python
from pydantic import BaseModel
from silkloom_core import TaskLoom

class UserProfile(BaseModel):
    name: str
    skills: list[str]

loom = TaskLoom(
    model="gpt-4o-mini",
    prompt_template="提取简历中的信息：{{ text }}",
    response_model=UserProfile,
)

# 纯文本列表，自动包装为 {"text": ...}
results = loom.map(
    ["我是张三，精通 Python 和 Java", "李四，会前端和设计"],
    run_id="cv_parse_v1"  # 开启断点续跑
)

print(results[0].data.name)   # "张三"
print(results[0].data.skills) # ["Python", "Java"]
```

### 场景二：图片识别 + 字典灵活提取（图文多模态）

传入 `images` 字段，底层自动将本地图片转码并与渲染后的文本组装，无需定义 Pydantic 类，直接输出字典。

```python
from silkloom_core import TaskLoom

loom = TaskLoom(
    model="qwen-vl-max",  # 或 gpt-4o 等多模态模型
    prompt_template="请根据我的要求分析这张图片。用户要求：{{ instruction }}",
    response_model=dict,
)

result = loom.process({
    "instruction": "提取图中的菜名和总价，用 JSON 返回",
    "images": [
        "./receipt_01.jpg",               # 引擎内部自动读取并转 Base64
        "https://example.com/menu.png"    # URL 直接透传
    ]
})

if result.is_success:
    print(result.data.get("总价"))
```

### 场景三：异常捕获与脏数据排查

如果遇到极其混乱的模型输出，连 `json_repair` 都无法挽救，SilkLoom 会将原始信息原样保留在结果中，供开发者排查。

```python
failed_tasks = results.failed()

for task in failed_tasks:
    print(f"失败原因: {task.error}")
    print(f"导致失败的原始输入: {task.input_data}")
    print(f"大模型的胡言乱语: {task.raw_output}")
```

---

### 架构侧的设计收益总结

通过明确只支持文字和图片，你的引擎内部 `MessageBuilder` 逻辑将变得极其收敛和稳定，只需处理 Base64 图片的通用函数，不再面对各大模型乱七八糟的 File API 兼容问题，这让包的代码体积可以控制在极小的范围内。
