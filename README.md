# 📚 SilkLoom Core 2.0 API 设计与架构指南

## 1. 核心设计哲学 (Design Philosophy)

SilkLoom Core 2.0 定位于**极简、高可用、结构化的大模型多模态批处理引擎**。

* **对外：低心智负担。** 坚持“一行代码跑通”，仅保留一套核心 API。
* **对内：工程痛点吞噬者。** 采用“计算与存储分离”的流水线，默默处理并发调度、断点续跑、图片转换组装，以及依赖 `json_repair` 的脏 JSON 自动挽救。

---

## 2. 内部架构与数据流转 (Architecture & Pipeline)

为了实现“可并行、可中断、可异步”，引擎底层执行器遵循以下状态流转：

1. **哈希指纹化 (Fingerprinting)**：引擎遍历输入序列，为每个输入字典生成 SHA-256 哈希值作为唯一 `Task ID`。
2. **缓存拦截 (Cache Intercept)**：如果指定了 `task_name`，引擎会查询本地 SQLite 库，命中的任务直接标记成功，瞬间通过。
3. **并发调度 (Worker Pool)**：未命中缓存的任务进入同步线程池 (`ThreadPoolExecutor`) 或异步任务组 (`asyncio.gather`)。
4. **原子持久化 (Atomic Persist)**：任务一旦成功（含 JSON 修复），立刻原子级 `Upsert` 写入 SQLite WAL 模式缓存。无论外部环境如何崩溃，已完成数据绝对安全。
5. **灵活消费 (Flexible Consumption)**：通过 `map`（阻塞重排组装）或 `stream`（流式实时释放）将结果交付给前端。

---

## 3. 核心 API 参考 (Core API Reference)

### 3.1 主引擎：`TaskLoom`

`TaskLoom` 是系统的唯一入口。支持作为上下文管理器（Context Manager）使用以自动释放底层连接资源。

```python
from typing import Any, AsyncGenerator, Generator, Generic, Iterable, TypeVar
from pydantic import BaseModel
from silkloom_core import TaskResult, BatchResult

T = TypeVar("T")

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
        db_path: str = ".silkloom.db",
        **llm_kwargs: Any
    ): ...

    def close(self): ...
    
    # 支持 Context Manager
    def __enter__(self) -> "TaskLoom": ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

```

**核心参数说明：**

* `response_model`: 决定输出形态。传入 `Pydantic 模型` 返回强类型对象；传入 `dict` 返回字典；传入 `None` 返回原始文本。
* `prompt_template`: Jinja2 语法的提示词模板（如 `"分析：{{ text }}"`）。

---

### 3.2 数据模型 (Data Models)

**单条快照：`TaskResult[T]**`

```python
class TaskResult(BaseModel, Generic[T]):
    task_id: str              # 根据输入内容生成的唯一 Hash 指纹
    is_success: bool          # 任务是否成功解析
    data: T | None            # 最终结构化数据 (Pydantic实例 / Dict / Str)
    error: str | None         # 异常堆栈 (仅失败时存在)
    input_data: dict          # 触发该任务的原始输入字典
    raw_output: str | None    # 大模型返回的原始字符串 (供脏数据兜底排查)
    reasoning: str | None     # 模型推理过程 (如 DeepSeek <think>)
    cached: bool              # 标识该结果是否来自于 SQLite 缓存

```

**批次集合：`BatchResult[T]**` (仅 `map/amap` 返回)

```python
class BatchResult(Generic[T]):
    results: list[TaskResult[T]]
    
    def successful(self) -> list[TaskResult[T]]: ...
    def failed(self) -> list[TaskResult[T]]: ...
    def to_pandas(self) -> "pd.DataFrame": ...  # 自动展平 data 导出

```

---

### 3.3 执行模式 (Execution Matrix)

所有输入数据源（`data` / `sequence`）统一接受 **`dict` 或 `dict` 的集合**。包含 `images` 键时自动触发多模态逻辑。

#### 1. 单例执行 (Single)

适用于即时问答或单条测试。

```python
def process(self, data: str | dict) -> TaskResult[T]: ...
async def aprocess(self, data: str | dict) -> TaskResult[T]: ...

```

#### 2. 阻塞批处理 (Blocking Batch)

适用于后台脚本或定时任务。等待所有任务跑完后一次性返回汇总结果。**默认保证返回顺序与输入顺序完全一致。**

```python
def map(
    self,
    sequence: Iterable[str | dict],
    task_name: str | None = None,
    max_workers: int = 5,
) -> BatchResult[T]: ...

async def amap(
    self,
    sequence: Iterable[str | dict],
    task_name: str | None = None,
    max_workers: int = 5,
) -> BatchResult[T]: ...

```

#### 3. 流式批处理 (Streaming Batch) - ✨ 核心高阶 API

适用于前端 UI（Gradio / Streamlit）或响应式 API。**极低内存占用，按完成状态实时 Yield**。

```python
def stream(
    self,
    sequence: Iterable[str | dict],
    task_name: str | None = None,
    max_workers: int = 5,
    ordered: bool = False,
) -> Generator[TaskResult[T], None, None]: ...

async def astream(
    self,
    sequence: Iterable[str | dict],
    task_name: str | None = None,
    max_workers: int = 5,
    ordered: bool = False,
) -> AsyncGenerator[TaskResult[T], None]: ...

```

> **关于 `ordered` 参数：**
> * `ordered=False` (默认)：先处理完的任务先返回。速度最快，UI 响应体验最佳。
> * `ordered=True`：内部建立缓冲队列，严格按 `sequence` 的原顺序阻塞 Yield。
> 
> 

---

## 4. 典型应用范例 (Best Practices)

### 场景一：纯文本结构化抽取 (Pydantic 兜底)

文本数据通过 Jinja2 渲染，底层自动执行校验与重试。

```python
from pydantic import BaseModel
from silkloom_core import TaskLoom

class UserProfile(BaseModel):
    name: str
    skills: list[str]

with TaskLoom(
    model="gpt-4o-mini",
    prompt_template="提取简历中的信息：{{ text }}",
    response_model=UserProfile,
) as loom:

    # 自动开启缓存与断点续跑机制
    results = loom.map(
        [{"text": "张三精通 Python"}, {"text": "李四会设计"}], 
        task_name="cv_parse_v1",
        max_workers=5
    )

    print(results.successful()[0].data.skills) # ["Python"]

```

### 场景二：多模态图像批处理

引擎自动拦截 `images` 字段，完成本地图片转码/远程图片拉取，组装为大模型支持的复杂协议。

```python
loom = TaskLoom(
    model="qwen-vl-max", 
    prompt_template="根据要求分析图片：{{ instruction }}",
    response_model=dict,
)

result = loom.process({
    "instruction": "提取图中的菜名和总价",
    "images": [
        "./receipt_01.jpg",               # 自动读取并转 Base64
        "https://example.com/menu.png"    # URL 直接透传
    ]
})

```

### 场景三：Gradio 流式渲染与断点续跑 (UI 融合)

利用 `stream`，哪怕网页中途关闭，已跑完的数据早已安全落库。再次点击瞬间完成缓存加载，进度条无缝续接。

```python
import gradio as gr
from silkloom_core import TaskLoom

loom = TaskLoom(
    model="deepseek-chat",
    prompt_template="总结论文核心方法：{{ text }}",
    response_model=dict
)

def process_papers(papers_list, progress=gr.Progress()):
    total = len(papers_list)
    results_list = []
    
    # stream(ordered=False) 保证最快的视觉反馈
    generator = loom.stream(
        sequence=papers_list,
        task_name="gradio_paper_batch_v1",  # 开启容灾缓存
        max_workers=10
    )
    
    for i, task in enumerate(generator, 1):
        status = "✅ 成功" if task.is_success else "❌ 失败"
        progress(i / total, desc=f"进度: {i}/{total} | 最新: {status}")
        
        results_list.append({
            "处理状态": status,
            "提取数据": task.data,
            "缓存命中": task.cached
        })
        
        # 实时逐行更新 UI 表格
        yield results_list

```

### 场景四：FastAPI 异步 SSE 实时推送

在现代后端架构中，使用 `astream` 释放 ASGI 容器的最高并发性能。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
loom = TaskLoom(
    model="gpt-4o-mini",
    prompt_template="提取发票关键信息：{{ text }}",
    response_model=dict
)

@app.post("/api/batch_extract")
async def batch_extract(payload: list[dict]):
    
    async def event_stream():
        # 充分利用 asyncio 的并发调度机制
        async for task in loom.astream(payload, task_name="invoice_prod", max_workers=20):
            yield f"data: {json.dumps(task.model_dump())}\n\n"
            
    return StreamingResponse(event_stream(), media_type="text/event-stream")

```