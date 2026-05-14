# SilkLoom Core API 优化指南 (v4.1+)

本文档介绍了对 SilkLoom Core 接口设计的主要改进，重点关注 **简化使用、扩展功能、改进错误反馈** 三个方面。

---

## 📋 优化概览

| 优化项 | 痛点 | 解决方案 |
|--------|------|--------|
| **模板参数映射** | Jinja2 语法不够直观 | 简单格式模式、变量验证 |
| **结果处理** | 导出能力有限 | 链式操作、更多导出格式 |
| **缓存管理** | SQLite 机制难以理解 | CacheManager 工具类 |
| **异步配置** | 需要手动初始化 AsyncOpenAI | 自动初始化 |
| **错误反馈** | 错误消息不清楚 | 详细解决建议 |

---

## 🎯 优化 1: 简化模板参数映射

### 问题
之前必须使用 Jinja2 语法 `{{ variable }}`，对学术研究场景来说有些复杂。

### 新功能

#### 1.1 简单格式模式 (`simple_format_mode`)
```python
from silkloom_core import PromptMapper

# ❌ 原来的方式（仍然支持）
pm_jinja = PromptMapper(
    model="gpt-4",
    user_prompt="分析以下论文：{{ paper }}\n关键词：{{ keywords }}"
)

# ✓ 新增：简单格式模式
pm_simple = PromptMapper(
    model="gpt-4",
    user_prompt="分析以下论文：{paper}\n关键词：{keywords}",
    simple_format_mode=True  # 启用简单模式
)

# 两种方式功能完全相同，但简单模式更容易理解
```

#### 1.2 查看模板需要的变量
```python
pm = PromptMapper(
    model="gpt-4",
    user_prompt="{title}\n{abstract}\n{keywords}"
)

# 查看模板需要哪些变量
print(pm.required_variables)
# 输出: {'title', 'abstract', 'keywords'}
```

#### 1.3 验证输入数据
```python
# 验证输入是否包含所有需要的变量
papers = [
    {"title": "AI in Education", "abstract": "...", "keywords": "AI,学习"},
    {"title": "Deep Learning", "abstract": "...", "abstract_zh": "..."},  # ❌ 缺少 keywords
]

is_valid = pm.validate_inputs(papers, verbose=True)
# 输出:
#   ✓ 输入 #0: 字段完整
#   ❌ 输入 #1: 缺少字段 {'keywords'}
#
# ⚠️  模板需要字段：{'title', 'abstract', 'keywords'}
#    缺失字段：{'keywords'}

if not is_valid:
    print("请检查输入数据，确保包含所有必要字段")
```

---

## 🎯 优化 2: 强化结果处理能力

### 问题
结果集的导出和转换能力有限，难以满足学术研究的多种需求。

### 新功能

#### 2.1 链式过滤 (`filter`)
```python
# 获取所有成功运行的结果
successful = results.filter(lambda r: r.is_success)

# 获取 token 消耗超过 1000 的结果
expensive = results.filter(
    lambda r: r.usage and r.usage.get('total_tokens', 0) > 1000
)

# 支持链式调用
high_quality_successful = (results
    .filter(lambda r: r.is_success)
    .filter(lambda r: r.usage['total_tokens'] < 2000)
)
```

#### 2.2 映射操作 (`map`)
```python
# 提取所有 token 数
token_counts = results.map(lambda r: r.usage.get('total_tokens', 0) if r.usage else 0)
# 输出: [150, 200, 125, ...]

# 提取所有错误信息
errors = results.map(lambda r: r.error if not r.is_success else None)
# 输出: [None, None, 'ParseError', None, ...]

# 统计成功率
success_rate = len(results.successful()) / len(results) * 100
```

#### 2.3 转换数据 (`transform`)
```python
# 对成功的结果应用转换函数
# 示例：将所有文本转为大写
uppercase_results = results.transform(
    lambda data: data.upper() if isinstance(data, str) else data
)

# 示例：提取特定字段
extracted = results.transform(
    lambda data: data.get('summary', '') if isinstance(data, dict) else data
)

# transform 返回一个新的 ResultSet，支持继续链式调用
high_token = (results
    .transform(lambda d: {"summary": d[:100]})
    .filter(lambda r: r.is_success)
)
```

#### 2.4 多种导出格式
```python
# 导出为字典列表
dicts = results.to_dicts(merge_input=True)
# [
#   {"title": "...", "abstract": "...", "output": "...", "is_success": True, ...},
#   ...
# ]

# 导出为 Pandas DataFrame（需要安装 pandas）
df = results.to_pandas()
df.to_excel('results.xlsx', index=False)  # 导出到 Excel

# 导出为 JSONL（逐行 JSON）
results.export_jsonl('output.jsonl')

# 导出为 CSV（支持打平列表字段）
results.export_csv('results.csv', flatten=True, include_usage=True)
```

---

## 🎯 优化 3: 简化缓存管理

### 问题
SQLite 缓存机制工作原理不清晰，断点续跑容易出错。

### 新功能: CacheManager

#### 3.1 查看缓存状态
```python
from silkloom_core import CacheManager

cache = CacheManager(db_path=".silkloom_cache.db")

# 查看所有运行的统计
summary = cache.inspect()
# 输出:
# {
#   "run_summaries": [
#     {"run_id": "auto_abc123", "total_tasks": 100, "successful": 95, "failed": 5, "pending": 0},
#     {"run_id": "auto_def456", "total_tasks": 50, "successful": 50, "failed": 0, "pending": 0},
#   ]
# }

# 查看特定运行的详情
run_info = cache.inspect(run_id="auto_abc123")
# 输出:
# {
#   "run_id": "auto_abc123",
#   "total_tasks": 100,
#   "successful": 95,
#   "failed": 5,
#   "pending": 0
# }
```

#### 3.2 清空缓存
```python
# 清空特定运行的结果
cache.clear(run_id="auto_abc123", confirm=True)
# 输出: ✓ 已清空 run_id='auto_abc123' 的缓存

# 清空所有缓存（谨慎使用！）
cache.clear(confirm=True)
# 输出: ✓ 已清空所有缓存
```

#### 3.3 回滚已完成任务（重新运行）
```python
# 将已完成的任务标记为待处理状态，下次 map() 时会重新运行
cache.rollback(run_id="auto_abc123", confirm=True)
# 输出: ✓ 已回滚 95 个任务为待处理状态

# 之后再调用 map 时，会重新处理这些任务
results = mapper.map(inputs, run_id="auto_abc123")
```

---

## 🎯 优化 4: 自动初始化异步客户端

### 问题
异步 API 需要手动初始化 `AsyncOpenAI`，配置复杂。

### 新功能

#### 4.1 自动初始化（新默认行为）
```python
from silkloom_core import PromptMapper
import asyncio

# ✓ 现在自动初始化 AsyncOpenAI（如果已安装）
mapper = PromptMapper(
    model="gpt-4",
    user_prompt="{text}",
    # 不需要显式传递 async_client！
)

# 直接使用异步 API
async def process_batch():
    results = await mapper.amap(
        ["文本1", "文本2", "文本3"],
        max_concurrent=5
    )
    return results

# 运行异步任务
results = asyncio.run(process_batch())
```

#### 4.2 手动配置（仍然支持）
```python
from openai import AsyncOpenAI

custom_async_client = AsyncOpenAI(api_key="sk-...")

mapper = PromptMapper(
    model="gpt-4",
    user_prompt="{text}",
    async_client=custom_async_client  # 覆盖默认自动初始化
)
```

---

## 🎯 优化 5: 改进错误处理与反馈

### 问题
错误消息不清晰，用户难以快速定位和修复问题。

### 改进示例

#### 配置错误
```python
# ❌ 错误：max_retries < 1
pm = PromptMapper(model="gpt-4", user_prompt="{text}", max_retries=0)
# 抛出 ConfigurationError，附带解决建议：
# "max_retries must be >= 1."

# ✓ 正确
pm = PromptMapper(model="gpt-4", user_prompt="{text}", max_retries=3)
```

#### 模板渲染错误
```python
# ❌ 错误：输入缺少 'keywords' 字段
try:
    result = pm.run_one({"title": "AI", "abstract": "..."})  # 缺少 keywords
except TemplateRenderError as e:
    # 错误提示会包含如何修复的建议
    print(f"❌ {e}")
    # "Prompt template rendering failed."
    # 解决方案：检查输入数据是否包含模板中所有必需的变量
    # 使用 mapper.required_variables 查看模板需要的字段
```

#### 结构化输出解析失败
```python
# ❌ 当输出无法解析为 response_model
try:
    results = pm.map(inputs)  # response_model 指定为 PaperSummary
except ResponseParseError as e:
    # 错误提示会建议查看 raw_output
    print(f"❌ {e}")
    # 查看实际模型输出
    for result in results:
        if not result.is_success:
            print("模型实际输出:", result.raw_output)
```

---

## 📚 完整使用示例：学术论文批处理

```python
from silkloom_core import PromptMapper, CacheManager
from pydantic import BaseModel
import asyncio

# 定义输出结构
class PaperAnalysis(BaseModel):
    summary: str
    keywords: list[str]
    novelty_score: float

# 创建 mapper（使用简单格式模式）
mapper = PromptMapper(
    model="gpt-4-turbo",
    system_prompt="你是学术文献分析专家。",
    user_prompt="""{title}

摘要：{abstract}

请分析这篇论文并输出 JSON 格式：
{{
    "summary": "...",
    "keywords": ["...", "..."],
    "novelty_score": 0.0-1.0
}}""",
    response_model=PaperAnalysis,
    simple_format_mode=True,
    temperature=0.3
)

# 准备数据
papers = [
    {
        "title": "Vision Transformers",
        "abstract": "We introduce an image recognition model based on transformer architecture..."
    },
    {
        "title": "BERT: Pre-training",
        "abstract": "A method for pre-training deep bidirectional transformers..."
    },
]

# 1️⃣ 验证输入
if not mapper.validate_inputs(papers):
    print("输入数据有问题，请检查")
    exit(1)

# 2️⃣ 批量处理（同步）
results = mapper.map(
    papers,
    workers=4,
    show_progress=True
)

# 3️⃣ 分析结果
print(f"✓ 成功: {results.success_count}")
print(f"✗ 失败: {results.failed_count}")
print(f"📊 总 tokens: {results.total_tokens}")

# 4️⃣ 过滤和转换
successful = results.filter(lambda r: r.is_success)
novelties = successful.map(lambda r: r.data.novelty_score)
print(f"平均新颖度: {sum(novelties) / len(novelties):.2f}")

# 5️⃣ 导出结果
results.export_csv("papers_analysis.csv", flatten=False, include_usage=True)
results.export_jsonl("papers_analysis.jsonl")

# 6️⃣ 管理缓存（下次运行时会复用）
cache = CacheManager()
status = cache.inspect()
print(f"缓存状态: {status}")

# 7️⃣ 异步处理（可选，用于更大规模批处理）
async def async_process():
    async_results = await mapper.amap(
        papers,
        max_concurrent=10,
        show_progress=True
    )
    return async_results

# async_results = asyncio.run(async_process())
```

---

## 🔄 迁移指南

### 从旧版本升级到 v4.1+

#### ✓ 完全向后兼容
```python
# 旧代码仍然完全工作，无需改动！
pm = PromptMapper(
    model="gpt-4",
    user_prompt="{{ text }}"  # 仍然支持标准 Jinja2 语法
)
results = pm.map(["A", "B", "C"], workers=3)
print(results.success_count)
```

#### 🆕 建议采用新特性
```python
# 新特性：更清晰的参数映射
pm = PromptMapper(
    model="gpt-4",
    user_prompt="{text}",
    simple_format_mode=True
)

# 新特性：验证输入
if pm.validate_inputs(data):
    results = pm.map(data)

# 新特性：链式操作
successful = (results
    .filter(lambda r: r.is_success)
    .transform(lambda d: d.upper())
)

# 新特性：缓存管理
cache = CacheManager()
print(cache.inspect())
```

---

## 🚀 性能提示

### 内存优化
```python
# 对于超大批次，使用分批处理避免内存溢出
import math

papers = [...]  # 10,000 篇论文
batch_size = 1000

all_results = []
for i in range(0, len(papers), batch_size):
    batch = papers[i : i + batch_size]
    results = mapper.map(batch, workers=8, run_id=f"batch_{i}")
    all_results.extend(results.results)
```

### 并发优化
```python
# 同步处理：增加 workers（线程数）
results = mapper.map(data, workers=8)  # 默认 5

# 异步处理：更高的并发（用于网络 I/O 密集）
results = await mapper.amap(data, max_concurrent=20)  # 默认 5
```

---

## ❓ 常见问题

**Q: 我的 Jinja2 模板用来做什么？**
A: 模板定义了如何将输入数据转化为发送给 LLM 的提示词。`{{ variable }}` 会被替换为相应的输入值。

**Q: `simple_format_mode=True` 和标准 Jinja2 有区别吗？**
A: 功能完全相同，只是 `{var}` 比 `{{ var }}` 更简洁直观。建议在学术文本处理中使用。

**Q: 缓存何时自动触发？**
A: 每次调用 `map()` 或 `amap()` 时，引擎会检查数据库。已成功的任务不会重新运行。

**Q: 数据库文件可以删除吗？**
A: 可以，但删除后所有缓存都会丢失。下次运行会从零开始。建议定期备份。

**Q: 链式操作的性能如何？**
A: `filter`, `map`, `transform` 都是内存操作，非常快。建议在结果小于 100,000+ 项时使用。

---

## 📝 版本历史

- **v4.1** (2026-05-14)
  - ✨ 添加 `simple_format_mode` 简化参数映射
  - ✨ 新增 `.required_variables` 属性和 `.validate_inputs()` 方法
  - ✨ 添加 `.filter()`, `.map()`, `.transform()` 链式操作
  - ✨ 新增 `CacheManager` 用于缓存管理
  - ✨ 自动初始化 `AsyncOpenAI`
  - 📖 改进异常消息，增加解决建议
  
- v4.0：初始版本

---

**反馈与建议**：欢迎在项目 issue 中提出改进建议！
