# SilkLoom Core

SilkLoom Core is a small batch engine for OpenAI-compatible LLM and VLM calls.
It focuses on the parts that become painful in production scripts:

- Jinja prompt rendering
- text and image message assembly
- raw text, JSON, and Pydantic output parsing
- retry capture with raw output and reasoning extraction
- ordered batch execution
- completion-order streaming
- SQLite checkpoints for resumable runs
- sync and async APIs with the same shape

## Install

```bash
pip install silkloom-core
```

Optional extras:

```bash
pip install "silkloom-core[data]"      # pandas export
pip install "silkloom-core[progress]"  # tqdm, if you build your own progress UI
```

## The API

```python
from silkloom_core import Loom

loom = Loom(
    model="gpt-4o-mini",
    prompt="Summarize this text in one sentence: {{ text }}",
    temperature=0.2,
)

result = loom.run({"text": "SilkLoom is a batch engine for model workloads."})
print(result.value)
```

`Loom` is the only high-level object. It has six execution methods:

- `run(data)` runs one item.
- `arun(data)` runs one item asynchronously.
- `batch(items, name=None, concurrency=5)` returns a `Batch` in input order.
- `abatch(items, name=None, concurrency=5)` is the async batch API.
- `each(items, name=None, concurrency=5, ordered=False)` yields `Result` objects.
- `aeach(items, name=None, concurrency=5, ordered=False)` is the async streaming API.

Input can be a string or a dictionary. Strings are normalized to `{"text": value}`.

## Structured Output

Use `output=dict` for JSON objects:

```python
from silkloom_core import Loom

loom = Loom(
    model="gpt-4o-mini",
    prompt="Return JSON with keys sentiment and keywords for: {{ text }}",
    output=dict,
)

result = loom.run("The method is useful, but the experiments are weak.")
print(result.unwrap()["keywords"])
```

Use a Pydantic model for validated typed output:

```python
from pydantic import BaseModel
from silkloom_core import Loom

class Review(BaseModel):
    sentiment: str
    keywords: list[str]

loom = Loom(
    model="gpt-4o-mini",
    prompt="Return JSON matching the schema for: {{ text }}",
    output=Review,
)

review = loom.run({"text": "Clear writing, limited evaluation."}).unwrap()
print(review.sentiment)
```

SilkLoom extracts fenced JSON, can repair malformed JSON with `json-repair`, and separates DeepSeek-style `<think>...</think>` reasoning into `Result.reasoning`.

## Checkpointed Batch Runs

Pass `name` to enable SQLite checkpointing. The cache key includes the input, model, prompt, system message, output schema, and model parameters.

```python
from silkloom_core import Loom

loom = Loom(
    model="gpt-4o-mini",
    prompt="Rewrite formally: {{ text }}",
    cache=".silkloom.db",
)

items = [
    "This works pretty well.",
    "The experiment needs more detail.",
]

batch = loom.batch(items, name="formal_rewrite_v1", concurrency=8)
print(batch.values())

# A later run with the same namespace and configuration reuses successful results.
again = loom.batch(items, name="formal_rewrite_v1", concurrency=8)
print([item.cache_hit for item in again])
```

## Streaming

Use completion-order streaming for responsive UIs:

```python
for result in loom.each(items, name="formal_rewrite_v1", concurrency=8):
    print(result.ok, result.value)
```

Use `ordered=True` when consumers require input order:

```python
for result in loom.each(items, concurrency=8, ordered=True):
    print(result.value)
```

Async streaming has the same shape:

```python
async for result in loom.aeach(items, name="formal_rewrite_v1", concurrency=20):
    print(result.value)
```

## Images

If an input dictionary contains `images`, SilkLoom builds a multimodal OpenAI-compatible message. HTTP(S) and `data:image/...` URLs are passed through. Local files are converted to base64 data URLs.

```python
loom = Loom(
    model="gpt-4o-mini",
    prompt="Extract the menu item names and prices. Instruction: {{ instruction }}",
    output=dict,
)

result = loom.run(
    {
        "instruction": "Return JSON only.",
        "images": ["./receipt.jpg", "https://example.com/menu.png"],
    }
)
```

## Result Objects

Each run returns a `Result`:

```python
result.ok          # bool
result.value       # parsed value, or None on failure
result.error       # traceback string on failure
result.input       # normalized input dict
result.output      # raw model text
result.reasoning   # extracted <think> content, if present
result.cache_hit   # true when loaded from SQLite
result.attempts    # number of attempts used
```

`result.unwrap()` returns `value` or raises if the run failed.

`Batch` is an iterable container with helpers:

```python
batch.successful()
batch.failed()
batch.values()
batch.to_dicts()
batch.to_pandas()
```

## Custom Clients

By default, `Loom` creates OpenAI SDK clients. You can also pass an existing OpenAI-compatible client:

```python
from openai import OpenAI
from silkloom_core import Loom

client = OpenAI(base_url="https://api.example.com/v1", api_key="...")

loom = Loom(
    model="provider-model",
    prompt="{{ text }}",
    client=client,
)
```

For full control, pass an object that implements:

```python
class MyClient:
    def complete(self, *, model, messages, params) -> str: ...
    async def acomplete(self, *, model, messages, params) -> str: ...
    def close(self) -> None: ...
    async def aclose(self) -> None: ...
```
