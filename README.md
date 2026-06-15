# SilkLoom Core

SilkLoom Core is a small pandas accessor for batch LLM extraction.

```text
DataFrame rows -> Jinja prompt render -> OpenAI-compatible chat call -> repaired JSON -> result DataFrame
```

## Install

```bash
pip install silkloom-core
```

## Quick Start

Importing `silkloom_core` registers `df.llm`.

```python
import pandas as pd
import silkloom_core

df = pd.DataFrame(
    {
        "title": ["A clear experiment", "A weak evaluation"],
        "abstract": ["Reliable and reproducible.", "Too small to conclude much."],
    }
)

results = df.llm.setup(
    api_key="...",
    base_url="https://api.openai.com/v1",
    cache_path=".llm_cache.db",
).extract(
    "Title: {{ title }}\nAbstract: {{ abstract }}\nReturn JSON with keys label and summary.",
    model="gpt-4o-mini",
    max_workers=8,
    json_mode=True,
)
```

`results` contains only the parsed model output columns and keeps the original index, so you can join it back when needed:

```python
df = df.join(results)
```

## Client Setup

You can let SilkLoom create an OpenAI client:

```python
df.llm.setup(api_key="...", base_url="...")
```

Or pass any OpenAI-compatible client with `client.chat.completions.create(...)`:

```python
from openai import OpenAI

client = OpenAI(api_key="...", base_url="...")
df.llm.setup(client=client)
```

## Extraction

Use Jinja placeholders that match DataFrame columns. Literal JSON braces can stay as normal braces.

```python
out = df.llm.extract(
    'Classify {{ text }} and return JSON like {"label": "positive", "score": 0.9}',
    model="gpt-4o-mini",
    temperature=0.1,
    max_workers=4,
    max_retries=2,
    verbose=True,
)
```

Malformed JSON is parsed with `json_repair`. If the model returns a JSON object, its keys become columns. If it returns another JSON value, the value is placed in `_llm_raw`. Parse or request failures are returned in `_llm_error`.

## Cache And Audit Records

SQLite stores successful responses for cache reuse and also keeps richer request records for inspection. The cache key includes the model, rendered messages, JSON mode, and request options.

```python
df.llm.setup(cache_path="cache/llm.sqlite").extract(...)
```

The `cache` table includes:

- `cache_key`
- `ok`
- `model`
- `messages_json`
- `params_json`
- `request_json`
- `response`
- `parsed_json`
- `error`
- `attempts`
- `created_at`
- `updated_at`

Only rows with `ok = 1` are reused as cache hits. Failed requests and parse errors are recorded for debugging but are retried on the next run. Use a new cache path or delete the SQLite file when you want a fresh run.

## Images

Pass `image_column` for local image paths, HTTP(S) image URLs, or existing `data:image/...` URLs. Local files are encoded as base64 data URLs.

```python
out = df.llm.extract(
    "Extract fields from this receipt and return JSON.",
    image_column="receipt_path",
    model="gpt-4o-mini",
)
```

Rows with missing image values fall back to text-only prompts.

## Progress And Cancel

Use `progress_callback` for UI integration:

```python
def progress(done, total):
    print(done, total)

out = df.llm.extract("Analyze {{ text }}", progress_callback=progress)
```

From another thread or UI event, call:

```python
df.llm.cancel()
```

Queued work is cancelled where possible, and running rows stop before the next retry.
