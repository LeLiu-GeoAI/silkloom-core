---
name: silkloom-core
description: Batch LLM extraction on pandas DataFrames. Use when the user wants to run LLM extraction, classification, or structured-data extraction across many rows of a DataFrame concurrently, with caching and JSON parsing.
version: 1.0.0
---

# SilkLoom Core

Batch LLM extraction on pandas DataFrames — `df.llm.extract(template)` concurrently calls an OpenAI-compatible API for every row, parses JSON, and returns a result DataFrame.

## Install

```bash
pip install silkloom-core

# optional: progress bar
pip install silkloom-core[progress]
```

## Usage

### 1. Configure (once per script)

```python
import pandas as pd
import silkloom_core
from openai import OpenAI

silkloom_core.configure(
    client=OpenAI(api_key="...", base_url="https://api.openai.com/v1"),
    cache_path=".llm_cache.db",
)
```

### 2. Extract

```python
df = pd.DataFrame({
    "title": ["Paper A", "Paper B"],
    "abstract": ["Reliable results.", "Small sample size."],
})

results = df.llm.extract(
    "Title: {{ title }}\nAbstract: {{ abstract }}\nReturn JSON with keys label and confidence.",
    model="gpt-4o-mini",
    max_workers=8,
    json_mode=True,
)

df = df.join(results)
```

### 3. Multiple providers (optional)

```python
silkloom_core.configure(client=openai_client)
df.llm.extract("...", model="gpt-4o")                           # OpenAI
df.llm.extract("...", model="glm-4-flash", client=zhipu_client) # Zhipu
```

## Key points

- **Template engine**: Jinja2 with `StrictUndefined` — column name typos raise immediately.
- **Result columns**: JSON object keys become columns; non-object values go to `_llm_raw`; errors go to `_llm_error`.
- **Cache**: SQLite with full audit trail; successful responses are reused automatically.
- **Images**: Pass `image_column=` for local paths, HTTP(S) URLs, or `data:image/...` URLs.
- **Cancel**: Call `df.llm.cancel()` from another thread to stop in-flight work.
- **Client priority**: `extract(client=)` > `setup(client=)` > `configure(client=)` > error.
