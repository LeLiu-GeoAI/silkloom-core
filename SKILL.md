---
name: silkloom-core
description: Batch LLM extraction on pandas DataFrames. Use when the user wants to run LLM extraction, classification, or structured-data extraction across many rows of a DataFrame concurrently, with caching and JSON parsing.
version: 1.1.0
---

# SilkLoom Core

Batch LLM extraction on pandas DataFrames — `df.llm.extract(template)` concurrently calls an OpenAI-compatible API for every row, parses JSON, and returns the DataFrame with extracted columns appended.

## Install

```bash
pip install silkloom-core
```

## Usage

```python
import pandas as pd
import silkloom_core

# 1. Configure (once per script)
silkloom_core.configure(api_key="...", base_url="https://api.openai.com/v1")

# 2. Extract — results are joined into the DataFrame automatically
df = pd.DataFrame({
    "title": ["Paper A", "Paper B"],
    "abstract": ["Reliable results.", "Small sample size."],
})

df = df.llm.extract(
    "Title: {{ title }}\nAbstract: {{ abstract }}\nReturn JSON with keys label and confidence.",
    model="gpt-4o-mini",
)
# df now has: title, abstract, label, confidence
```

### Multiple providers

```python
silkloom_core.configure(api_key="...", base_url="https://api.openai.com/v1")
df.llm.extract("...", model="gpt-4o")
df.llm.extract("...", model="glm-4-flash", client=zhipu_client)  # per-call override
```

## Key points

- **Auto-join**: `extract()` returns the original DataFrame with extracted columns appended. Pass `join=False` for results only.
- **Template engine**: Jinja2 with `StrictUndefined` — column name typos raise immediately.
- **Result columns**: JSON object keys become columns; non-object values go to `_llm_raw`; errors go to `_llm_error`.
- **Cache**: SQLite with full audit trail; successful responses are reused automatically.
- **Images**: Pass `image_column=` for local paths, HTTP(S) URLs, or `data:image/...` URLs.
- **Cancel**: Call `df.llm.cancel()` from another thread to stop in-flight work.
- **Client priority**: `extract(client=)` > `setup(client=)` > `configure(client=)` > error.
