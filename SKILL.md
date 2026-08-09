---
name: silkloom-core
description: Batch LLM extraction on pandas DataFrames. Use when the user wants to run LLM extraction, classification, or structured-data extraction across many rows of a DataFrame concurrently, with caching and JSON parsing.
version: 7.2.0
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

### API key rotation

Split multiple keys with `|` — each API call rotates to the next key round-robin:

```python
silkloom_core.configure(
    api_key="key1|key2|key3",
    base_url="https://api.openai.com/v1",
)
df.llm.extract("{{ text }}", max_workers=8)  # 8 threads, 3 keys → even distribution
```

Or build a `KeyRotatingClient` manually:

```python
from silkloom_core import KeyRotatingClient
client = KeyRotatingClient([client1, client2, client3])
silkloom_core.configure(client=client)
```

## Parameters

All tuning is done via `extract()` keyword arguments — no separate config object.

### `extract()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt_template` | `str` | — | Jinja2 template (required) |
| `client` | `Any` | `None` | Per-call client override |
| `image_column` | `str` | `None` | Column with image paths/URLs |
| `system_prompt` | `str` | `"Please output valid JSON only."` | System message; `None` to omit |
| `model` | `str` | `"gpt-4o-mini"` | Model name |
| `max_workers` | `int` | `4` | Concurrent API call threads |
| `json_mode` | `bool` | `False` | Set `response_format={"type":"json_object"}` |
| `max_retries` | `int` | `2` | Retries on API error (exponential backoff) |
| `join` | `bool` | `True` | If True, return original DataFrame + extracted columns; False = extracted only |
| `progress_callback` | `Callable` | `None` | Called with `(completed, total)` |
| `verbose` | `bool` | `True` | Show tqdm progress bar |
| `**request_options` | | | Extra kwargs forwarded to `chat.completions.create()` |

### Concurrency

```python
# 16 concurrent threads — adjust to provider's rate limit
df.llm.extract("{{ text }}", max_workers=16)

# Sequential (debugging or strict rate limits)
df.llm.extract("{{ text }}", max_workers=1)
```

Guideline: 4–8 for OpenAI default tier; 8–16 for high-volume providers. Lower the value if hitting `429`.

### API parameters

Any unrecognized kwarg is forwarded to `client.chat.completions.create()`:

```python
out = df.llm.extract(
    "Classify {{ text }}",
    model="gpt-4o-mini",
    max_workers=8,
    temperature=0.1,
    max_tokens=200,
    top_p=0.9,
    seed=42,
)
```

Common forwarded parameters: `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`.

## Key points

- **Auto-join**: `extract()` returns the original DataFrame with extracted columns appended. Pass `join=False` for results only.
- **Template engine**: Jinja2 with `StrictUndefined` — column name typos raise immediately.
- **Result columns**: JSON object keys become columns; non-object values go to `_llm_raw`; errors go to `_llm_error`.
- **Cache**: SQLite with full audit trail; successful responses are reused automatically.
- **Images**: Pass `image_column=` for local paths, HTTP(S) URLs, or `data:image/...` URLs.
- **Cancel**: Call `df.llm.cancel()` from another thread to stop in-flight work.
- **Client priority**: `extract(client=)` > `setup(client=)` > `configure(client=)` > error.
- **Key rotation**: Split keys with `|` in `api_key` (or use `KeyRotatingClient`) for round-robin load balancing across API keys.
