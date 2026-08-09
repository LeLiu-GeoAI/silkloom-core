# SilkLoom Core

A lightweight pandas accessor for batch LLM extraction.

```text
DataFrame rows → Jinja2 prompt render → OpenAI-compatible API → repaired JSON → result DataFrame
```

One call — `df.llm.extract(template)` — concurrently sends every row to an LLM, parses the JSON response, and returns a DataFrame you can `join` back to the original.

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Global Configuration](#global-configuration)
  - [Per-DataFrame Setup](#per-dataframe-setup)
  - [Per-Call Client](#per-call-client)
  - [API Key Rotation](#api-key-rotation)
  - [Priority Chain](#priority-chain)
- [Extraction](#extraction)
  - [Prompt Templates](#prompt-templates)
  - [Result Columns](#result-columns)
- [Concurrency and Tuning](#concurrency-and-tuning)
- [Cache and Audit](#cache-and-audit)
- [Images](#images)
- [Progress and Cancel](#progress-and-cancel)
- [API Reference](#api-reference)

## Install

```bash
pip install silkloom-core

# optional: progress bar
pip install silkloom-core[progress]
```

Importing `silkloom_core` registers the `df.llm` accessor on every DataFrame.

## Quick Start

```python
import pandas as pd
import silkloom_core

silkloom_core.configure(api_key="...", base_url="https://api.openai.com/v1")

df = pd.DataFrame({
    "title": ["A clear experiment", "A weak evaluation"],
    "abstract": ["Reliable and reproducible.", "Too small to conclude much."],
})

df = df.llm.extract(
    "Title: {{ title }}\nAbstract: {{ abstract }}\nReturn JSON with keys label and summary.",
    model="gpt-4o-mini",
    max_workers=8,
    json_mode=True,
)
# df now has: title, abstract, label, summary
```

`extract()` returns the original DataFrame with extracted columns appended. Pass `join=False` if you only want the extracted columns.

## Configuration

SilkLoom supports three layers of client configuration, from broadest to narrowest.

### Global Configuration

Configure once at the start of a script — every DataFrame can call `extract()` directly:

```python
silkloom_core.configure(
    api_key="...",
    base_url="https://api.openai.com/v1",
)

# Or pass a pre-built client:
silkloom_core.configure(client=OpenAI(api_key="...", base_url="..."))
```

### Per-DataFrame Setup

Override the global default for a specific DataFrame:

```python
df.llm.setup(
    api_key="...",
    base_url="...",
    cache_path="special_cache.db",
)

# Chain directly into extract:
df.llm.setup(client=client).extract("...", model="gpt-4o-mini")
```

### Per-Call Client

Override the client for a single `extract()` call — useful for mixing providers:

```python
from openai import OpenAI

openai_client = OpenAI(api_key="...", base_url="https://api.openai.com/v1")
zhipu_client = OpenAI(api_key="...", base_url="https://open.bigmodel.cn/api/paas/v4")

silkloom_core.configure(client=openai_client)

# Same DataFrame, different providers:
df.llm.extract("...", model="gpt-4o")                          # → OpenAI
df.llm.extract("...", model="glm-4-flash", client=zhipu_client) # → Zhipu
```

### API Key Rotation

When `api_key` contains `|`, the keys are split and wrapped in a `KeyRotatingClient` that distributes API calls round-robin across all keys — useful for staying under rate limits without manual load balancing:

```python
# 3 keys → each request goes to the next key in rotation
silkloom_core.configure(
    api_key="key1|key2|key3",
    base_url="https://api.openai.com/v1",
)

df.llm.extract("{{ text }}", max_workers=8)  # 8 threads, 3 keys → ~2-3 calls per key
```

Works with `setup()` too:

```python
df.llm.setup(
    api_key="key1|key2|key3",
    base_url="https://api.openai.com/v1",
).extract("{{ text }}")
```

You can also build a `KeyRotatingClient` manually — for example, to mix keys from different providers:

```python
from openai import OpenAI
from silkloom_core import KeyRotatingClient

client = KeyRotatingClient([
    OpenAI(api_key="openai-key", base_url="https://api.openai.com/v1"),
    OpenAI(api_key="deepseek-key", base_url="https://api.deepseek.com/v1"),
])
silkloom_core.configure(client=client)
```

> **Tip:** Combine key rotation with `max_workers` to parallelize across keys. With 3 keys and `max_workers=9`, each key handles ~3 concurrent requests.

### Priority Chain

```
extract(client=...)  >  df.llm.setup(client=...)  >  silkloom_core.configure(client=...)  >  error
```

If no client is configured, `extract()` raises `RuntimeError` with a clear message.

> **Note:** The SQLite cache file is created lazily — only on the first `extract()` call, never as a side effect of `configure()` or `setup()`.

## Extraction

### Prompt Templates

Prompts use [Jinja2](https://jinja.palletsprojects.com/) with `StrictUndefined` — typos in column names raise immediately instead of silently producing empty strings. Literal JSON braces (`{` `}`) are safe; only `{{ }}` is treated as a template expression.

```python
out = df.llm.extract(
    'Classify {{ text }} and return JSON like {"label": "positive", "score": 0.9}',
    model="gpt-4o-mini",
    temperature=0.1,
    max_workers=4,
    max_retries=2,
)
```

### Result Columns

The returned DataFrame has the same index as the input. Column semantics:

| Condition | Column(s) |
|---|---|
| Model returns a JSON object | Each key becomes a column |
| Model returns a non-object JSON value | `_llm_raw` |
| JSON parse fails | `_llm_error` + `_llm_raw` |
| API call fails after all retries | `_llm_error` |

Malformed JSON is repaired with [`json_repair`](https://github.com/mangiucugna/json_repair) before parsing.

## Concurrency and Tuning

All tuning parameters are passed directly to `extract()` — no separate config object needed.

### Concurrency (`max_workers`)

Controls the thread pool size for parallel API calls. Default is 4.

```python
# 16 concurrent threads — faster, but watch out for rate limits
df.llm.extract("{{ text }}", max_workers=16)

# Sequential — useful for debugging or strict rate-limit scenarios
df.llm.extract("{{ text }}", max_workers=1)
```

**Guideline:** Set `max_workers` according to your provider's rate limit (RPM). For OpenAI's default tier, 4–8 is safe; for high-volume providers like DeepSeek, 8–16 works well. If you hit `429 Too Many Requests`, lower the value or increase `max_retries`.

### Retries (`max_retries`)

On API errors (timeouts, rate limits, server errors), the request is retried with exponential backoff (1s → 2s → 4s …). Default is 2 retries.

```python
# More resilient for flaky endpoints
df.llm.extract("{{ text }}", max_retries=5)
```

### API Parameters (`**request_options`)

Any keyword argument not recognized by `extract()` is forwarded directly to `client.chat.completions.create()`. Common ones:

| Parameter | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature (0–2). Lower = more deterministic. |
| `max_tokens` | int | Maximum tokens to generate in the response. |
| `top_p` | float | Nucleus sampling probability. |
| `frequency_penalty` | float | Penalize repeated tokens (-2–2). |
| `presence_penalty` | float | Penalize tokens already present (-2–2). |
| `seed` | int | Random seed for reproducibility (provider-dependent). |
| `stop` | str \| list[str] | Stop sequences. |

```python
out = df.llm.extract(
    "Classify {{ text }} into positive/negative/neutral.",
    model="gpt-4o-mini",
    max_workers=8,
    temperature=0.1,
    max_tokens=200,
    top_p=0.9,
    seed=42,
)
```

### Combined Example

```python
silkloom_core.configure(api_key="...", base_url="https://api.deepseek.com/v1")

result = df.llm.extract(
    "Analyze: {{ content }}\nReturn JSON with keys sentiment and confidence.",
    model="deepseek-chat",
    max_workers=12,        # 12 concurrent threads
    max_retries=3,         # retry up to 3 times on failure
    temperature=0.2,       # low temperature for consistent output
    max_tokens=500,        # cap response length
    json_mode=True,        # force JSON output
    system_prompt="You are a sentiment analysis engine.",
)
```

## Cache and Audit

Every API call is recorded in a SQLite database. Successful responses (`ok = 1`) are reused as cache hits on subsequent runs with the same request. Failed requests are also stored for debugging but are retried next time.

```python
df.llm.setup(cache_path="cache/llm.sqlite").extract(...)
```

The `cache` table schema:

| Column | Type | Description |
|---|---|---|
| `cache_key` | TEXT PK | SHA-256 of the full request |
| `ok` | INTEGER | 1 = success (cacheable), 0 = failure |
| `model` | TEXT | Model name used |
| `messages_json` | TEXT | Rendered messages array |
| `params_json` | TEXT | Request params (excluding model/messages) |
| `request_json` | TEXT | Full request payload |
| `response` | TEXT | Raw model response text |
| `parsed_json` | TEXT | Parsed result dict |
| `error` | TEXT | Error message (NULL on success) |
| `attempts` | INTEGER | Number of attempts made |
| `created_at` | TEXT | Row creation timestamp |
| `updated_at` | TEXT | Last update timestamp |

To start fresh: delete the SQLite file or use a different `cache_path`.

## Images

Pass `image_column` for local file paths, HTTP(S) URLs, or `data:image/...` URLs. Local files are auto-encoded as base64 data URLs with MIME detection. A single cell can hold a list of images for multi-image input.

```python
out = df.llm.extract(
    "Extract fields from this receipt and return JSON.",
    image_column="receipt_path",
    model="gpt-4o-mini",
)
```

Rows with missing image values (`NaN`, `None`) fall back to text-only prompts.

## Progress and Cancel

**Progress bar** — tqdm is used when `verbose=True` (default). If tqdm isn't installed, it degrades silently.

**Callback** — for UI integration:

```python
def progress(done, total):
    print(f"{done}/{total}")

out = df.llm.extract("Analyze {{ text }}", progress_callback=progress)
```

**Cancel** — from another thread:

```python
df.llm.cancel()
```

Queued work is cancelled where possible. Running rows stop before their next retry. Already-completed results are preserved in the returned DataFrame.

## API Reference

### `silkloom_core.configure(...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | API key. If contains `\|`, split into multiple keys for round-robin rotation |
| `base_url` | `str \| None` | `None` | API base URL |
| `cache_path` | `str \| Path` | `".llm_cache.db"` | SQLite cache file path |
| `client` | `Any \| None` | `None` | Pre-built client (overrides api_key/base_url) |
| `**client_options` | | | Extra kwargs for `OpenAI()` |

### `df.llm.setup(...)`

Same parameters as `configure()`. Returns `self` for chaining.

### `df.llm.extract(prompt_template, *, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt_template` | `str` | — | Jinja2 template (required) |
| `client` | `Any \| None` | `None` | Per-call client override |
| `image_column` | `str \| None` | `None` | Column with image paths/URLs |
| `system_prompt` | `str \| None` | `"Please output valid JSON only."` | System message; `None` to omit |
| `model` | `str` | `"gpt-4o-mini"` | Model name |
| `max_workers` | `int` | `4` | Concurrent API call threads |
| `json_mode` | `bool` | `False` | Set `response_format={"type":"json_object"}` |
| `max_retries` | `int` | `2` | Retries on API error (exponential backoff) |
| `join` | `bool` | `True` | If True, return original DataFrame with extracted columns appended; if False, return only extracted columns |
| `progress_callback` | `Callable[[int, int], None] \| None` | `None` | Called with (completed, total) |
| `verbose` | `bool` | `True` | Show tqdm progress bar |
| `**request_options` | | | Extra kwargs for `chat.completions.create()` |

### `df.llm.cancel()`

No parameters. Signals cancellation to all in-flight work.

### `silkloom_core.KeyRotatingClient(clients)`

Wraps multiple OpenAI-compatible clients and rotates through them round-robin. Thread-safe.

| Parameter | Type | Description |
|---|---|---|
| `clients` | `list[Any]` | List of OpenAI-compatible client instances |

```python
from silkloom_core import KeyRotatingClient
from openai import OpenAI

client = KeyRotatingClient([
    OpenAI(api_key="key1", base_url="https://api.openai.com/v1"),
    OpenAI(api_key="key2", base_url="https://api.openai.com/v1"),
])
silkloom_core.configure(client=client)
```
