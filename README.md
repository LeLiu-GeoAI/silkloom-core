# SilkLoom Core

SilkLoom Core is a DataFrame-first batch runner for OpenAI-compatible LLM and VLM workloads.

Its core shape is:

```text
indexed data -> selected input columns -> model calls -> repaired JSON -> wider DataFrame
```

## Install

```bash
pip install silkloom-core
```

## Quick Start

```python
import pandas as pd
from silkloom_core import Loom

df = pd.read_csv("papers.csv")

loom = Loom(
    model="gpt-4o-mini",
    prompt="""
Analyze this paper and return JSON only.

Title: {{ title }}
Abstract: {{ abstract }}

Return keys: sentiment, summary, keywords.
""",
)

df = loom(
    df,
    input=["title", "abstract"],
    resume="paper_analysis_v1",
    concurrency=10,
)

df.to_csv("papers_with_analysis.csv", index=False)
```

The returned DataFrame keeps the original index and columns, then appends model output columns.

## Inputs

Use a pandas DataFrame:

```python
out = loom(df, input=["title", "abstract"])
```

Or a sequence of row mappings:

```python
out = loom(
    [
        {"text": "The experiment is promising."},
        {"text": "The evaluation is incomplete."},
    ],
    input="text",
)
```

The output is always a pandas DataFrame.

## Output

By default, SilkLoom expects JSON object output and expands its keys into columns.

```python
loom = Loom(
    model="gpt-4o-mini",
    prompt="Classify this text and return JSON with keys label and score: {{ text }}",
)

out = loom(df, input="text")
```

If the model returns:

```json
{"label": "positive", "score": 0.92}
```

SilkLoom appends:

```text
label | score
```

Malformed JSON is repaired with `json_repair.repair_json(..., return_objects=True)`.

## Status Columns

SilkLoom appends status columns by default:

```text
_loom_ok
_loom_error
_loom_checkpoint
_loom_attempts
```

You can also include raw model output and extracted reasoning:

```python
out = loom(
    df,
    input="text",
    include_output=True,
    include_reasoning=True,
)
```

This adds:

```text
_loom_output
_loom_reasoning
```

Disable status columns with:

```python
out = loom(df, input="text", status=False)
```

## Progress

Show a tqdm progress bar:

```python
out = loom(
    df,
    input="text",
    progress=True,
)
```

Use a custom progress label:

```python
out = loom(
    df,
    input="text",
    progress="Analyzing papers",
)
```

For UI frameworks such as Gradio, use `on_progress`. The callback receives `completed`, `total`, and the full checkpoint-style row state.

```python
import gradio as gr

def analyze(file, progress=gr.Progress()):
    df = pd.read_csv(file.name)

    def update(done, total, state):
        progress(done / total, desc=f"{done}/{total}")

    return loom(
        df,
        input="text",
        resume="gradio_text_analysis_v1",
        on_progress=update,
    )
```

## Resumable Runs

Pass `resume` to enable SQLite checkpointing. SQLite is the default checkpoint backend.

```python
out = loom(
    df,
    input=["title", "abstract"],
    resume="paper_analysis_v1",
    concurrency=10,
)
```

The checkpoint fingerprint includes the selected row input, model, prompt, system message, output schema, and model parameters.

Each SQLite row stores a self-describing JSON payload with:

- SilkLoom version
- resume namespace
- model, prompt, system message, model parameters, retry settings, and JSON repair setting
- selected input columns and image columns
- normalized row input, including resolved `images`
- rendered OpenAI-compatible message payload
- parsed JSON result
- raw model output, extracted reasoning, error trace, attempt count, and checkpoint status

To disable checkpointing:

```python
loom = Loom(..., checkpoint=None)
```

## Column Conflicts

If model output columns conflict with existing DataFrame columns, SilkLoom raises. Use `prefix` when you want to keep both.

```python
out = loom(df, input="text", prefix="llm_")
```

## Images

If an input row contains an `images` column, SilkLoom builds a multimodal OpenAI-compatible message. HTTP(S) and `data:image/...` URLs are passed through. Local files are converted to base64 data URLs.

```python
df = pd.DataFrame(
    {
        "instruction": ["Extract menu item names and prices."],
        "images": [["./receipt.jpg"]],
    }
)

out = loom(df, input="instruction", images="images")
```

You can also pass multiple image columns:

```python
out = loom(df, input="instruction", images=["front_image", "back_image"])
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
