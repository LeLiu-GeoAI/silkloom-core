from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from silkloom_core import Loom


def load_env(path: str = ".env") -> None:
    env = Path(path)
    if not env.exists():
        return

    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    load_env()

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("BASE_URL"),
    )

    df = pd.DataFrame(
        {
            "text": [
                "The paper is clear, but the evaluation is too small.",
                "The implementation is reliable and easy to reproduce.",
            ]
        }
    )

    loom = Loom(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        prompt=(
            "Analyze the text and return JSON only with keys "
            "sentiment, summary, and keywords. Text: {{ text }}"
        ),
        client=client,
        temperature=0.1,
    )

    out = loom(
        df,
        input="text",
        resume="simple_text_analysis_v1",
        concurrency=3,
    )
    print(out)


if __name__ == "__main__":
    main()
