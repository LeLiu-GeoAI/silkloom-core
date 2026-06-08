from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

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


class TextAnalysis(BaseModel):
    sentiment: str
    keywords: list[str]


def main() -> None:
    load_env()

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("BASE_URL"),
    )

    raw = Loom(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        prompt="Rewrite this sentence in a more formal academic tone: {{ text }}",
        client=client,
        temperature=0.2,
    )
    print(raw.run("We found that this method works pretty well.").unwrap())

    structured = Loom(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        prompt=(
            "Analyze the text and return JSON with keys sentiment and keywords. "
            "Text: {{ text }}"
        ),
        output=TextAnalysis,
        client=client,
        temperature=0.1,
    )
    analysis = structured.run("The paper is clear, but the evaluation is too small.").unwrap()
    print(analysis)

    items = [
        "The implementation is reliable.",
        "The experiment section needs more details.",
        "The result is promising but not conclusive.",
    ]

    batch = raw.batch(items, name="simple_rewrite_v1", concurrency=3)
    print(batch.values())

    for result in raw.each(items, name="simple_rewrite_v1", concurrency=3):
        print("cached" if result.cache_hit else "fresh", result.unwrap())


if __name__ == "__main__":
    main()
