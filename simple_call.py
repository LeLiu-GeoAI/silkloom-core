from __future__ import annotations

import pandas as pd
from openai import OpenAI

import silkloom_core


def main() -> None:
    client = OpenAI(
        api_key="071ab49a8174143d6cc2a19d287b61ee.BenBTW0a0JlkeVmJ",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    silkloom_core.configure(client=client)

    df = pd.DataFrame(
        {
            "text": [
                "The paper is clear, but the evaluation is too small.",
                "The implementation is reliable and easy to reproduce.",
            ]
        }
    )

    extracted = df.llm.extract(
        "Analyze the text and return JSON with keys sentiment, summary, and keywords. Text: {{ text }}",
        model="glm-4-flash",
        temperature=0.1,
        max_workers=3,
    )

    print(df.join(extracted))


if __name__ == "__main__":
    main()
