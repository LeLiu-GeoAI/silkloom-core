import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from silkloom_core import Loom


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeChat:
    def __init__(self, responses, delays=None):
        self.responses = list(responses)
        self.delays = delays or [0] * len(self.responses)
        self.index = 0
        self.lock = threading.Lock()

    def create(self, model, messages, **kwargs):
        try:
            content = messages[-1]["content"]
            if isinstance(content, str) and content.isdigit():
                index = int(content) - 1
                delay = self.delays[index] if index < len(self.delays) else 0
                if delay:
                    time.sleep(delay)
                return FakeResponse(self.responses[index])
        except Exception:
            pass

        with self.lock:
            index = self.index
            self.index += 1
        delay = self.delays[index] if index < len(self.delays) else 0
        if delay:
            time.sleep(delay)
        return FakeResponse(self.responses[index])


class FakeClient:
    def __init__(self, responses, delays=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=FakeChat(responses, delays).create))


class CloseTrackingClient(FakeClient):
    def __init__(self):
        super().__init__(['{"ok": true}'])
        self.close_count = 0

    def close(self):
        self.close_count += 1


def test_dataframe_call_expands_json_results():
    df = pd.DataFrame(
        {
            "title": ["A", "B"],
            "abstract": ["first", "second"],
        },
        index=["row-a", "row-b"],
    )
    loom = Loom(
        model="x",
        prompt="{{ title }}: {{ abstract }}",
        client=FakeClient(
            [
                '{"sentiment":"positive","score":0.9}',
                '{"sentiment":"neutral","score":0.5}',
            ]
        ),
        checkpoint=None,
    )

    out = loom(df, input=["title", "abstract"], concurrency=2)

    assert list(out.index) == ["row-a", "row-b"]
    assert list(out["title"]) == ["A", "B"]
    assert list(out["sentiment"]) == ["positive", "neutral"]
    assert list(out["score"]) == [0.9, 0.5]
    assert list(out["_loom_ok"]) == [True, True]
    assert list(out["_loom_checkpoint"]) == [False, False]


def test_records_input_returns_dataframe():
    loom = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"label":"a"}', '{"label":"b"}']),
        checkpoint=None,
    )

    out = loom([{"text": "one"}, {"text": "two"}], input="text")

    assert isinstance(out, pd.DataFrame)
    assert list(out["text"]) == ["one", "two"]
    assert list(out["label"]) == ["a", "b"]


def test_checkpoint_reuses_successful_rows(tmp_path):
    df = pd.DataFrame({"text": ["a", "b"]})
    db = tmp_path / "checkpoint.db"
    loom = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"value":"one"}', '{"value":"two"}']),
        checkpoint=db,
    )

    first = loom(df, input="text", resume="example", concurrency=2)
    second = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"value":"unused"}', '{"value":"unused"}']),
        checkpoint=db,
    )(df, input="text", resume="example", concurrency=2)

    assert list(first["value"]) == ["one", "two"]
    assert list(first["_loom_checkpoint"]) == [False, False]
    assert list(second["value"]) == ["one", "two"]
    assert list(second["_loom_checkpoint"]) == [True, True]


def test_checkpoint_payload_is_self_describing(tmp_path):
    df = pd.DataFrame({"text": ["a"]})
    db = tmp_path / "checkpoint.db"
    Loom(
        model="x",
        prompt="Prompt: {{ text }}",
        system="system",
        client=FakeClient(['{"value":"one"}']),
        checkpoint=db,
        temperature=0.2,
    )(df, input="text", resume="audit")

    with sqlite3.connect(db) as conn:
        payload = conn.execute("SELECT payload FROM silkloom_results").fetchone()[0]
    record = json.loads(payload)

    assert record["context"]["namespace"] == "audit"
    assert record["context"]["model"] == "x"
    assert record["context"]["prompt"] == "Prompt: {{ text }}"
    assert record["context"]["system"] == "system"
    assert record["context"]["params"] == {"temperature": 0.2}
    assert record["context"]["input_columns"] == ["text"]
    assert record["input"] == {"text": "a"}
    assert record["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Prompt: a"},
    ]
    assert record["result"]["value"] == {"value": "one"}
    assert record["result"]["ok"] is True


def test_dirty_json_is_repaired():
    df = pd.DataFrame({"text": ["profile"]})
    loom = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"name": "Ada", "skills": ["Python",]}']),
        checkpoint=None,
    )

    out = loom(df, input="text")

    assert out.loc[0, "name"] == "Ada"
    assert out.loc[0, "skills"] == ["Python"]


def test_output_column_conflict_requires_prefix():
    df = pd.DataFrame({"sentiment": ["source"], "text": ["hello"]})
    loom = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"sentiment":"model"}']),
        checkpoint=None,
    )

    with pytest.raises(ValueError, match="prefix"):
        loom(df, input="text")

    out = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"sentiment":"model"}']),
        checkpoint=None,
    )(df, input="text", prefix="llm_")
    assert out.loc[0, "sentiment"] == "source"
    assert out.loc[0, "llm_sentiment"] == "model"


def test_reserved_status_column_conflict_requires_prefix():
    df = pd.DataFrame({"text": ["hello"]})
    loom = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"_loom_ok":"model"}']),
        checkpoint=None,
    )

    with pytest.raises(ValueError, match="reserved"):
        loom(df, input="text")

    out = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['{"_loom_ok":"model"}']),
        checkpoint=None,
    )(df, input="text", prefix="llm_")
    assert out.loc[0, "llm__loom_ok"] == "model"


def test_status_columns_can_include_raw_output_and_reasoning():
    df = pd.DataFrame({"text": ["x"]})
    loom = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(['<think>trace</think>{"label":"ok"}']),
        checkpoint=None,
    )

    out = loom(df, input="text", include_output=True, include_reasoning=True)

    assert out.loc[0, "label"] == "ok"
    assert out.loc[0, "_loom_reasoning"] == "trace"
    assert '{"label":"ok"}' in out.loc[0, "_loom_output"]


def test_image_column_builds_multimodal_messages(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    seen_messages = []

    class ImageClient(FakeClient):
        def __init__(self):
            super().__init__(['{"scene":"ok"}'])

        def _create(self, model, messages, **kwargs):
            seen_messages.append(messages)
            return FakeResponse('{"scene":"ok"}')

    client = ImageClient()
    client.chat.completions.create = client._create
    df = pd.DataFrame({"instruction": ["describe"], "image_path": [str(image)]})

    out = Loom(model="x", prompt="{{ instruction }}", client=client, checkpoint=None)(
        df,
        input="instruction",
        images="image_path",
    )

    content = seen_messages[0][-1]["content"]
    assert out.loc[0, "scene"] == "ok"
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_checkpoint_payload_records_image_context(tmp_path):
    seen_messages = []

    class ImageClient(FakeClient):
        def __init__(self):
            super().__init__(['{"scene":"ok"}'])

        def _create(self, model, messages, **kwargs):
            seen_messages.append(messages)
            return FakeResponse('{"scene":"ok"}')

    client = ImageClient()
    client.chat.completions.create = client._create
    db = tmp_path / "checkpoint.db"
    df = pd.DataFrame({"instruction": ["describe"], "image_url": ["https://example.com/image.png"]})

    Loom(model="x", prompt="{{ instruction }}", client=client, checkpoint=db)(
        df,
        input="instruction",
        images="image_url",
        resume="vlm-audit",
    )

    with sqlite3.connect(db) as conn:
        payload = conn.execute("SELECT payload FROM silkloom_results").fetchone()[0]
    record = json.loads(payload)

    assert record["context"]["image_columns"] == ["image_url"]
    assert record["input"] == {
        "instruction": "describe",
        "images": ["https://example.com/image.png"],
    }
    assert record["messages"] == seen_messages[0]


def test_multiple_image_columns_are_combined():
    seen_messages = []

    class ImageClient(FakeClient):
        def __init__(self):
            super().__init__(['{"count":2}'])

        def _create(self, model, messages, **kwargs):
            seen_messages.append(messages)
            return FakeResponse('{"count":2}')

    client = ImageClient()
    client.chat.completions.create = client._create
    df = pd.DataFrame(
        {
            "text": ["compare"],
            "front": ["https://example.com/front.png"],
            "back": ["https://example.com/back.png"],
        }
    )

    out = Loom(model="x", prompt="{{ text }}", client=client, checkpoint=None)(
        df,
        input="text",
        images=["front", "back"],
    )

    content = seen_messages[0][-1]["content"]
    urls = [part["image_url"]["url"] for part in content[1:]]
    assert out.loc[0, "count"] == 2
    assert urls == ["https://example.com/front.png", "https://example.com/back.png"]


def test_context_manager_closes_client():
    client = CloseTrackingClient()

    with Loom(model="x", prompt="{{ text }}", client=client, checkpoint=None):
        pass

    assert client.close_count == 1
