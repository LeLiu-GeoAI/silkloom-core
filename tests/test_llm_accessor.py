from __future__ import annotations

import sqlite3
import threading
from types import SimpleNamespace

import pandas as pd

import silkloom_core


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeChat:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict] = []
        self.index = 0
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            index = self.index
            self.index += 1
        return FakeResponse(self.responses[index])


class FakeClient:
    def __init__(self, responses: list[str]):
        self.chat_impl = FakeChat(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_impl.create))


def test_accessor_extracts_json_to_result_frame(tmp_path):
    client = FakeClient(['{"sentiment":"positive","score":0.9}', '{"sentiment":"neutral","score":0.5}'])
    df = pd.DataFrame(
        {"title": ["A", "B"], "abstract": ["first", "second"]},
        index=["row-a", "row-b"],
    )

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{title}: {abstract}",
        model="test-model",
        max_workers=2,
        verbose=False,
    )

    assert list(out.index) == ["row-a", "row-b"]
    assert list(out["sentiment"]) == ["positive", "neutral"]
    assert list(out["score"]) == [0.9, 0.5]
    assert client.chat_impl.calls[0]["model"] == "test-model"


def test_cache_reuses_successful_response(tmp_path):
    db = tmp_path / "cache.db"
    df = pd.DataFrame({"text": ["same"]})

    first = FakeClient(['{"value":"cached"}'])
    out1 = df.llm.setup(client=first, cache_path=db).extract("{text}", verbose=False)

    second = FakeClient(['{"value":"unused"}'])
    out2 = df.llm.setup(client=second, cache_path=db).extract("{text}", verbose=False)

    assert out1.loc[0, "value"] == "cached"
    assert out2.loc[0, "value"] == "cached"
    assert len(first.chat_impl.calls) == 1
    assert second.chat_impl.calls == []

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM cache").fetchone()[0] == 1


def test_dirty_json_is_repaired(tmp_path):
    client = FakeClient(['{"name": "Ada", "skills": ["Python",]}'])
    df = pd.DataFrame({"text": ["profile"]})

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract("{text}", verbose=False)

    assert out.loc[0, "name"] == "Ada"
    assert out.loc[0, "skills"] == ["Python"]


def test_non_object_json_returns_raw_column(tmp_path):
    client = FakeClient(["not json"])
    df = pd.DataFrame({"text": ["profile"]})

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract("{text}", verbose=False)

    assert out.loc[0, "_llm_raw"] == "not json"


def test_image_column_builds_multimodal_message(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    client = FakeClient(['{"scene":"ok"}'])
    df = pd.DataFrame({"instruction": ["describe"], "image_path": [str(image)]})

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{instruction}",
        image_column="image_path",
        verbose=False,
    )

    content = client.chat_impl.calls[0]["messages"][-1]["content"]
    assert out.loc[0, "scene"] == "ok"
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_progress_callback_reports_completed_rows(tmp_path):
    client = FakeClient(['{"value":"one"}', '{"value":"two"}'])
    df = pd.DataFrame({"text": ["one", "two"]})
    calls = []

    df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{text}",
        max_workers=2,
        progress_callback=lambda done, total: calls.append((done, total)),
        verbose=False,
    )

    assert calls == [(1, 2), (2, 2)]
    assert silkloom_core.__version__ == "6.0.0"
