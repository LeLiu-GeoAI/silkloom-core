from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import silkloom_core


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #

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


class FailingChat:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise RuntimeError("network down")


class FailingClient:
    def __init__(self):
        self.chat_impl = FailingChat()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_impl.create))


def _save_globals():
    import silkloom_core.taskloom as tl
    return tl._DEFAULT_CLIENT, tl._DEFAULT_CACHE_PATH


def _restore_globals(saved):
    import silkloom_core.taskloom as tl
    tl._DEFAULT_CLIENT, tl._DEFAULT_CACHE_PATH = saved


# --------------------------------------------------------------------------- #
# Core extraction
# --------------------------------------------------------------------------- #

def test_accessor_extracts_json_to_result_frame(tmp_path):
    client = FakeClient(['{"sentiment":"positive","score":0.9}', '{"sentiment":"neutral","score":0.5}'])
    df = pd.DataFrame(
        {"title": ["A", "B"], "abstract": ["first", "second"]},
        index=["row-a", "row-b"],
    )

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{{ title }}: {{ abstract }}",
        model="test-model",
        max_workers=1,
        verbose=False,
    )

    assert list(out.index) == ["row-a", "row-b"]
    assert list(out["sentiment"]) == ["positive", "neutral"]
    assert list(out["score"]) == [0.9, 0.5]
    assert client.chat_impl.calls[0]["model"] == "test-model"


def test_empty_dataframe_returns_empty_frame_with_same_index(tmp_path):
    client = FakeClient([])
    df = pd.DataFrame({"text": []}, index=[])

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{{ text }}", verbose=False
    )

    assert out.empty
    assert list(out.index) == []


# --------------------------------------------------------------------------- #
# Cache behavior
# --------------------------------------------------------------------------- #

def test_cache_reuses_successful_response(tmp_path):
    db = tmp_path / "cache.db"
    df = pd.DataFrame({"text": ["same"]})

    first = FakeClient(['{"value":"cached"}'])
    out1 = df.llm.setup(client=first, cache_path=db).extract("{{ text }}", verbose=False)

    second = FakeClient(['{"value":"unused"}'])
    out2 = df.llm.setup(client=second, cache_path=db).extract("{{ text }}", verbose=False)

    assert out1.loc[0, "value"] == "cached"
    assert out2.loc[0, "value"] == "cached"
    assert len(first.chat_impl.calls) == 1
    assert second.chat_impl.calls == []

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM cache").fetchone()[0] == 1


def test_cache_records_request_and_parsed_result(tmp_path):
    db = tmp_path / "cache.db"
    client = FakeClient(['{"value":"saved"}'])
    df = pd.DataFrame({"text": ["hello"]})

    df.llm.setup(client=client, cache_path=db).extract(
        "Analyze {{ text }}",
        model="audit-model",
        temperature=0.2,
        verbose=False,
    )

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """
            SELECT ok, model, messages_json, params_json, request_json, response, parsed_json, error, attempts
            FROM cache
            """
        ).fetchone()

    messages = json.loads(row[2])
    params = json.loads(row[3])
    request = json.loads(row[4])
    parsed = json.loads(row[6])

    assert row[0] == 1
    assert row[1] == "audit-model"
    assert messages[-1]["content"] == "Analyze hello"
    assert params == {"temperature": 0.2}
    assert request["model"] == "audit-model"
    assert row[5] == '{"value":"saved"}'
    assert parsed == {"value": "saved"}
    assert row[7] is None
    assert row[8] == 1


def test_cache_records_failed_request_without_cache_hit(tmp_path):
    db = tmp_path / "cache.db"
    client = FailingClient()
    df = pd.DataFrame({"text": ["hello"]})

    out = df.llm.setup(client=client, cache_path=db).extract("{{ text }}", max_retries=0, verbose=False)
    retry = df.llm.setup(client=client, cache_path=db).extract("{{ text }}", max_retries=0, verbose=False)

    assert "network down" in out.loc[0, "_llm_error"]
    assert "network down" in retry.loc[0, "_llm_error"]
    assert client.chat_impl.calls == 2

    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT ok, response, error, attempts FROM cache").fetchone()

    assert row[0] == 0
    assert row[1] == ""
    assert "network down" in row[2]
    assert row[3] == 1


def test_setup_does_not_create_database_file(tmp_path):
    """setup() should be lazy — no SQLite file until extract() is called."""
    cache_path = tmp_path / "lazy.db"
    client = FakeClient(['{"value":"ok"}'])
    df = pd.DataFrame({"text": ["hello"]})

    df.llm.setup(client=client, cache_path=cache_path)

    assert not cache_path.exists(), "setup() should not create the database file"

    df.llm.extract("{{ text }}", verbose=False)

    assert cache_path.exists(), "extract() should create the database file"


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #

def test_dirty_json_is_repaired(tmp_path):
    client = FakeClient(['{"name": "Ada", "skills": ["Python",]}'])
    df = pd.DataFrame({"text": ["profile"]})

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract("{{ text }}", verbose=False)

    assert out.loc[0, "name"] == "Ada"
    assert out.loc[0, "skills"] == ["Python"]


def test_non_object_json_returns_raw_column(tmp_path):
    client = FakeClient(["not json"])
    df = pd.DataFrame({"text": ["profile"]})

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract("{{ text }}", verbose=False)

    assert out.loc[0, "_llm_raw"] == "not json"


def test_parse_error_includes_raw_content(tmp_path):
    """When json_repair raises, _llm_error and _llm_raw should both be present."""
    import silkloom_core.taskloom as tl
    from unittest.mock import patch

    accessor = pd.DataFrame({"text": ["x"]}).llm
    with patch.object(tl.json_repair, "loads", side_effect=ValueError("bad json")):
        result = accessor._parse_content("some content")
    assert "_llm_error" in result
    assert "_llm_raw" in result
    assert "bad json" in result["_llm_error"]
    assert result["_llm_raw"] == "some content"


def test_prompt_can_contain_literal_json_braces(tmp_path):
    client = FakeClient(['{"label":"ok"}'])
    df = pd.DataFrame({"text": ["hello"]})

    df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        'Read {{ text }} and return JSON like {"label": "short"}',
        verbose=False,
    )

    prompt = client.chat_impl.calls[0]["messages"][-1]["content"]
    assert prompt == 'Read hello and return JSON like {"label": "short"}'


# --------------------------------------------------------------------------- #
# Multimodal
# --------------------------------------------------------------------------- #

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
        "{{ instruction }}",
        image_column="image_path",
        verbose=False,
    )

    content = client.chat_impl.calls[0]["messages"][-1]["content"]
    assert out.loc[0, "scene"] == "ok"
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


# --------------------------------------------------------------------------- #
# Progress and cancel
# --------------------------------------------------------------------------- #

def test_progress_callback_reports_completed_rows(tmp_path):
    client = FakeClient(['{"value":"one"}', '{"value":"two"}'])
    df = pd.DataFrame({"text": ["one", "two"]})
    calls = []

    df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{{ text }}",
        max_workers=2,
        progress_callback=lambda done, total: calls.append((done, total)),
        verbose=False,
    )

    assert calls == [(1, 2), (2, 2)]
    assert silkloom_core.__version__ == "7.0.0"


# --------------------------------------------------------------------------- #
# Client configuration layers
# --------------------------------------------------------------------------- #

def test_configure_provides_default_client(tmp_path):
    client = FakeClient(['{"value":"global"}'])
    saved = _save_globals()
    try:
        silkloom_core.configure(client=client, cache_path=tmp_path / "cache.db")
        df = pd.DataFrame({"text": ["hello"]})
        out = df.llm.extract("{{ text }}", verbose=False)
        assert out.loc[0, "value"] == "global"
    finally:
        _restore_globals(saved)


def test_configure_raises_without_client_or_api_key():
    with pytest.raises(ValueError, match="requires either client"):
        silkloom_core.configure()


def test_extract_raises_without_client():
    saved = _save_globals()
    import silkloom_core.taskloom as tl
    tl._DEFAULT_CLIENT = None
    try:
        df = pd.DataFrame({"text": ["hello"]})
        with pytest.raises(RuntimeError, match="No LLM client configured"):
            df.llm.extract("{{ text }}", verbose=False)
    finally:
        _restore_globals(saved)


def test_setup_overrides_configure(tmp_path):
    global_client = FakeClient(['{"value":"global"}'])
    local_client = FakeClient(['{"value":"local"}'])
    saved = _save_globals()
    try:
        silkloom_core.configure(client=global_client, cache_path=tmp_path / "global.db")
        df = pd.DataFrame({"text": ["hello"]})
        out = df.llm.setup(client=local_client, cache_path=tmp_path / "local.db").extract(
            "{{ text }}", verbose=False
        )
        assert out.loc[0, "value"] == "local"
        assert len(global_client.chat_impl.calls) == 0
        assert len(local_client.chat_impl.calls) == 1
    finally:
        _restore_globals(saved)


def test_extract_client_overrides_configure(tmp_path):
    global_client = FakeClient(['{"value":"global"}'])
    call_client = FakeClient(['{"value":"per-call"}'])
    saved = _save_globals()
    try:
        silkloom_core.configure(client=global_client, cache_path=tmp_path / "global.db")
        df = pd.DataFrame({"text": ["hello"]})
        out = df.llm.extract("{{ text }}", client=call_client, verbose=False)
        assert out.loc[0, "value"] == "per-call"
        assert len(global_client.chat_impl.calls) == 0
        assert len(call_client.chat_impl.calls) == 1
    finally:
        _restore_globals(saved)


def test_extract_client_overrides_setup(tmp_path):
    setup_client = FakeClient(['{"value":"setup"}'])
    call_client = FakeClient(['{"value":"per-call"}'])
    df = pd.DataFrame({"text": ["hello"]})

    out = df.llm.setup(client=setup_client, cache_path=tmp_path / "cache.db").extract(
        "{{ text }}", client=call_client, verbose=False
    )
    assert out.loc[0, "value"] == "per-call"
    assert len(setup_client.chat_impl.calls) == 0
    assert len(call_client.chat_impl.calls) == 1


# --------------------------------------------------------------------------- #
# Join behavior
# --------------------------------------------------------------------------- #

def test_join_true_includes_original_columns(tmp_path):
    client = FakeClient(['{"label":"positive"}'])
    df = pd.DataFrame({"title": ["A"], "body": ["text"]})

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{{ title }}", verbose=False
    )

    assert "title" in out.columns       # original
    assert "body" in out.columns        # original
    assert "label" in out.columns       # extracted


def test_join_false_returns_only_extracted_columns(tmp_path):
    client = FakeClient(['{"label":"positive"}'])
    df = pd.DataFrame({"title": ["A"], "body": ["text"]})

    out = df.llm.setup(client=client, cache_path=tmp_path / "cache.db").extract(
        "{{ title }}", join=False, verbose=False
    )

    assert "title" not in out.columns
    assert "body" not in out.columns
    assert "label" in out.columns
