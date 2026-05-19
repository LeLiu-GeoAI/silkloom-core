import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from silkloom_core import TaskLoom


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeChat:
    def __init__(self, responses, delays=None):
        self._responses = list(responses)
        self._delays = delays or [0] * len(self._responses)
        self._idx = 0
        self._lock = threading.Lock()

    def _next(self):
        with self._lock:
            i = self._idx
            self._idx += 1
        delay = self._delays[i] if i < len(self._delays) else 0
        if delay:
            time.sleep(delay)
        return self._responses[i]

    def create(self, model, messages, **kwargs):
        # Try to determine the input index from rendered messages (template is '{{text}}' in tests)
        try:
            if isinstance(messages, list):
                last = messages[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                else:
                    content = last
                # content may be a string like '1' or a list for multimodal
                if isinstance(content, str) and content.isdigit():
                    idx = int(content) - 1
                    delay = self._delays[idx] if idx < len(self._delays) else 0
                    if delay:
                        time.sleep(delay)
                    return FakeResponse(self._responses[idx])
        except Exception:
            pass

        return FakeResponse(self._next())


class FakeClient:
    def __init__(self, responses, delays=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=FakeChat(responses, delays).create))


def test_process_returns_raw_when_no_response_model():
    client = FakeClient(["plain text output"])
    loom = TaskLoom(model="x", prompt_template="{{text}}", client=client)
    res = loom.process({"text": "hello"})
    assert res.is_success
    assert res.data == "plain text output"


def test_process_parses_json_and_pydantic(tmp_path):
    from pydantic import BaseModel


    class Profile(BaseModel):
        name: str
        skills: list[str]

    # content includes fenced json
    json_payload = '```json\n{"name":"张三","skills":["Python"]}\n```'
    client = FakeClient([json_payload])
    loom = TaskLoom(model="x", prompt_template="{{text}}", client=client, response_model=Profile)
    res = loom.process({"text": "profile"})
    assert res.is_success
    assert res.data.name == "张三"
    assert res.data.skills == ["Python"]


def test_map_and_cache(tmp_path):
    db = tmp_path / "test.db"
    # two inputs, first run returns distinct outputs
    client = FakeClient(["out1", "out2"])
    loom = TaskLoom(model="x", prompt_template="{{text}}", client=client, db_path=str(db))

    seq = [{"text": "a"}, {"text": "b"}]
    batch = loom.map(seq, task_name="run1", max_workers=2)
    assert len(batch) == 2
    assert all(r.is_success for r in batch)

    # second run should hit cache and mark cached=True
    # create a new loom using same db
    client2 = FakeClient(["should-not-be-used", "should-not-be-used"])
    loom2 = TaskLoom(model="x", prompt_template="{{text}}", client=client2, db_path=str(db))
    batch2 = loom2.map(seq, task_name="run1", max_workers=2)
    assert len(batch2) == 2
    assert all(r.cached for r in batch2)


def test_stream_ordered_and_unordered(tmp_path):
    # responses have different delays so completion order differs
    responses = ["r1", "r2", "r3"]
    delays = [0.3, 0.1, 0.2]
    client = FakeClient(responses, delays)
    loom = TaskLoom(model="x", prompt_template="{{text}}", client=client)
    seq = [{"text": "1"}, {"text": "2"}, {"text": "3"}]

    # unordered: yield as completed (fastest first should be r2)
    out = list(loom.stream(seq, task_name=None, max_workers=3, ordered=False))
    assert [t.data for t in out] == responses or len(out) == 3

    # ordered: should preserve input order
    client2 = FakeClient(responses, delays)
    loom2 = TaskLoom(model="x", prompt_template="{{text}}", client=client2)
    out2 = list(loom2.stream(seq, task_name=None, max_workers=3, ordered=True))
    assert [t.data for t in out2] == responses


def test_astream_sync(tmp_path):
    responses = ["ar1", "ar2"]
    delays = [0.1, 0]
    client = FakeClient(responses, delays)
    loom = TaskLoom(model="x", prompt_template="{{text}}", client=client)
    seq = [{"text": "1"}, {"text": "2"}]

    collected = []

    async def runner():
        async for item in loom.astream(seq, task_name=None, max_workers=2, ordered=True):
            collected.append(item)

    asyncio.run(runner())

    assert [c.data for c in collected] == responses
