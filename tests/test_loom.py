import asyncio
import threading
import time
from types import SimpleNamespace

from pydantic import BaseModel

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
        super().__init__(["ok"])
        self.close_count = 0

    def close(self):
        self.close_count += 1


def test_run_returns_raw_text():
    loom = Loom(model="x", prompt="{{ text }}", client=FakeClient(["plain"]))

    result = loom.run("hello")

    assert result.ok
    assert result.value == "plain"
    assert result.input == {"text": "hello"}


def test_context_manager_closes_client():
    client = CloseTrackingClient()

    with Loom(model="x", prompt="{{ text }}", client=client):
        pass

    assert client.close_count == 1


def test_run_parses_json_and_pydantic():
    class Profile(BaseModel):
        name: str
        skills: list[str]

    loom = Loom(
        model="x",
        prompt="{{ text }}",
        output=Profile,
        client=FakeClient(['```json\n{"name":"Ada","skills":["Python"]}\n```']),
    )

    result = loom.run({"text": "profile"})

    assert result.ok
    assert result.value.name == "Ada"
    assert result.value.skills == ["Python"]


def test_batch_uses_cache(tmp_path):
    db = tmp_path / "cache.db"
    seq = [{"text": "a"}, {"text": "b"}]
    loom = Loom(model="x", prompt="{{ text }}", client=FakeClient(["out1", "out2"]), cache=db)

    first = loom.batch(seq, name="example", concurrency=2)
    second = Loom(
        model="x",
        prompt="{{ text }}",
        client=FakeClient(["unused", "unused"]),
        cache=db,
    ).batch(seq, name="example", concurrency=2)

    assert [item.value for item in first] == ["out1", "out2"]
    assert [item.value for item in second] == ["out1", "out2"]
    assert all(item.cache_hit for item in second)


def test_each_ordered_and_unordered():
    responses = ["r1", "r2", "r3"]
    delays = [0.3, 0.1, 0.2]
    seq = [{"text": "1"}, {"text": "2"}, {"text": "3"}]

    unordered = list(
        Loom(model="x", prompt="{{ text }}", client=FakeClient(responses, delays), cache=None).each(
            seq,
            concurrency=3,
            ordered=False,
        )
    )
    ordered = list(
        Loom(model="x", prompt="{{ text }}", client=FakeClient(responses, delays), cache=None).each(
            seq,
            concurrency=3,
            ordered=True,
        )
    )

    assert sorted(item.value for item in unordered) == responses
    assert [item.value for item in ordered] == responses


def test_aeach_ordered_with_sync_compatible_client():
    async def runner():
        loom = Loom(
            model="x",
            prompt="{{ text }}",
            client=FakeClient(["ar1", "ar2"], delays=[0.1, 0]),
            cache=None,
        )
        return [item async for item in loom.aeach([{"text": "1"}, {"text": "2"}], concurrency=2, ordered=True)]

    collected = asyncio.run(runner())

    assert [item.value for item in collected] == ["ar1", "ar2"]
