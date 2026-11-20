import asyncio
import logging

import pytest

from app.factories.search_sync_factory import REQUIRED_ENV, SearchSyncFactory
from app.services.search_sync import SearchSync
from tests.test_search_index_service import FakeEmbedder


class Part:
    def __init__(self, log, name, fail_stop=False):
        self.log = log
        self.name = name
        self.fail_stop = fail_stop

    def start(self):
        self.log.append(f"{self.name}.start")

    def request(self):
        self.log.append(f"{self.name}.request")

    async def stop(self):
        self.log.append(f"{self.name}.stop")
        if self.fail_stop:
            raise RuntimeError("stop failed")


def build(log, ticks, loaded=True, fail_listener_stop=False):
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > ticks:
            await asyncio.Event().wait()  # park until stopped

    async def close():
        log.append("gaia.close")

    sync = SearchSync(Part(log, "queue"), Part(log, "reconciler"),
                      Part(log, "listener", fail_stop=fail_listener_stop), interval_seconds=60,
                      model_loaded=lambda: loaded, close=close, sleep=sleep)
    return sync, sleeps


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def done_future(error=None):
    future = asyncio.get_running_loop().create_future()
    if error:
        future.set_exception(error)
    else:
        future.set_result(None)
    return future


def test_starts_once_the_model_is_loaded_reconciles_at_once_and_on_a_timer():
    async def scenario():
        log = []
        sync, sleeps = build(log, ticks=2)
        sync.start(done_future())
        await settle()
        await sync.stop()
        return log, sleeps

    log, sleeps = asyncio.run(scenario())
    assert log == ["queue.start", "listener.start", "reconciler.request", "reconciler.request",
                   "reconciler.request", "listener.stop", "reconciler.stop", "queue.stop", "gaia.close"]
    assert sleeps[:2] == [60, 60]


def test_nothing_starts_before_the_model_load_finishes():
    async def scenario():
        log = []
        sync, _ = build(log, ticks=0)
        load = asyncio.get_running_loop().create_future()
        sync.start(load)
        await settle()
        before = list(log)
        await sync.stop()
        return before, load

    before, load = asyncio.run(scenario())
    assert before == []
    assert not load.cancelled()


@pytest.mark.parametrize("error, loaded", [(None, False), (RuntimeError("load raised"), False)])
def test_a_model_that_did_not_load_leaves_indexing_off(caplog, error, loaded):
    async def scenario():
        log = []
        sync, _ = build(log, ticks=0, loaded=loaded)
        sync.start(done_future(error))
        await settle()
        await sync.stop()
        return log

    with caplog.at_level(logging.ERROR):
        log = asyncio.run(scenario())
    assert "queue.start" not in log and "listener.start" not in log
    assert "indexing is off" in caplog.text


def test_stop_goes_on_when_a_part_fails_to_stop(caplog):
    async def scenario():
        log = []
        sync, _ = build(log, ticks=0, fail_listener_stop=True)
        sync.start(done_future())
        await settle()
        await sync.stop()
        return log

    with caplog.at_level(logging.ERROR):
        log = asyncio.run(scenario())
    assert log[-4:] == ["listener.stop", "reconciler.stop", "queue.stop", "gaia.close"]
    assert "failed: RuntimeError" in caplog.text


@pytest.mark.parametrize("missing", REQUIRED_ENV)
def test_factory_leaves_indexing_off_without_its_settings(monkeypatch, missing):
    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "https://x.test" if name.endswith("URL") else "value")
    monkeypatch.delenv(missing)

    assert SearchSyncFactory.create_search_sync(FakeEmbedder()) is None


def test_factory_leaves_indexing_off_without_a_model(monkeypatch):
    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "value")

    assert SearchSyncFactory.create_search_sync(None) is None


def test_factory_builds_indexing_when_configured(monkeypatch):
    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "https://x.test" if name.endswith("URL") else "value")

    async def scenario():
        sync = SearchSyncFactory.create_search_sync(FakeEmbedder())
        assert isinstance(sync, SearchSync)
        await sync.stop()

    asyncio.run(scenario())
