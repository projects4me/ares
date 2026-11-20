import asyncio
import logging

from app.adapters.gaia_client import PendingPage, PendingRecord
from app.adapters.gaia_errors import GaiaUnavailable
from app.services.search_index_queue import SearchIndexQueue
from app.services.search_reconciler import SearchReconciler


class FakeGaia:
    """Serves the pending list from canned pages keyed by cursor, recording each call."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []
        self.gate = None

    async def list_pending(self, after=None, limit=None):
        self.calls.append((after, limit))
        if self.gate:
            await self.gate.wait()
        page = self.pages[after]
        if isinstance(page, Exception):
            raise page
        return page


class Sink:
    def __init__(self):
        self.records = []

    async def put(self, resource, record_id):
        self.records.append((resource, record_id))


def page(ids, next_cursor=None):
    return PendingPage([PendingRecord("issue", "issue", i) for i in ids], next_cursor)


def run(coro):
    return asyncio.run(coro)


def test_reconcile_pages_the_pending_list_into_the_sink():
    async def scenario():
        gaia = FakeGaia({None: page(["1", "2"], "issue:2"), "issue:2": page(["3"])})
        sink = Sink()
        reconciler = SearchReconciler(gaia, sink, page_size=2)
        reconciler.request()
        await reconciler.wait()
        return gaia, sink

    gaia, sink = run(scenario())
    assert gaia.calls == [(None, 2), ("issue:2", 2)]
    assert sink.records == [("issue", "1"), ("issue", "2"), ("issue", "3")]


def test_reconcile_queues_each_record_once_while_it_waits():
    async def scenario():
        processed = []

        async def process(resource, record_id):
            processed.append((resource, record_id))

        # The same record listed twice (and announced by an event) before a worker takes it.
        gaia = FakeGaia({None: page(["1", "2"], "issue:2"), "issue:2": page(["1"])})
        queue = SearchIndexQueue(process, max_size=10, concurrency=1)
        queue.offer("issue", "1")
        reconciler = SearchReconciler(gaia, queue)
        reconciler.request()
        await reconciler.wait()
        queue.start()
        await queue.join()
        await queue.stop()
        return processed

    assert run(scenario()) == [("issue", "1"), ("issue", "2")]


def test_requests_during_a_run_ask_for_exactly_one_more_run():
    async def scenario():
        gaia = FakeGaia({None: page(["1"])})
        gaia.gate = asyncio.Event()
        reconciler = SearchReconciler(gaia, Sink())
        reconciler.request()
        await asyncio.sleep(0)
        reconciler.request()
        reconciler.request()
        gaia.gate.set()
        await reconciler.wait()
        return gaia

    assert len(run(scenario()).calls) == 2


def test_a_request_after_a_run_starts_a_new_one():
    async def scenario():
        gaia = FakeGaia({None: page(["1"])})
        reconciler = SearchReconciler(gaia, Sink())
        reconciler.request()
        await reconciler.wait()
        reconciler.request()
        await reconciler.wait()
        return gaia

    assert len(run(scenario()).calls) == 2


def test_a_gaia_failure_stops_the_run_and_is_logged(caplog):
    async def scenario():
        gaia = FakeGaia({None: page(["1"], "issue:1"), "issue:1": GaiaUnavailable("Gaia is down")})
        sink = Sink()
        reconciler = SearchReconciler(gaia, sink)
        reconciler.request()
        await reconciler.wait()
        return sink

    with caplog.at_level(logging.ERROR):
        sink = run(scenario())
    assert sink.records == [("issue", "1")]
    assert "Reconcile stopped after 1 pending records: GaiaUnavailable" in caplog.text


def test_a_cursor_that_does_not_move_stops_the_run(caplog):
    async def scenario():
        gaia = FakeGaia({None: page(["1"], "issue:1"), "issue:1": page(["2"], "issue:1")})
        reconciler = SearchReconciler(gaia, Sink())
        reconciler.request()
        await reconciler.wait()
        return gaia

    with caplog.at_level(logging.ERROR):
        gaia = run(scenario())
    assert len(gaia.calls) == 2
    assert "same cursor twice" in caplog.text


def test_request_soon_runs_once_after_the_delay_however_often_asked():
    async def scenario():
        gaia = FakeGaia({None: page(["1"])})
        reconciler = SearchReconciler(gaia, Sink(), retry_delay=0.01)
        reconciler.request_soon()
        reconciler.request_soon()
        await asyncio.sleep(0.05)
        await reconciler.wait()
        await reconciler.stop()
        return gaia

    assert len(run(scenario()).calls) == 1


def test_stop_cancels_a_pending_request_soon():
    async def scenario():
        gaia = FakeGaia({None: page(["1"])})
        reconciler = SearchReconciler(gaia, Sink(), retry_delay=0.01)
        reconciler.request_soon()
        await reconciler.stop()
        await asyncio.sleep(0.05)
        return gaia

    assert run(scenario()).calls == []
