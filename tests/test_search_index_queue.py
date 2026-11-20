import asyncio
import logging

from app.services.search_index_queue import SearchIndexQueue


class Recorder:
    """A processor that records each run; runs of a gated key wait until the test opens its gate."""

    def __init__(self):
        self.runs = []
        self.active = set()
        self.overlapped = False
        self.gates = {}
        self.started = {}
        self.fail = set()

    def gate(self, key):
        self.gates[key] = asyncio.Event()
        self.started[key] = asyncio.Event()

    async def __call__(self, resource, record_id):
        key = (resource, record_id)
        if key in self.active:
            self.overlapped = True
        self.active.add(key)
        self.runs.append(key)
        try:
            if key in self.gates:
                self.started[key].set()
                await self.gates.pop(key).wait()
            if key in self.fail:
                raise RuntimeError("boom")
            return "stored"
        finally:
            self.active.discard(key)


def run(coro):
    return asyncio.run(coro)


def test_each_offered_record_is_processed():
    async def scenario():
        recorder = Recorder()
        queue = SearchIndexQueue(recorder, max_size=10, concurrency=2)
        queue.start()
        queue.offer("issue", "1")
        queue.offer("issue", "2")
        await queue.join()
        await queue.stop()
        return recorder

    assert sorted(run(scenario()).runs) == [("issue", "1"), ("issue", "2")]


def test_a_record_already_queued_is_not_queued_again():
    async def scenario():
        recorder = Recorder()
        queue = SearchIndexQueue(recorder, max_size=10, concurrency=1)
        queue.offer("issue", "1")
        queue.offer("issue", "1")
        await queue.put("issue", "1")
        queue.start()
        await queue.join()
        await queue.stop()
        return recorder

    assert run(scenario()).runs == [("issue", "1")]


def test_a_record_changed_while_processed_runs_again_after_never_in_parallel():
    async def scenario():
        recorder = Recorder()
        key = ("issue", "1")
        recorder.gate(key)
        gate = recorder.gates[key]
        queue = SearchIndexQueue(recorder, max_size=10, concurrency=3)
        queue.start()
        queue.offer(*key)
        await recorder.started[key].wait()
        # Two edits during the run ask for one more run, which waits for this one.
        queue.offer(*key)
        queue.offer(*key)
        await asyncio.sleep(0)
        assert recorder.runs == [key]
        gate.set()
        await queue.join()
        await queue.stop()
        return recorder

    recorder = run(scenario())
    assert recorder.runs == [("issue", "1"), ("issue", "1")]
    assert not recorder.overlapped


def test_a_full_queue_drops_an_offer_and_logs_it(caplog):
    async def scenario():
        queue = SearchIndexQueue(Recorder(), max_size=1, concurrency=1)
        first = queue.offer("issue", "1")
        second = queue.offer("issue", "2")
        return first, second

    with caplog.at_level(logging.WARNING):
        assert run(scenario()) == (True, False)
    assert "queue is full" in caplog.text


def test_put_waits_for_room_instead_of_dropping():
    async def scenario():
        recorder = Recorder()
        queue = SearchIndexQueue(recorder, max_size=1, concurrency=1)
        await queue.put("issue", "1")
        waiting = asyncio.create_task(queue.put("issue", "2"))
        await asyncio.sleep(0)
        assert not waiting.done()
        queue.start()
        await waiting
        await queue.join()
        await queue.stop()
        return recorder

    assert run(scenario()).runs == [("issue", "1"), ("issue", "2")]


def test_a_failure_is_logged_and_the_queue_goes_on(caplog):
    async def scenario():
        recorder = Recorder()
        recorder.fail.add(("issue", "1"))
        queue = SearchIndexQueue(recorder, max_size=10, concurrency=1)
        queue.start()
        queue.offer("issue", "1")
        queue.offer("issue", "2")
        await queue.join()
        await queue.stop()
        return recorder

    with caplog.at_level(logging.ERROR):
        recorder = run(scenario())
    assert recorder.runs == [("issue", "1"), ("issue", "2")]
    assert "Indexing issue 1 failed: RuntimeError" in caplog.text


def test_a_failure_is_handed_to_on_failure():
    async def scenario():
        recorder = Recorder()
        recorder.fail.add(("issue", "1"))
        failures = []
        queue = SearchIndexQueue(recorder, max_size=10, concurrency=1, on_failure=failures.append)
        queue.start()
        queue.offer("issue", "1")
        await queue.join()
        await queue.stop()
        return failures

    failures = run(scenario())
    assert len(failures) == 1 and isinstance(failures[0], RuntimeError)


def test_stopping_during_a_run_queues_no_rerun():
    async def scenario():
        recorder = Recorder()
        key = ("issue", "1")
        recorder.gate(key)
        queue = SearchIndexQueue(recorder, max_size=10, concurrency=1)
        queue.start()
        queue.offer(*key)
        await recorder.started[key].wait()
        queue.offer(*key)
        await queue.stop()
        return queue

    queue = run(scenario())
    assert queue._queue.qsize() == 0


def test_a_reconcile_listing_a_running_record_queues_no_rerun():
    async def scenario():
        recorder = Recorder()
        key = ("issue", "1")
        recorder.gate(key)
        gate = recorder.gates[key]
        queue = SearchIndexQueue(recorder, max_size=10, concurrency=1)
        queue.start()
        queue.offer(*key)
        await recorder.started[key].wait()
        # Still pending in Gaia until this run stores it: nothing new to do.
        await queue.put(*key)
        gate.set()
        await queue.join()
        await queue.stop()
        return recorder

    assert run(scenario()).runs == [("issue", "1")]
