"""
The indexing queue: one bounded async queue of records to index, worked by a few workers.

Events and reconciled records share it. A record is a (resource, id) key:
- one already waiting in the queue is not queued again;
- one being processed is marked to run once more when its run ends, never handed to a
  second worker in parallel, so an edit made while it was embedded is not lost;
- a failure is logged and handed to on_failure (which may ask for a reconcile); no retry here.

offer() never waits and drops the record when the queue is full (an event; the next
reconcile finds it). put() waits for room (a reconcile, which pages at the queue's pace).
"""

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

RecordKey = Tuple[str, str]
Processor = Callable[[str, str], Awaitable[object]]
FailureHandler = Callable[[Exception], None]


class SearchIndexQueue:
    """Owns which records wait to be indexed and the workers that index them, one run per record at a time."""

    def __init__(self, process: Processor, max_size: int, concurrency: int,
                 on_failure: Optional[FailureHandler] = None):
        self._process = process
        self._on_failure = on_failure
        self._queue: "asyncio.Queue[RecordKey]" = asyncio.Queue(max_size)
        self._concurrency = concurrency
        self._waiting: Set[RecordKey] = set()
        self._running: Set[RecordKey] = set()
        self._rerun: Set[RecordKey] = set()
        self._workers: List[asyncio.Task] = []

    def start(self) -> None:
        if not self._workers:
            self._workers = [asyncio.create_task(self._work(), name=f"search-index-worker-{n}")
                             for n in range(self._concurrency)]

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    def offer(self, resource: str, record_id: str) -> bool:
        """Queue the record unless it is already waiting; False when the queue is full and it was dropped."""
        key = (resource, record_id)
        if self._absorbed(key):
            return True
        try:
            self._queue.put_nowait(key)
        except asyncio.QueueFull:
            logger.warning("The search index queue is full; %s %s waits for the next reconcile", resource, record_id)
            return False
        self._waiting.add(key)
        return True

    async def put(self, resource: str, record_id: str) -> None:
        """Queue the record unless it is already waiting, waiting for room when the queue is full."""
        key = (resource, record_id)
        if self._absorbed(key):
            return
        # Marked before waiting, so a concurrent offer of the same record does not queue it twice.
        self._waiting.add(key)
        try:
            await self._queue.put(key)
        except BaseException:
            self._waiting.discard(key)
            raise

    async def join(self) -> None:
        """Wait until every queued record has been processed."""
        await self._queue.join()

    def _absorbed(self, key: RecordKey) -> bool:
        """True when the key needs no new queue entry: it is waiting already, or running and now marked to rerun."""
        if key in self._waiting:
            return True
        if key in self._running:
            self._rerun.add(key)
            return True
        return False

    async def _work(self) -> None:
        while True:
            key = await self._queue.get()
            self._waiting.discard(key)
            self._running.add(key)
            try:
                await self._run(key)
                rerun = key in self._rerun
                self._release(key)
                if rerun:
                    # Changed while it ran: once more. A full queue drops it (logged) for the next reconcile.
                    self.offer(*key)
            finally:
                # Also on cancellation, which queues no rerun.
                self._release(key)
                self._queue.task_done()

    def _release(self, key: RecordKey) -> None:
        self._running.discard(key)
        self._rerun.discard(key)

    async def _run(self, key: RecordKey) -> Optional[object]:
        resource, record_id = key
        try:
            outcome = await self._process(resource, record_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Left for the next reconcile; the message holds no token or text.
            logger.error("Indexing %s %s failed: %s: %s", resource, record_id, type(e).__name__, e)
            if self._on_failure:
                self._on_failure(e)
            return None
        logger.info("Indexed %s %s: %s", resource, record_id, getattr(outcome, "value", outcome))
        return outcome
