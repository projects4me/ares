"""
Reconcile: catch up on the records whose embedding is missing or outdated.

Hermes keeps no history, so events missed while Ares was down or disconnected are
recovered by paging Gaia's pending list (GET /api/v1/searchpending, Gaia search D-009)
into the indexing queue. One reconcile runs at a time; a request during a run asks for
exactly one more run after it, so a reconnect during a long run is not lost.
request_soon() asks for a run after a short delay, at most one pending at a time: after a
transient failure, so a Gaia blip is caught up on in a minute rather than at the next timer.
"""

import asyncio
import logging
from typing import Optional, Protocol

from app.schemas.search import PendingPage

logger = logging.getLogger(__name__)

PAGE_SIZE = 200
RETRY_DELAY_SECONDS = 60


class PendingGateway(Protocol):
    """The Gaia call the reconciler needs (adapters.gaia_client.GaiaClient)."""

    async def list_pending(self, after: Optional[str] = None, limit: Optional[int] = None) -> PendingPage: ...


class RecordSink(Protocol):
    """Where pending records go (services.search_index_queue.SearchIndexQueue)."""

    async def put(self, resource: str, record_id: str) -> None: ...


class SearchReconciler:
    """Owns when the pending list is paged into the queue: one run at a time, one more when asked during a run."""

    def __init__(self, gaia: PendingGateway, sink: RecordSink, page_size: int = PAGE_SIZE,
                 retry_delay: float = RETRY_DELAY_SECONDS):
        self._gaia = gaia
        self._sink = sink
        self._page_size = page_size
        self._retry_delay = retry_delay
        self._task: Optional[asyncio.Task] = None
        self._again = False
        self._soon: Optional[asyncio.TimerHandle] = None

    def request(self) -> None:
        """Start a reconcile, or ask for one more after the current one."""
        if self._task and not self._task.done():
            self._again = True
            return
        self._task = asyncio.create_task(self._run_until_settled(), name="search-reconcile")

    def request_soon(self) -> None:
        """Ask for a reconcile after the retry delay, unless one is already due."""
        if self._soon is None:
            self._soon = asyncio.get_running_loop().call_later(self._retry_delay, self._request_due)

    def _request_due(self) -> None:
        self._soon = None
        self.request()

    async def wait(self) -> None:
        """Wait for the current reconcile, and the one more it was asked for, to finish."""
        if self._task:
            await asyncio.shield(self._task)

    async def stop(self) -> None:
        if self._soon:
            self._soon.cancel()
            self._soon = None
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run_until_settled(self) -> None:
        while True:
            self._again = False
            await self._run_once()
            if not self._again:
                return

    async def _run_once(self) -> None:
        queued = 0
        cursor: Optional[str] = None
        try:
            while True:
                page = await self._gaia.list_pending(cursor, self._page_size)
                for record in page.records:
                    await self._sink.put(record.resource, record.id)
                queued += len(page.records)
                if page.next_cursor is None:
                    break
                if page.next_cursor == cursor:
                    # A cursor that does not move would page forever.
                    raise ValueError(f"Gaia returned the same cursor twice: {cursor}")
                cursor = page.next_cursor
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The next reconcile starts over; what was queued is indexed meanwhile.
            logger.error("Reconcile stopped after %s pending records: %s: %s", queued, type(e).__name__, e)
            return
        logger.info("Reconcile queued %s pending records", queued)
