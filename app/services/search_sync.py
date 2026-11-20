"""
Keeps record embeddings in step with Gaia: the running parts of semantic search indexing.

Once the embedding model has loaded it starts the indexing workers and the Hermes
listener (whose every subscribe asks for a reconcile), reconciles once at once (so a
backlog is indexed even while Hermes is unreachable), and asks for a reconcile on a
timer. Before the model is loaded nothing runs; the first reconcile covers what changed
meanwhile. A model that does not load leaves indexing off, logged.
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class _Startable(Protocol):
    def start(self) -> None: ...

    async def stop(self) -> None: ...


class _Reconciler(Protocol):
    def request(self) -> None: ...

    async def stop(self) -> None: ...


class SearchSync:
    """Owns the lifecycle of indexing: start the parts once the model is loaded, run the reconcile timer, stop."""

    def __init__(self, queue: _Startable, reconciler: _Reconciler, listener: _Startable, interval_seconds: float,
                 model_loaded: Callable[[], bool], close: Callable[[], Awaitable[None]],
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self._queue = queue
        self._reconciler = reconciler
        self._listener = listener
        self._interval = interval_seconds
        self._model_loaded = model_loaded
        self._close = close
        self._sleep = sleep
        self._task: Optional[asyncio.Task] = None

    def start(self, model_load: "asyncio.Future") -> None:
        """Start once model_load (the embedding model's load task) is done."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(model_load), name="search-sync")

    async def stop(self) -> None:
        """Stop every part, even when one of them fails to stop, and close the Gaia client."""
        steps = [self._cancel_run, self._listener.stop, self._reconciler.stop, self._queue.stop, self._close]
        for step in steps:
            try:
                await step()
            except Exception as e:
                logger.error("Stopping semantic search indexing: %s failed: %s", step.__qualname__, type(e).__name__)

    async def _cancel_run(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self, model_load: "asyncio.Future") -> None:
        try:
            # Shielded: stopping indexing must not cancel the model load the API also waits for.
            await asyncio.shield(model_load)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("The embedding model load failed: %s", type(e).__name__)
        if not self._model_loaded():
            logger.error("Semantic search indexing is off: the embedding model did not load")
            return
        self._queue.start()
        self._listener.start()
        self._reconciler.request()
        logger.info("Semantic search indexing started; reconciling every %ss", self._interval)
        while True:
            await self._sleep(self._interval)
            self._reconciler.request()
