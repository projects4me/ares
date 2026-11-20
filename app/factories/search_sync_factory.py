"""
Builds semantic search indexing from env, or None when it is not configured.

Env:
- HERMES_URL, GAIA_URL, GAIA_CLIENT_ID, GAIA_CLIENT_SECRET: all required; without
  any of them indexing is off (logged), and the rest of Ares runs as before
- SEARCH_RECONCILE_INTERVAL: seconds between timed reconciles (default 3600)
- SEARCH_INDEX_QUEUE_SIZE: records that may wait to be indexed (default 1000)
- SEARCH_INDEX_CONCURRENCY: records indexed at once (default 2)
"""

import functools
import logging
import os
from typing import Optional

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.adapters.gaia_client import GaiaClient
from app.adapters.hermes_listener import HermesListener
from app.services.search_index_queue import SearchIndexQueue
from app.services.search_index_service import index_record, is_transient
from app.services.search_reconciler import SearchReconciler
from app.services.search_sync import SearchSync

logger = logging.getLogger(__name__)

REQUIRED_ENV = ("HERMES_URL", "GAIA_URL", "GAIA_CLIENT_ID", "GAIA_CLIENT_SECRET")
DEFAULT_RECONCILE_INTERVAL = 3600
DEFAULT_QUEUE_SIZE = 1000
DEFAULT_CONCURRENCY = 2


class SearchSyncFactory:
    """Wires the Gaia client, Hermes listener, indexing queue and reconciler into one SearchSync."""

    @classmethod
    def create_search_sync(cls, embedder: Optional[BaseEmbeddingAdapter]) -> Optional[SearchSync]:
        missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
        if missing:
            logger.warning("Semantic search indexing is off: %s not set", ", ".join(missing))
            return None
        if embedder is None:
            logger.warning("Semantic search indexing is off: there is no embedding model")
            return None

        gaia = GaiaClient.from_env()

        def on_failure(error: Exception) -> None:
            # A Gaia blip or a model failure is caught up on soon; anything else waits for the timer.
            if is_transient(error):
                reconciler.request_soon()

        queue = SearchIndexQueue(functools.partial(index_record, gaia, embedder),
                                 max_size=_positive_int_env("SEARCH_INDEX_QUEUE_SIZE", DEFAULT_QUEUE_SIZE),
                                 concurrency=_positive_int_env("SEARCH_INDEX_CONCURRENCY", DEFAULT_CONCURRENCY),
                                 on_failure=on_failure)
        reconciler = SearchReconciler(gaia, queue)
        listener = HermesListener(os.environ["HERMES_URL"], gaia.auth, on_change=queue.offer,
                                  on_subscribed=reconciler.request)
        return SearchSync(queue, reconciler, listener,
                          interval_seconds=_positive_int_env("SEARCH_RECONCILE_INTERVAL", DEFAULT_RECONCILE_INTERVAL),
                          model_loaded=embedder.is_loaded, close=gaia.aclose)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name) or default)
    except ValueError:
        logger.warning("%s is not a whole number; using %s", name, default)
        return default
    return value if value > 0 else default
