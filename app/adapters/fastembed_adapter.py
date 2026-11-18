"""
fastembed adapter: runs an embedding model in-process on ONNX Runtime (CPU).

Configured from env:
- EMBEDDING_MODEL: the fastembed model (default nomic-ai/nomic-embed-text-v1.5)
- EMBEDDING_CACHE_DIR: where the weights live; the image downloads them there at build time,
  and a cached model loads without the network
- EMBEDDING_THREADS: ONNX Runtime threads (default: ONNX Runtime's own choice)

fastembed is imported on load, so the tests and the routes that never embed
don't pay for ONNX Runtime.
"""

import asyncio
import logging
import os
import threading
from typing import List, Optional

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def _threads_env() -> Optional[int]:
    try:
        value = int(os.getenv("EMBEDDING_THREADS") or 0)
    except ValueError:
        return None
    return value if value > 0 else None


class FastEmbedAdapter(BaseEmbeddingAdapter):
    """Loads one fastembed TextEmbedding model and embeds texts in a worker thread."""

    def __init__(self):
        self._model_id = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
        self._cache_dir = os.getenv("EMBEDDING_CACHE_DIR") or None
        self._threads = _threads_env()
        self._model = None
        self._dimensions = 0
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    def model_name(self) -> str:
        return self._model_id.rsplit("/", 1)[-1]

    def dimensions(self) -> int:
        return self._dimensions

    def is_loaded(self) -> bool:
        return self._model is not None

    def load_error(self) -> Optional[str]:
        return self._error

    async def load(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from fastembed import TextEmbedding

                described = [m for m in TextEmbedding.list_supported_models() if m["model"] == self._model_id]
                if not described:
                    raise ValueError(f"fastembed does not support the model {self._model_id}")
                model = self._create(TextEmbedding)
                # Dimensions first: requests read is_loaded() from the event loop without the lock.
                self._dimensions = described[0]["dim"]
                self._error = None
                self._model = model
            except Exception as e:
                self._error = f"{type(e).__name__}: {e}"
                logger.error("The embedding model %s failed to load: %s", self._model_id, self._error)

    def _create(self, text_embedding):
        """Load from the cache without the network first (the image is built with the weights), then download."""
        try:
            return text_embedding(
                self._model_id, cache_dir=self._cache_dir, threads=self._threads, local_files_only=True)
        except Exception as e:
            logger.info("The embedding model %s is not in the cache (%s: %s), downloading it",
                        self._model_id, type(e).__name__, e)
            return text_embedding(self._model_id, cache_dir=self._cache_dir, threads=self._threads)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("The embedding model is not loaded")
        return await asyncio.to_thread(self._embed, texts)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=len(texts))]
