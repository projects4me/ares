"""
Embedding use case: turn texts into search vectors for one task.

Used by POST /embeddings (search queries and records from Gaia). It adds the model's task prefix, checks every vector the
adapter returns and L2-normalises it. Any failure raises
EmbeddingUnavailableError, which app.main maps to 503; it never returns a
fake or partial vector.
"""

import math
from dataclasses import dataclass
from typing import List

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.schemas.embeddings import EmbeddingTask

TASK_PREFIXES = {
    EmbeddingTask.document: "search_document: ",
    EmbeddingTask.query: "search_query: ",
}


@dataclass(frozen=True)
class Embeddings:
    """The model's vectors for the texts, in their order."""

    model: str
    dimensions: int
    vectors: List[List[float]]


class EmbeddingUnavailableError(Exception):
    """The model is not loaded, or it failed to embed the texts."""


async def embed(adapter: BaseEmbeddingAdapter, texts: List[str], task: EmbeddingTask) -> Embeddings:
    """Embed the texts for the task, one normalised vector per text in the same order."""
    if not adapter.is_loaded():
        reason = adapter.load_error() or "it is still loading"
        raise EmbeddingUnavailableError(f"The embedding model {adapter.model_name()} is not loaded: {reason}")

    prefix = TASK_PREFIXES[task]
    try:
        vectors = await adapter.embed([prefix + text for text in texts])
    except Exception as e:
        raise EmbeddingUnavailableError(f"The embedding model failed: {type(e).__name__}") from e

    dimensions = adapter.dimensions()
    if len(vectors) != len(texts):
        raise EmbeddingUnavailableError(
            f"The embedding model returned {len(vectors)} vectors for {len(texts)} texts")
    return Embeddings(
        model=adapter.model_name(),
        dimensions=dimensions,
        vectors=[_normalised(vector, dimensions) for vector in vectors],
    )


def _normalised(vector: List[float], dimensions: int) -> List[float]:
    if len(vector) != dimensions:
        raise EmbeddingUnavailableError(
            f"The embedding model returned a {len(vector)}-dimension vector, expected {dimensions}")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise EmbeddingUnavailableError("The embedding model returned an unusable vector")
    return [value / norm for value in vector]
