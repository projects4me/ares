"""
Request and response models for the embeddings endpoint.

These models are the contract between Gaia and POST /embeddings. The router
validates requests against them and the embedding service reads from them,
so they live outside both layers.
"""

from enum import Enum
from typing import Annotated, List

from pydantic import BaseModel, Field, StringConstraints

TEXTS_MAX = 64
TEXT_MAX = 8000

EmbeddingText = Annotated[str, StringConstraints(min_length=1, max_length=TEXT_MAX)]


class EmbeddingTask(str, Enum):
    """What the texts are for: records to be searched, or a search query."""

    document = "document"
    query = "query"


class EmbeddingsRequest(BaseModel):
    """1 to 64 non-empty texts to embed for one task."""

    texts: List[EmbeddingText] = Field(..., min_length=1, max_length=TEXTS_MAX)
    task: EmbeddingTask


class EmbeddingsResponse(BaseModel):
    """One L2-normalised vector per text, in the order of the request."""

    model: str
    dimensions: int
    embeddings: List[List[float]]
