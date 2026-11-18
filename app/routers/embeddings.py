"""
Embeddings router: POST /embeddings turns texts into search vectors.

Gaia sends search queries here. The route validates the request and hands it
to the embedding service; app.main maps EmbeddingUnavailableError to 503.
"""

from fastapi import APIRouter, Depends

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.dependencies import get_embedding_adapter
from app.schemas.embeddings import EmbeddingsRequest, EmbeddingsResponse
from app.services import embedding_service

router = APIRouter(
    prefix="/embeddings",
    tags=["embeddings"]
)


@router.post("", response_model=EmbeddingsResponse)
async def create_embeddings(
    request: EmbeddingsRequest, adapter: BaseEmbeddingAdapter = Depends(get_embedding_adapter),
) -> EmbeddingsResponse:
    """Embed 1 to 64 texts for a document or query task."""
    result = await embedding_service.embed(adapter, request.texts, request.task)
    return EmbeddingsResponse(model=result.model, dimensions=result.dimensions, embeddings=result.vectors)
