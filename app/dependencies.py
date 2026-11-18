"""
FastAPI dependencies shared by the routers.

get_ai_adapter hands a route the configured AI adapter, so the routers never
name a provider and tests can override it with a fake adapter. The adapter is
built once and reused, so its HTTP connection pool is shared across requests.
get_embedding_adapter does the same for the embedding model;
get_embedding_adapter_or_none is its variant for the health check, which must
answer even when the embedding provider is broken.
"""

import logging
from typing import Optional

from fastapi import HTTPException, status

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.factories.ai_service_factory import AIServiceFactory
from app.factories.embedding_adapter_factory import EmbeddingAdapterFactory


logger = logging.getLogger(__name__)

_adapter: Optional[BaseAIAdapter] = None


async def get_ai_adapter() -> BaseAIAdapter:
    global _adapter
    if _adapter is None:
        try:
            _adapter = AIServiceFactory.create_ai_service()
        except Exception as e:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"AI provider is not available: {e}")
    return _adapter


_embedding_adapter: Optional[BaseEmbeddingAdapter] = None


async def get_embedding_adapter() -> BaseEmbeddingAdapter:
    """The one embedding adapter, so the model is loaded once and shared by every request."""
    global _embedding_adapter
    if _embedding_adapter is None:
        try:
            _embedding_adapter = EmbeddingAdapterFactory.create_embedding_adapter()
        except Exception as e:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Embedding provider is not available: {e}")
    return _embedding_adapter


async def get_embedding_adapter_or_none() -> Optional[BaseEmbeddingAdapter]:
    """The embedding adapter, or None (logged) when it can't be built."""
    try:
        return await get_embedding_adapter()
    except HTTPException as e:
        logger.error("%s", e.detail)
        return None
