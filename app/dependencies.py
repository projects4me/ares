"""
FastAPI dependencies shared by the routers.

get_ai_adapter hands a route the configured AI adapter, so the routers never
name a provider and tests can override it with a fake adapter. The adapter is
built once and reused, so its HTTP connection pool is shared across requests.
"""

from typing import Optional

from fastapi import HTTPException, status

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.factories.ai_service_factory import AIServiceFactory


_adapter: Optional[BaseAIAdapter] = None


async def get_ai_adapter() -> BaseAIAdapter:
    global _adapter
    if _adapter is None:
        try:
            _adapter = AIServiceFactory.create_ai_service()
        except Exception as e:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"AI provider is not available: {e}")
    return _adapter
