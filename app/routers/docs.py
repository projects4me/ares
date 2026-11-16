"""
Docs AI router: HTTP endpoints for the Projects4Me Docs editor.

Gaia (core/services/LLM.php) is the only caller. The routes validate the
request and hand it to the use-case service; app.main maps a provider
failure (AIProviderError) to 502.
Prompts and provider logic live in the services.
"""

from fastapi import APIRouter, Depends

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.dependencies import get_ai_adapter
from app.schemas.docs import (
    AskRequest, AskResponse, CompleteRequest, CompleteResponse, ReviewRequest, ReviewResponse,
)
from app.services import docs_ask_service, docs_complete_service, docs_review_service

router = APIRouter(
    prefix="/docs",
    tags=["docs"]
)


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, adapter: BaseAIAdapter = Depends(get_ai_adapter)) -> AskResponse:
    """Answer the writer's instruction about the selection with replacement HTML."""
    return await docs_ask_service.ask(adapter, request)


@router.post("/complete", response_model=CompleteResponse)
async def complete(request: CompleteRequest, adapter: BaseAIAdapter = Depends(get_ai_adapter)) -> CompleteResponse:
    """Continue the sentence being written."""
    return await docs_complete_service.complete(adapter, request)


@router.post("/review", response_model=ReviewResponse, response_model_exclude_none=True)
async def review(request: ReviewRequest, adapter: BaseAIAdapter = Depends(get_ai_adapter)) -> ReviewResponse:
    """Review the whole document and return findings tied to exact quotes."""
    return await docs_review_service.review(adapter, request)
