"""
Main FastAPI application module.

This module serves as the entry point for the LLM API application. It creates
and configures the FastAPI application instance, sets up debugging capabilities
conditionally based on LOG_LEVEL environment variable, and provides the foundation
for the API server.

The application is designed to provide AI-powered services through a RESTful
API interface, with support for health checks and issue planning functionality.
It includes debugpy integration for remote debugging capabilities during development,
but only when LOG_LEVEL is set to 'debug'.

The application includes four router modules:
- health: Provides health check endpoints for monitoring service status
- planissue: Provides AI-powered issue planning functionality
- docs: Provides AI help for the Docs editor (ask, complete, review)
- embeddings: Turns texts into search vectors with the in-process embedding model

Author:
    Rana Nouman <ranamnouman@gmail.com>
"""

import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app.dependencies import get_embedding_adapter_or_none
from app.routers import docs, embeddings, health, planissue
from app.services.ai_call import AIProviderError
from app.services.embedding_service import EmbeddingUnavailableError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model in the background: the API starts at once and /embeddings answers 503 until it is ready."""
    adapter = await get_embedding_adapter_or_none()
    load = asyncio.create_task(adapter.load()) if adapter else None
    yield
    if load and not load.done():
        load.cancel()


# Create FastAPI instance
app = FastAPI(
    title="LLM API",
    description="A simple FastAPI LLM application",
    version="1.0.0",
    lifespan=lifespan
)

# Conditionally enable debugging based on LOG_LEVEL
log_level = os.getenv("LOG_LEVEL", "info").lower()
if log_level == "debug":
    import debugpy
    # listen for debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("Debug mode enabled. Waiting for debugger to attach...")

# Include routers
app.include_router(health.router)
app.include_router(planissue.router)
app.include_router(docs.router)
app.include_router(embeddings.router)


@app.exception_handler(AIProviderError)
async def ai_provider_error_handler(request: Request, exc: AIProviderError) -> JSONResponse:
    """A failed or unusable provider reply is a bad gateway, never an empty result."""
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})


@app.exception_handler(EmbeddingUnavailableError)
async def embedding_unavailable_handler(request: Request, exc: EmbeddingUnavailableError) -> JSONResponse:
    """No vector is better than a fake one: an unloaded or failing model is a 503."""
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)})
