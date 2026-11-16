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

The application includes three router modules:
- health: Provides health check endpoints for monitoring service status
- planissue: Provides AI-powered issue planning functionality
- docs: Provides AI help for the Docs editor (ask, complete, review)

Author:
    Rana Nouman <ranamnouman@gmail.com>
"""

import os
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app.routers import docs, health, planissue
from app.services.ai_call import AIProviderError

# Create FastAPI instance
app = FastAPI(
    title="LLM API",
    description="A simple FastAPI LLM application",
    version="1.0.0"
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


@app.exception_handler(AIProviderError)
async def ai_provider_error_handler(request: Request, exc: AIProviderError) -> JSONResponse:
    """A failed or unusable provider reply is a bad gateway, never an empty result."""
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})