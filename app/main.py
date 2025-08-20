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

The application includes two main router modules:
- health: Provides health check endpoints for monitoring service status
- planissue: Provides AI-powered issue planning functionality

Author:
    Rana Nouman <ranamnouman@gmail.com>
"""

import os
from fastapi import FastAPI
from app.routers import health, planissue

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