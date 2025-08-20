"""
Health check router module.

This module provides health check endpoints for monitoring the application's
status and ensuring it's running properly. It's commonly used by load balancers,
monitoring systems, and deployment tools to verify service availability.

The health endpoint returns a comprehensive status including both application
health and LLM service connectivity status.

Author:
    Rana Nouman <ranamnouman@gmail.com>
"""

from fastapi import APIRouter
from app.factories.ai_service_factory import AIServiceFactory

router = APIRouter(
    tags=["health"]
)

@router.get("/")
async def read_root():
    """
    Health check endpoint to verify service status.
    
    This endpoint provides a comprehensive health check that verifies both
    the application status and the LLM service connectivity. It makes a
    lightweight API call to the configured AI provider to ensure the service
    is accessible and responding correctly.
    
    The endpoint is designed to respond quickly while providing meaningful
    health information for monitoring systems, load balancers, and deployment tools.
    
    Returns:
        dict: A comprehensive health status containing:
            - message (str): Application status message
            - llm_health (dict): LLM service health check results
                - success (bool): Whether the LLM service is accessible
                - provider (str): The AI provider name
                - status (str): Health status message
                - error (str): Error message (if unsuccessful)
    
    Example:
        GET /health/
        Response: {
            "message": "Application is healthy",
            "llm_health": {
                "success": true,
                "provider": "anthropic",
                "status": "Anthropic API is accessible and healthy"
            }
        }
    
    Raises:
        None: This endpoint is designed to always return successfully
            unless there's a critical system failure.
    """
    try:
        # Create AI service instance and perform health check
        ai_service = AIServiceFactory.create_ai_service()
        llm_health = await ai_service.health_check()
        
        return {
            "message": "Application is healthy",
            "llm_health": llm_health
        }
    except Exception as e:
        return {
            "message": "Application is healthy",
            "llm_health": {
                "success": False,
                "provider": "unknown",
                "status": "LLM service health check failed",
                "error": f"Health check error: {str(e)}"
            }
        } 