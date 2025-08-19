"""
Issue planning service module.

This module provides service layer functionality for AI-powered issue planning.
It acts as an intermediary between the router layer and the AI factory, handling
the communication with AI providers and managing error responses.

The service abstracts the AI provider interaction details and provides a clean
interface for the router to obtain AI-generated responses for issue planning
requests.

Author:
    Rana Nouman <ranamnouman@gmail.com>
"""

from app.factories.ai_service_factory import AIServiceFactory

async def get_ai_response(message: list, provider: str = None):
    """
    Get AI response for issue planning through the AI service factory.
    
    This function serves as the main service method for obtaining AI-generated
    responses for issue planning requests. It uses the AI service factory to
    create the appropriate AI adapter and handles communication with the AI
    provider.
    
    The function includes comprehensive error handling for configuration issues,
    provider availability, and general service errors. It returns standardized
    response objects that include success status, response content, and error
    information when applicable.
    
    Args:
        message (list): A list of message objects to send to the AI provider.
            Each message should be a dictionary with 'role' and 'content' keys.
            Example: [
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"}
            ]
        provider (str, optional): The AI provider to use for the request.
            If None, uses the default provider configured in the factory.
            Defaults to None.
    
    Returns:
        dict: A response object containing the following structure:
            On success:
            {
                "success": True,
                "response": str,        # The AI-generated response text
                "provider": str,        # The provider name used
                "model": str,           # The model used (if available)
                "tokens_used": int      # Number of tokens used (if available)
            }
            
            On configuration error:
            {
                "success": False,
                "error": str,           # Configuration error message
                "available_providers": list  # List of available providers
            }
            
            On service error:
            {
                "success": False,
                "error": str            # Service error message
            }
    
    Raises:
        ValueError: If the specified provider is not supported or available.
        Exception: For other errors during AI service communication or processing.
    
    Example:
        # Send a message to the default AI provider
        response = await get_ai_response([
            {"role": "user", "content": "Plan this issue..."}
        ])
        
        # Send a message to a specific provider
        response = await get_ai_response([
            {"role": "user", "content": "Plan this issue..."}
        ], provider="anthropic")
        
        # Check response
        if response["success"]:
            ai_response = response["response"]
        else:
            error_message = response["error"]
    """
    try:
        ai_service = AIServiceFactory.create_ai_service(provider)
        
        result = await ai_service.chat(message)
        
        return result
        
    except ValueError as e:
        return {
            "success": False,
            "error": f"Configuration error: {str(e)}",
            "available_providers": AIServiceFactory.get_available_providers()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Service error: {str(e)}"
        } 