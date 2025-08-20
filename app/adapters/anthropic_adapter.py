import anthropic
import os
from typing import Dict, Any
from .base_ai_adapter import BaseAIAdapter

class AnthropicAdapter(BaseAIAdapter):
    """
    Adapter for integrating with Anthropic's Claude API.
    
    This adapter provides a standardized interface for sending chat messages to Claude
    and handling responses in a consistent format across different AI providers.
    
    Attributes:
        client: The Anthropic client instance for API communication
    
    Author:
        Rana Nouman <ranamnouman@gmail.com>
    """
    
    def __init__(self):
        """
        Initialize the AnthropicAdapter.
        
        Sets up the adapter instance and authenticates with the Anthropic API.
        The client will be automatically available for all subsequent operations.
        """
        self.client = None
        self._authenticate()
    
    def _authenticate(self) -> None:
        """
        Authenticate with the Anthropic API.
        
        Validates the API key and creates the Anthropic client instance.
        This method is called during initialization to ensure the client
        is available for all subsequent operations.
        
        Raises:
            Exception: If the API key is invalid or missing
        """
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise Exception("ANTHROPIC_API_KEY environment variable is not set")
        
        # Create the Anthropic client with the API key
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    async def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with the Anthropic API.
        
        This method is maintained for compatibility with the base interface.
        Since authentication is handled during initialization, this method
        simply returns the current authentication status.
        
        Returns:
            Dict[str, Any]: Authentication response object containing:
                - success (bool): Whether the authentication was successful
                - provider (str): The provider name ("anthropic")
                - status (str): Authentication status message
                - error (str): Error message (if unsuccessful)
        """
        try:
            if not self.client:
                self._authenticate()
            
            return {
                "success": True,
                "provider": self.get_provider_name(),
                "status": "Successfully authenticated with Anthropic API"
            }
            
        except Exception as e:
            return {
                "success": False,
                "provider": self.get_provider_name(),
                "status": "Authentication with Anthropic API failed",
                "error": f"Authentication error: {str(e)}"
            }
    
    async def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Send a chat message to Claude and return the response.
        
        Handles message formatting, API communication, and response processing.
        Supports configurable model selection and token limits.
        
        Args:
            message (str): The message to send to Claude
            **kwargs: Additional parameters for the API call
                model (str, optional): The Claude model to use. 
                    Defaults to "claude-sonnet-4-20250514"
                max_tokens (int, optional): Maximum tokens for the response. 
                    Defaults to 10000
        
        Returns:
            Dict[str, Any]: Response object containing:
                - success (bool): Whether the API call was successful
                - response (str): The response text from Claude (if successful)
                - provider (str): The provider name ("anthropic")
                - model (str): The model used for the request
                - tokens_used (int, optional): Number of tokens used in the response
                - error (str): Error message (if unsuccessful)
        
        Raises:
            Exception: If the API call fails or credentials are invalid
        """
        try:
            # Extract parameters with defaults
            model = kwargs.get("model", "claude-sonnet-4-20250514")
            max_tokens = kwargs.get("max_tokens", 10000)
            
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=message
            )
            
            return {
                "success": True,
                "response": response.content[0].text,
                "provider": self.get_provider_name(),
                "model": model,
                "tokens_used": response.usage.output_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Anthropic API error: {str(e)}",
                "provider": self.get_provider_name()
            }
    
    def get_provider_name(self) -> str:
        """
        Return the provider name for this adapter.
        
        Used for identification and logging purposes.
        
        Returns:
            str: The provider name ("anthropic")
        """
        return "anthropic"
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the Anthropic API is accessible and healthy.
        
        Performs a lightweight health check by making a minimal API call to
        verify that the Anthropic service is accessible and responding correctly.
        Uses a simple test message with minimal tokens to avoid unnecessary costs.
        
        Returns:
            Dict[str, Any]: Health check response object containing:
                - success (bool): Whether the health check was successful
                - provider (str): The provider name ("anthropic")
                - status (str): Health status message
                - error (str): Error message (if unsuccessful)
        
        Raises:
            Exception: If the API call fails or credentials are invalid
        """
        try:
            if self.client:
                return {
                    "success": True,
                    "provider": self.get_provider_name(),
                    "status": "Anthropic API is accessible and healthy"
                }
            
            raise Exception("Anthropic client is not initialized")
        except Exception as e:
            return {
                "success": False,
                "provider": self.get_provider_name(),
                "status": "Anthropic API health check failed",
                "error": f"Anthropic API error: {str(e)}"
            } 