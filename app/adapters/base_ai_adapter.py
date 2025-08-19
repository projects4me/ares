from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAIAdapter(ABC):
    """
    Abstract base class for AI provider adapters.
    
    This class defines the interface that all AI provider adapters must implement.
    It provides a standardized way to interact with different AI services through
    a common interface, ensuring consistency across various AI providers.
    
    All concrete adapter implementations should inherit from this class and
    implement the required abstract methods to provide AI chat functionality.
    
    Attributes:
        None (abstract base class)
    
    Author:
        Rana Nouman <ranamnouman@gmail.com>
    """
    
    @abstractmethod
    async def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Send a chat message to the AI provider and return the response.
        
        This is an abstract method that must be implemented by all concrete
        adapter classes. It handles the communication with the specific AI
        provider's API and returns a standardized response format.
        
        Args:
            message (str): The message to send to the AI provider
            **kwargs: Additional provider-specific parameters that may be required
                by different AI providers (e.g., model selection, token limits,
                temperature settings, etc.)
        
        Returns:
            Dict[str, Any]: A standardized response object containing:
                - success (bool): Whether the API call was successful
                - response (str): The response text from the AI provider (if successful)
                - provider (str): The provider name for identification
                - Additional provider-specific metadata (model used, tokens consumed, etc.)
                - error (str): Error message (if unsuccessful)
        
        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
            Exception: Provider-specific exceptions may be raised by implementations
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the AI provider for this adapter.
        
        This method returns a unique identifier for the AI provider that this
        adapter connects to. It's used for logging, identification, and
        provider-specific logic throughout the application.
        
        Returns:
            str: A unique string identifier for the AI provider
                (e.g., "anthropic", "openai", "google", etc.)
        
        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the AI provider service is accessible and healthy.
        
        This method performs a lightweight health check to verify that the
        AI provider's API is accessible and responding correctly. It should
        make a minimal API call to test connectivity and service status.
        
        Returns:
            Dict[str, Any]: A health check response object containing:
                - success (bool): Whether the health check was successful
                - provider (str): The provider name for identification
                - status (str): Health status message
                - error (str): Error message (if unsuccessful)
        
        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
        """
        pass
    
    @abstractmethod
    async def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with the AI provider API.
        
        This method handles the authentication process with the AI provider's API.
        It validates API credentials, establishes connection, and verifies that
        the authentication is successful. This method should be called before
        making any API calls to ensure proper authentication.
        
        Returns:
            Dict[str, Any]: An authentication response object containing:
                - success (bool): Whether the authentication was successful
                - provider (str): The provider name for identification
                - status (str): Authentication status message
                - error (str): Error message (if unsuccessful)
        
        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
        """
        pass 