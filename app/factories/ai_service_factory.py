import os
from typing import Optional
from app.adapters.base_ai_adapter import BaseAIAdapter
from app.adapters.anthropic_adapter import AnthropicAdapter

class AIServiceFactory:
    """
    Factory class for creating AI service instances based on provider configuration.
    
    This factory implements the Factory pattern to create appropriate AI adapter
    instances based on the specified provider. It centralizes the creation logic
    and provides a unified interface for obtaining AI services throughout the
    application.
    
    The factory supports multiple AI providers and can be easily extended by
    adding new adapters to the _adapters dictionary. It also provides utility
    methods to discover available providers.
    
    Attributes:
        _adapters (dict): A mapping of provider names to their corresponding
            adapter classes. This is a class-level attribute that defines
            all available AI providers in the system.
    
    Author:
        Rana Nouman <ranamnouman@gmail.com>
    """
    
    _adapters = {
        "anthropic": AnthropicAdapter,
        # Future providers can be added here:
        # "openai": OpenAIAdapter,
        # "google": GoogleAdapter,
    }
    
    @classmethod
    def create_ai_service(cls, provider: Optional[str] = None) -> BaseAIAdapter:
        """
        Create an AI service instance for the specified provider.
        
        This method creates and returns an appropriate AI adapter instance
        based on the provider parameter. If no provider is specified, it
        defaults to the value of the AI_PROVIDER environment variable,
        or 'anthropic' if the environment variable is not set.
        
        The method performs validation to ensure the requested provider
        is supported and available in the system.
        
        Args:
            provider (Optional[str]): The name of the AI provider to use.
                If None, uses the AI_PROVIDER environment variable or
                defaults to 'anthropic'. The provider name is case-insensitive.
        
        Returns:
            BaseAIAdapter: An instance of the appropriate AI adapter class
                for the specified provider.
        
        Raises:
            ValueError: If the specified provider is not supported or
                not available in the _adapters dictionary.
        
        Example:
            # Create an Anthropic adapter
            ai_service = AIServiceFactory.create_ai_service("anthropic")
            
            # Create using environment variable
            ai_service = AIServiceFactory.create_ai_service()
            
            # Use the service
            response = await ai_service.chat("Hello, how are you?")
        """
        if provider is None:
            provider = os.getenv("AI_PROVIDER", "anthropic").lower()
        
        provider = provider.lower()
        
        if provider not in cls._adapters:
            available_providers = ", ".join(cls.get_available_providers())
            raise ValueError(f"Unsupported AI provider: {provider}. Available: {available_providers}")
        
        return cls._adapters[provider]()
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get a list of all available AI providers in the system.
        
        This method returns the names of all AI providers that are currently
        supported and can be instantiated through the factory. It's useful
        for discovery, validation, and providing user feedback about
        available options.
        
        Returns:
            list: A list of strings containing the names of all available
                AI providers (e.g., ["anthropic", "openai", "google"]).
        
        Example:
            providers = AIServiceFactory.get_available_providers()
            print(f"Available providers: {providers}")
            # Output: Available providers: ['anthropic']
        """
        return list(cls._adapters.keys()) 