import os
from typing import Optional

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.adapters.fastembed_adapter import FastEmbedAdapter


class EmbeddingAdapterFactory:
    """
    Creates the embedding adapter for the configured provider.

    The provider comes from EMBEDDING_PROVIDER (default "fastembed"). New
    providers subclass BaseEmbeddingAdapter and register in _adapters.
    """

    _adapters = {
        "fastembed": FastEmbedAdapter,
    }

    @classmethod
    def create_embedding_adapter(cls, provider: Optional[str] = None) -> BaseEmbeddingAdapter:
        provider = (provider or os.getenv("EMBEDDING_PROVIDER") or "fastembed").lower()
        if provider not in cls._adapters:
            available = ", ".join(cls._adapters)
            raise ValueError(f"Unsupported embedding provider: {provider}. Available: {available}")
        return cls._adapters[provider]()
