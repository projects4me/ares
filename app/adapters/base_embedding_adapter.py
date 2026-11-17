from abc import ABC, abstractmethod
from typing import List, Optional


class BaseEmbeddingAdapter(ABC):
    """
    Abstract base class for embedding model adapters.

    An adapter owns one embedding model: it loads the model once and turns
    texts into vectors. Task prefixes, validation and normalisation belong to
    the embedding service, so an adapter embeds the texts exactly as given.
    """

    @abstractmethod
    def model_name(self) -> str:
        """The model's short name, e.g. "nomic-embed-text-v1.5"."""

    @abstractmethod
    def dimensions(self) -> int:
        """The length of every vector the model returns."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready to embed."""

    @abstractmethod
    def load_error(self) -> Optional[str]:
        """Why the last load failed, or None."""

    @abstractmethod
    async def load(self) -> None:
        """Load the model once; later calls do nothing. A failure is kept in load_error()."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed the texts, one vector per text in the same order, without blocking the event loop."""
