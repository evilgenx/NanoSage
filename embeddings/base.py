# embeddings/base.py
import abc
import torch

class BaseEmbedder(abc.ABC):
    """Abstract base class for embedding models."""

    @abc.abstractmethod
    def embed(self, text: str) -> torch.Tensor | None:
        """
        Generates an embedding for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            torch.Tensor | None: The embedding tensor, or None if embedding failed.
        """
        pass

    # Optional: Add methods for batch embedding if needed later
    # @abc.abstractmethod
    # def embed_batch(self, texts: list[str]) -> list[torch.Tensor | None]:
    #     pass
