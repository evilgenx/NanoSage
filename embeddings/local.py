# embeddings/local.py
import torch
import numpy as np # Added numpy for serialization
import logging # Added logging
from typing import Optional # Added Optional
from .base import BaseEmbedder
from cache_manager import CacheManager # Added CacheManager import

logger = logging.getLogger(__name__) # Added logger

# Use try-except blocks for optional dependencies
try:
    from transformers import ColPaliForRetrieval, ColPaliProcessor
except ImportError:
    ColPaliForRetrieval = None
    ColPaliProcessor = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class ColPaliEmbedder(BaseEmbedder):
    """Embedder using the ColPali model."""

    def __init__(self, model_hf_name="vidore/colpali-v1.2-hf", device="cpu", cache_manager: Optional[CacheManager] = None): # Added cache_manager
        if ColPaliForRetrieval is None or ColPaliProcessor is None:
            raise ImportError("ColPali dependencies not found. Please install transformers: pip install transformers")

        self.device = device
        self.model_hf_name = model_hf_name
        self.cache_manager = cache_manager # Store cache manager
        logger.info(f"Loading local ColPali model ({self.model_hf_name}) onto device: {self.device}")
        try:
            # Consider making dtype configurable if needed
            self.model = ColPaliForRetrieval.from_pretrained(
                self.model_hf_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(self.model_hf_name)
        except Exception as e:
            logger.error(f"Failed to load ColPali model '{self.model_hf_name}': {e}")
            raise

    def embed(self, text: str) -> torch.Tensor | None:
        """Generates embeddings using the loaded ColPali model, with caching."""
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text with ColPali.")
            return None

        # --- Cache Check ---
        if self.cache_manager:
            cached_embedding_np = self.cache_manager.get_embedding(text, self.model_hf_name)
            if cached_embedding_np is not None:
                # Convert numpy array from cache back to torch tensor
                return torch.from_numpy(cached_embedding_np).float() # Ensure float tensor
        # --- End Cache Check ---

        try:
            logger.debug(f"Embedding text with ColPali (cache miss): {text[:50]}...")
            inputs = self.processor(text=[text], truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling
            embedding = outputs.embeddings.mean(dim=1).squeeze(0).cpu() # Get embedding on CPU

            # --- Store in Cache ---
            if self.cache_manager:
                # Convert torch tensor to numpy array then bytes for storage
                embedding_np = embedding.numpy().astype(np.float32) # Ensure float32 numpy array
                self.cache_manager.store_embedding(text, self.model_hf_name, embedding_np)
            # --- End Store in Cache ---

            return embedding
        except Exception as e:
            snippet = text[:100].replace('\n', ' ') + "..."
            error_message = str(e)
            logger.error(f"Failed to embed text (snippet: '{snippet}...') using ColPali: {error_message}")
            return None


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using a Sentence Transformer model."""

    def __init__(self, model_st_name="all-MiniLM-L6-v2", device="cpu", cache_manager: Optional[CacheManager] = None): # Added cache_manager
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformer not found. Please install sentence-transformers: pip install sentence-transformers")

        self.device = device
        self.model_st_name = model_st_name
        self.cache_manager = cache_manager # Store cache manager
        logger.info(f"Loading local Sentence Transformer model ({self.model_st_name}) onto device: {self.device}")
        try:
            self.model = SentenceTransformer(self.model_st_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model '{self.model_st_name}': {e}")
            raise

    def embed(self, text: str) -> torch.Tensor | None:
        """Generates embeddings using the loaded Sentence Transformer model, with caching."""
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text with SentenceTransformer.")
            return None

        # --- Cache Check ---
        if self.cache_manager:
            cached_embedding_np = self.cache_manager.get_embedding(text, self.model_st_name)
            if cached_embedding_np is not None:
                # Convert numpy array from cache back to torch tensor
                return torch.from_numpy(cached_embedding_np).float() # Ensure float tensor
        # --- End Cache Check ---

        try:
            logger.debug(f"Embedding text with SentenceTransformer (cache miss): {text[:50]}...")
            # convert_to_tensor=True gives torch tensor directly
            embedding = self.model.encode(text, convert_to_tensor=True).cpu() # Get embedding on CPU

            # --- Store in Cache ---
            if self.cache_manager:
                # Convert torch tensor to numpy array then bytes for storage
                embedding_np = embedding.numpy().astype(np.float32) # Ensure float32 numpy array
                self.cache_manager.store_embedding(text, self.model_st_name, embedding_np)
            # --- End Store in Cache ---

            return embedding
        except Exception as e:
            snippet = text[:100].replace('\n', ' ') + "..."
            error_message = str(e)
            logger.error(f"Failed to embed text (snippet: '{snippet}...') using SentenceTransformer: {error_message}")
            return None
