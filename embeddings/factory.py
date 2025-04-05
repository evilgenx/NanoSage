# embeddings/factory.py
from typing import Optional # Added Optional
from .base import BaseEmbedder
from .local import ColPaliEmbedder, SentenceTransformerEmbedder
from cache_manager import CacheManager # Added CacheManager import

# Mapping from simple names to classes and their specific model names if needed
# This could be expanded to include API embedders later if desired
EMBEDDER_REGISTRY = {
    "colpali": {
        "class": ColPaliEmbedder,
        "default_model_name": "vidore/colpali-v1.2-hf"
    },
    "all-minilm": {
        "class": SentenceTransformerEmbedder,
        "default_model_name": "all-MiniLM-L6-v2"
    },
    # Add other SentenceTransformer models here if needed, e.g.:
    # "multi-qa-MiniLM-L6-cos-v1": {
    #     "class": SentenceTransformerEmbedder,
    #     "default_model_name": "multi-qa-MiniLM-L6-cos-v1"
    # },
}

def create_embedder(embedding_model_name: str, device: str, cache_manager: Optional[CacheManager] = None) -> BaseEmbedder: # Added cache_manager
    """
    Factory function to create an embedder instance based on configuration.

    Args:
        embedding_model_name (str): The name identifying the desired embedding model
                                   (e.g., 'colpali', 'all-minilm').
        device (str): The device to load the model onto ('cpu', 'cuda').
                      Note: API-based identifiers like 'Gemini' are not handled here.

    Returns:
        BaseEmbedder: An instance of the appropriate embedder subclass.

    Raises:
        ValueError: If the embedding_model_name is not supported or if the device
                    is incompatible (e.g., trying to load local model on 'Gemini').
        ImportError: If required dependencies for the selected model are missing.
    """
    if device not in ["cpu", "cuda", "rocm"]:
        # This factory currently only handles local models
        raise ValueError(f"Device '{device}' is not supported for local embedder creation. Use 'cpu', 'cuda', or 'rocm'.")

    config = EMBEDDER_REGISTRY.get(embedding_model_name.lower())

    if not config:
        # Maybe check if it looks like a path/HF model name for SentenceTransformer?
        # For now, strict registry check.
        raise ValueError(f"Unsupported embedding model name: '{embedding_model_name}'. "
                         f"Supported: {list(EMBEDDER_REGISTRY.keys())}")

    EmbedderClass = config["class"]
    model_name_arg = config["default_model_name"] # Use the specific model name associated with the key

    print(f"[INFO] Creating embedder: {EmbedderClass.__name__} with model '{model_name_arg}' on device '{device}'")
    try:
        # Pass the specific model name, device, and cache_manager to the constructor
        if EmbedderClass == ColPaliEmbedder:
            # ColPaliEmbedder uses 'model_hf_name'
            return EmbedderClass(model_hf_name=model_name_arg, device=device, cache_manager=cache_manager)
        elif EmbedderClass == SentenceTransformerEmbedder:
            # SentenceTransformerEmbedder uses 'model_st_name'
            return EmbedderClass(model_st_name=model_name_arg, device=device, cache_manager=cache_manager)
        else:
            # Fallback for potentially other registered classes
            # This assumes they take 'model_name', 'device', and 'cache_manager' args, adjust if needed
            # If a class doesn't accept cache_manager, this might fail. Consider adding checks or specific handling.
            return EmbedderClass(model_name=model_name_arg, device=device, cache_manager=cache_manager)
    except ImportError as ie:
        print(f"[ERROR] Missing dependency for {embedding_model_name}: {ie}")
        raise # Re-raise import error
    except Exception as e:
        print(f"[ERROR] Failed to instantiate embedder '{embedding_model_name}': {e}")
        # Wrap other exceptions in a ValueError or similar for consistent handling
        raise ValueError(f"Failed to create embedder '{embedding_model_name}': {e}") from e
