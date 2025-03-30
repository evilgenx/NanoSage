# embeddings/local.py
import torch
from .base import BaseEmbedder

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

    def __init__(self, model_hf_name="vidore/colpali-v1.2-hf", device="cpu"):
        if ColPaliForRetrieval is None or ColPaliProcessor is None:
            raise ImportError("ColPali dependencies not found. Please install transformers: pip install transformers")

        self.device = device
        self.model_hf_name = model_hf_name
        print(f"[INFO] Loading local ColPali model ({self.model_hf_name}) onto device: {self.device}")
        try:
            # Consider making dtype configurable if needed
            self.model = ColPaliForRetrieval.from_pretrained(
                self.model_hf_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(self.model_hf_name)
        except Exception as e:
            print(f"[ERROR] Failed to load ColPali model '{self.model_hf_name}': {e}")
            raise

    def embed(self, text: str) -> torch.Tensor | None:
        """Generates embeddings using the loaded ColPali model."""
        if not text or not text.strip():
            print("[WARN] Attempted to embed empty text with ColPali.")
            return None
        try:
            inputs = self.processor(text=[text], truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling
            embedding = outputs.embeddings.mean(dim=1).squeeze(0)
            return embedding.cpu() # Return on CPU
        except Exception as e:
            snippet = text[:100].replace('\n', ' ') + "..."
            error_message = str(e)
            print(f"[ERROR] Failed to embed text (snippet: '{snippet}...') using ColPali: {error_message}")
            return None


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using a Sentence Transformer model."""

    def __init__(self, model_st_name="all-MiniLM-L6-v2", device="cpu"):
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformer not found. Please install sentence-transformers: pip install sentence-transformers")

        self.device = device
        self.model_st_name = model_st_name
        print(f"[INFO] Loading local Sentence Transformer model ({self.model_st_name}) onto device: {self.device}")
        try:
            self.model = SentenceTransformer(self.model_st_name, device=self.device)
        except Exception as e:
            print(f"[ERROR] Failed to load Sentence Transformer model '{self.model_st_name}': {e}")
            raise

    def embed(self, text: str) -> torch.Tensor | None:
        """Generates embeddings using the loaded Sentence Transformer model."""
        if not text or not text.strip():
            print("[WARN] Attempted to embed empty text with SentenceTransformer.")
            return None
        try:
            # convert_to_tensor=True gives torch tensor directly
            embedding = self.model.encode(text, convert_to_tensor=True)
            return embedding.cpu() # Return on CPU
        except Exception as e:
            snippet = text[:100].replace('\n', ' ') + "..."
            error_message = str(e)
            print(f"[ERROR] Failed to embed text (snippet: '{snippet}...') using SentenceTransformer: {error_message}")
            return None
