# knowledge_base.py

# knowledge_base.py

import os
import torch
import numpy as np
import fitz  # PyMuPDF
# PIL is only needed for OCR, which is optional
# from PIL import Image

# Removed API embedding function imports
# from llm_providers.gemini import call_gemini_embedding
# from llm_providers.openrouter import call_openrouter_embedding

############################
# Load & Configure Retrieval
############################
# Note: embedding_model_name is the specific model ID (e.g., 'all-minilm', 'models/embedding-001')
# device now includes 'Gemini', 'OpenRouter' in addition to 'cpu', 'cuda'
def load_retrieval_model(embedding_model_name: str, device: str):
    """
    Loads a local retrieval model or prepares for API usage based on device type.

    Args:
        embedding_model_name (str): The name of the embedding model
                                   (e.g., 'colpali', 'all-minilm', 'models/embedding-001', 'openai/text-embedding-ada-002').
        device (str): The device to use ('cpu', 'cuda', 'Gemini', 'OpenRouter').

    Returns:
        tuple: (model, processor, model_type)
               - model: Loaded model object (local) or None (API).
               - processor: Loaded processor object (local) or None (API).
                - model_type: String identifier ('colpali', 'all-minilm').
    """
    # Removed API cases ("Gemini", "OpenRouter")

    # Handle local model cases (device is 'cpu' or 'cuda')
    if embedding_model_name == "colpali":
        try:
            from transformers import ColPaliForRetrieval, ColPaliProcessor
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        model_hf_name = "vidore/colpali-v1.2-hf"
        print(f"[INFO] Loading local ColPali model ({model_hf_name}) onto device: {device}")
        model = ColPaliForRetrieval.from_pretrained(
            model_hf_name,
            torch_dtype=torch.bfloat16, # Consider making dtype configurable
            device_map=device
        ).eval()
        processor = ColPaliProcessor.from_pretrained(model_hf_name)
        return model, processor, "colpali"
    elif embedding_model_name == "all-minilm":
        # Load local Sentence Transformer model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        model_st_name = "all-MiniLM-L6-v2" # Or make this match embedding_model_name if more ST models added
        print(f"[INFO] Loading local Sentence Transformer model ({model_st_name}) onto device: {device}")
        model = SentenceTransformer(model_st_name, device=device)
        return model, None, "all-minilm"
    else:
        # Fallback or error for unknown local models when device is cpu/cuda
        raise ValueError(f"Unsupported local retrieval model choice '{embedding_model_name}' for device '{device}'")


# Removed gemini_api_key, openrouter_api_key parameters
def embed_text(text: str, model, processor, model_type: str, embedding_model_name: str, device: str) -> torch.Tensor | None:
    """
    Generates embeddings for the given text using a local model.

    Args:
        text (str): The text to embed.
        model: The loaded local model.
        processor: The loaded local processor (if applicable).
        model_type (str): Identifier ('colpali', 'all-minilm').
        embedding_model_name (str): The specific name of the local model.
        device (str): The target device ('cpu', 'cuda').

    Returns:
        torch.Tensor | None: The embedding tensor, or None if embedding failed.
    """
    try:
        if model_type == "colpali":
            if model is None or processor is None:
                 raise ValueError("ColPali model/processor not loaded for local embedding.")
            inputs = processor(text=[text], truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad(): # Ensure no gradients are computed
                 outputs = model(**inputs)
            # For simplicity, we take the mean over all tokens.
            embedding = outputs.embeddings.mean(dim=1).squeeze(0)
            return embedding.cpu() # Return on CPU
        elif model_type == "all-minilm":
            if model is None:
                 raise ValueError("Sentence Transformer model not loaded for local embedding.")
            # Ensure text is not empty
            if not text.strip():
                 print("[WARN] Attempted to embed empty text with all-minilm.")
                 return None
            # SentenceTransformer returns numpy array by default, convert_to_tensor=True gives torch tensor
            embedding = model.encode(text, convert_to_tensor=True)
            return embedding.cpu() # Return on CPU
        # Removed 'gemini-api' and 'openrouter-api' cases
        else:
            raise ValueError(f"Unsupported local model_type for embedding: {model_type}")
    except Exception as e:
        # Log the error and the text snippet that caused it for debugging
        snippet = text[:100].replace('\n', ' ') + "..."
        # Explicitly convert exception to string for cleaner logging
        error_message = str(e)
        print(f"[ERROR] Failed to embed text (snippet: '{snippet}...') using {model_type}/{embedding_model_name}: {error_message}")
        # Optionally re-raise or return None based on desired error handling
        return None


##################
# Scoring & Search
##################
def late_interaction_score(query_emb, doc_emb):
    # Ensure tensors are on the same device (CPU) and are float32 for dot product
    q_vec = query_emb.cpu().float().view(-1)
    d_vec = doc_emb.cpu().float().view(-1)

    # Handle potential zero vectors
    q_norm_val = torch.linalg.norm(q_vec)
    d_norm_val = torch.linalg.norm(d_vec)

    if q_norm_val == 0 or d_norm_val == 0:
        return 0.0 # Or handle as an error case

    q_norm = q_vec / q_norm_val
    d_norm = d_vec / d_norm_val
    # Cosine similarity is the dot product of normalized vectors
    similarity = torch.dot(q_norm, d_norm)
    # Clamp values to avoid potential floating point issues outside [-1, 1]
    return float(torch.clamp(similarity, -1.0, 1.0))

# Removed gemini_api_key, openrouter_api_key parameters
def retrieve(query: str, corpus: list, model, processor, top_k: int, model_type: str, embedding_model_name: str, device: str) -> list:
    """Retrieves the top_k most relevant documents from the corpus for the query."""
    # Removed API keys from embed_text call
    query_embedding = embed_text(
        query, model, processor, model_type, embedding_model_name, device
    )

    if query_embedding is None:
        print("[ERROR] Could not generate query embedding. Retrieval failed.")
        return []

    scores = []
    valid_corpus_indices = []
    for i, entry in enumerate(corpus):
        if 'embedding' not in entry or entry['embedding'] is None:
             print(f"[WARN] Skipping corpus entry {i} due to missing or invalid embedding.")
             continue
        # Ensure document embedding is a tensor
        doc_embedding = entry['embedding']
        if not isinstance(doc_embedding, torch.Tensor):
             print(f"[WARN] Skipping corpus entry {i} because embedding is not a tensor (type: {type(doc_embedding)}).")
             continue

        score = late_interaction_score(query_embedding, doc_embedding)
        scores.append(score)
        valid_corpus_indices.append(i) # Keep track of indices corresponding to scores

    if not scores:
        print("[WARN] No valid documents found in corpus to score against.")
        return []

    # Use argsort on the scores, then map back to original corpus indices
    sorted_score_indices = np.argsort(scores)[::-1] # Indices of scores array, descending
    top_valid_indices = sorted_score_indices[:top_k] # Top k indices within the scores array

    # Map these back to the original corpus indices
    top_corpus_indices = [valid_corpus_indices[i] for i in top_valid_indices]

    return [corpus[i] for i in top_corpus_indices]


##################################
# Building a Corpus from a Folder
##################################
# Removed gemini_api_key, openrouter_api_key parameters
def load_corpus_from_dir(corpus_dir: str, model, processor, device: str, model_type: str, embedding_model_name: str, progress_callback=None) -> list:
    """
    Scan 'corpus_dir' for txt, pdf files, embed their text using the specified local model,
    and return a list of { 'embedding':..., 'metadata':... } entries.
    Calls progress_callback with status updates if provided.
    """
    corpus = []
    if not corpus_dir or not os.path.isdir(corpus_dir):
        if corpus_dir: # Only warn if a dir was actually provided
             print(f"[WARN] Corpus directory not found or invalid: {corpus_dir}")
        return corpus

    _progress = progress_callback or (lambda msg: None) # Use dummy if None

    supported_extensions = (".txt", ".pdf") # Removed image OCR for simplicity now
    all_files = [
        f for f in os.listdir(corpus_dir)
        if os.path.isfile(os.path.join(corpus_dir, f)) and f.lower().endswith(supported_extensions)
    ]
    total_files = len(all_files)
    _progress(f"Found {total_files} supported files in {corpus_dir}. Starting embedding...")

    for i, filename in enumerate(all_files):
        file_path = os.path.join(corpus_dir, filename)
        _progress(f"Processing file {i+1}/{total_files}: {filename}")

        text = ""
        try:
            if filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                    text = f.read()
            elif filename.lower().endswith(".pdf"):
                doc = fitz.open(file_path)
                extracted_text = ""
                for page_num, page in enumerate(doc):
                    extracted_text += page.get_text("text") + "\n" # Use "text" for better extraction
                    # Optional: Add progress per page?
                doc.close()
                text = extracted_text
        except Exception as e:
            print(f"[WARN] Failed to read/extract text from {file_path}: {e}")
            continue # Skip this file

        if not text or not text.strip():
            print(f"[INFO] Skipping empty or unreadable file: {filename}")
            continue

        # --- Embedding Step ---
        # Use the unified embed_text function (API keys removed)
        emb = embed_text(
            text, model, processor, model_type, embedding_model_name, device
        )

        if emb is None:
            print(f"[WARN] Failed to generate embedding for {filename}. Skipping.")
            continue # Skip if embedding failed

        # --- Add to Corpus ---
        snippet = text[:150].replace('\n', ' ').strip() + "..."
        corpus.append({
            "embedding": emb.cpu(), # Ensure embedding is on CPU before storing
            "metadata": {
                "file_path": file_path,
                "type": "local",
                "snippet": snippet
            }
        })
        # Optional: Add a small delay for API calls to avoid rate limits?
        # if model_type in ["gemini-api", "openrouter-api"]:
        #     time.sleep(0.1) # Adjust as needed

    _progress(f"Finished processing {len(corpus)} documents from {corpus_dir}.")
    return corpus



###########################
# KnowledgeBase Class (API)
###########################
class KnowledgeBase:
    """
    Manages the corpus of embeddings and provides search functionality.
    Handles both local models and API-based embedding generation.
    """
    def __init__(self, device: str, embedding_model_name: str, progress_callback=None):
        """
        Initializes the KnowledgeBase, loading models or preparing for API use.

        Args:
            device (str): The device ('cpu', 'cuda', 'Gemini', 'OpenRouter').
            embedding_model_name (str): The name of the embedding model to use.
            progress_callback (callable, optional): Function for status updates.
        """
        self.device = device
        self.embedding_model_name = embedding_model_name
        self._progress = progress_callback or (lambda msg: None)
        self.corpus = []

        self._progress(f"Initializing KnowledgeBase with device='{device}', model='{embedding_model_name}'")
        # Load local model/processor only if needed
        self.model, self.processor, self.model_type = load_retrieval_model(
            embedding_model_name=self.embedding_model_name,
            device=self.device
        )
        self._progress(f"KnowledgeBase initialized with model_type='{self.model_type}'")
        # Removed API key storage
        # self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        # self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


    # Removed API key parameters
    def build_from_directory(self, corpus_dir: str):
        """Loads and embeds documents from a directory."""
        if not corpus_dir:
             self._progress("No corpus directory provided, skipping local corpus build.")
             self.corpus = []
             return

        self._progress(f"Building knowledge base from directory: {corpus_dir}")
        self.corpus = load_corpus_from_dir(
            corpus_dir=corpus_dir,
            model=self.model,
            processor=self.processor,
            device=self.device,
            model_type=self.model_type,
            embedding_model_name=self.embedding_model_name,
            progress_callback=self._progress
            # Removed API keys
        )
        self._progress(f"Knowledge base built with {len(self.corpus)} documents.")

    def add_documents(self, entries: list):
        """
        Adds pre-computed document entries to the corpus.
        Ensures embeddings are torch tensors on CPU.

        Args:
            entries (list): List of dicts {'embedding': ..., 'metadata': ...}
        """
        count = 0
        for entry in entries:
             if 'embedding' in entry and entry['embedding'] is not None:
                 if isinstance(entry['embedding'], torch.Tensor):
                     entry['embedding'] = entry['embedding'].cpu() # Ensure CPU
                 else:
                     # Attempt conversion if it's list/numpy array, otherwise warn
                     try:
                         entry['embedding'] = torch.tensor(entry['embedding'], dtype=torch.float32).cpu()
                     except Exception as e:
                         print(f"[WARN] Could not convert provided embedding to tensor: {e}. Skipping entry.")
                         continue
                 self.corpus.append(entry)
                 count += 1
             else:
                 print("[WARN] Skipping document entry with missing or None embedding.")
        self._progress(f"Added {count} pre-computed document entries.")


    # Removed API key parameters
    def search(self, query: str, top_k: int = 3) -> list:
        """Searches the corpus for the most relevant documents."""
        self._progress(f"Searching knowledge base for query: '{query[:50]}...'")
        if not self.corpus:
            self._progress("Knowledge base is empty. Cannot perform search.")
            return []

        results = retrieve(
            query=query,
            corpus=self.corpus,
            model=self.model,
            processor=self.processor,
            top_k=top_k,
            model_type=self.model_type,
            embedding_model_name=self.embedding_model_name, # Pass the name
            device=self.device
            # Removed API keys
        )
        self._progress(f"Retrieval found {len(results)} results.")
        return results
