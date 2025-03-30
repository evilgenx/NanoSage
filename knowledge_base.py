# knowledge_base.py

# knowledge_base.py

import os
import torch
import numpy as np

# New imports for refactored structure
from embeddings.base import BaseEmbedder
import document_parsers.factory as parser_factory


##################
# Scoring (Keep this utility function here or move to a general utils module)
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


###########################
# KnowledgeBase Class
###########################
class KnowledgeBase:
    """
    Manages a corpus of document embeddings and provides search functionality.
    Relies on an injected Embedder instance for embedding generation.
    """
    def __init__(self, embedder: BaseEmbedder, progress_callback=None):
        """
        Initializes the KnowledgeBase with a specific embedder.

        Args:
            embedder (BaseEmbedder): An instance of an embedder (e.g., ColPaliEmbedder).
            progress_callback (callable, optional): Function for status updates.
        """
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError("embedder must be an instance of BaseEmbedder")

        self.embedder = embedder
        self._progress = progress_callback or (lambda msg: None)
        self.corpus = [] # List of {'embedding': torch.Tensor, 'metadata': dict}

        self._progress(f"KnowledgeBase initialized with embedder: {type(embedder).__name__}")

    def build_from_directory(self, corpus_dir: str):
        """
        Loads, parses, and embeds documents from a directory using the configured embedder.
        """
        if not corpus_dir or not os.path.isdir(corpus_dir):
            if corpus_dir: # Only warn if a dir was actually provided
                self._progress(f"[WARN] Corpus directory not found or invalid: {corpus_dir}. Skipping build.")
            self.corpus = []
            return

        self._progress(f"Building knowledge base from directory: {corpus_dir}")
        new_corpus = []
        all_files = [
            os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir)
            if os.path.isfile(os.path.join(corpus_dir, f))
        ]
        total_files = len(all_files)
        self._progress(f"Found {total_files} files in {corpus_dir}. Starting processing...")

        processed_count = 0
        for i, file_path in enumerate(all_files):
            filename = os.path.basename(file_path)
            self._progress(f"Processing file {i+1}/{total_files}: {filename}")

            parser = parser_factory.get_parser(file_path)
            if not parser:
                # Warning already printed by factory
                continue # Skip unsupported file types

            try:
                text = parser.parse(file_path)
            except Exception as e:
                self._progress(f"[WARN] Failed to parse {filename}: {e}. Skipping.")
                continue

            if not text or not text.strip():
                self._progress(f"[INFO] Skipping empty or unreadable file: {filename}")
                continue

            # --- Embedding Step ---
            emb = self.embedder.embed(text)

            if emb is None:
                self._progress(f"[WARN] Failed to generate embedding for {filename}. Skipping.")
                continue # Skip if embedding failed

            # --- Add to Corpus ---
            snippet = text[:150].replace('\n', ' ').strip() + "..."
            new_corpus.append({
                "embedding": emb.cpu(), # Ensure embedding is on CPU before storing
                "metadata": {
                    "file_path": file_path,
                    "type": "local", # Could add 'web' type later if needed
                    "snippet": snippet
                }
            })
            processed_count += 1

        self.corpus = new_corpus
        self._progress(f"Knowledge base built with {processed_count} documents from {corpus_dir}.")

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


    def search(self, query: str, top_k: int = 3) -> list:
        """
        Searches the corpus for the top_k most relevant documents to the query,
        using the configured embedder.
        """
        self._progress(f"Searching knowledge base for query: '{query[:50]}...'")
        if not self.corpus:
            self._progress("Knowledge base is empty. Cannot perform search.")
            return []

        query_embedding = self.embedder.embed(query)

        if query_embedding is None:
            self._progress("[ERROR] Could not generate query embedding. Search failed.")
            return []

        scores = []
        valid_corpus_indices = []
        for i, entry in enumerate(self.corpus):
            # Basic validation (already checked in add_documents/build)
            if 'embedding' not in entry or not isinstance(entry['embedding'], torch.Tensor):
                self._progress(f"[WARN] Skipping invalid corpus entry {i} during search.")
                continue

            doc_embedding = entry['embedding']
            try:
                score = late_interaction_score(query_embedding, doc_embedding)
                scores.append(score)
                valid_corpus_indices.append(i) # Keep track of indices corresponding to scores
            except Exception as e:
                 # Log error during scoring for a specific document
                 file_info = entry.get('metadata', {}).get('file_path', f'index {i}')
                 self._progress(f"[ERROR] Failed to score document '{file_info}': {e}")


        if not scores:
            self._progress("[WARN] No valid documents could be scored against the query.")
            return []

        # Use argsort on the scores, then map back to original corpus indices
        # Ensure scores is a numpy array for argsort
        sorted_score_indices = np.argsort(np.array(scores))[::-1] # Indices of scores array, descending
        top_valid_indices = sorted_score_indices[:top_k] # Top k indices within the scores array

        # Map these back to the original corpus indices
        top_corpus_indices = [valid_corpus_indices[i] for i in top_valid_indices]

        results = [self.corpus[i] for i in top_corpus_indices]
        self._progress(f"Search found {len(results)} results.")
        return results
