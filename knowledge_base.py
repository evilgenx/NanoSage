# knowledge_base.py

import os
import torch
import numpy as np
import logging
import hashlib # For generating consistent IDs
import chromadb # Import chromadb
from chromadb.utils import embedding_functions # For potential future use if not using own embedder

# New imports for refactored structure
from embeddings.base import BaseEmbedder
import document_parsers.factory as parser_factory
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

##################
# Scoring (Keep this utility function here or move to a general utils module)
# Note: ChromaDB handles distance calculation internally, so this might become less relevant
#       unless custom scoring logic is needed after retrieval.
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

def _generate_doc_id(identifier: str) -> str:
    """Generates a consistent ID for a document based on its identifier (path/URL)."""
    return hashlib.sha256(identifier.encode('utf-8')).hexdigest()

###########################
# KnowledgeBase Class (ChromaDB Implementation)
###########################
class KnowledgeBase:
    """
    Manages document embeddings using ChromaDB for persistent storage and retrieval.
    Relies on an injected Embedder instance for embedding generation.
    """
    def __init__(self, embedder: BaseEmbedder, db_path: str = "./chroma_db", collection_name: str = "nanosage_kb", progress_callback=None):
        """
        Initializes the KnowledgeBase with a specific embedder and ChromaDB settings.

        Args:
            embedder (BaseEmbedder): An instance of an embedder (e.g., ColPaliEmbedder).
            db_path (str): Path to the directory for ChromaDB persistence.
            collection_name (str): Name of the ChromaDB collection to use.
            progress_callback (callable, optional): Function for status updates.
        """
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError("embedder must be an instance of BaseEmbedder")

        self.embedder = embedder
        self._progress = progress_callback or (lambda msg: None)
        self.db_path = db_path
        self.collection_name = collection_name

        self._progress(f"Initializing ChromaDB client at path: {self.db_path}")
        try:
            # Initialize ChromaDB client for persistent storage
            self.client = chromadb.PersistentClient(path=self.db_path)
            # Get or create the collection
            # Note: We might need to specify the embedding function dimensions or metadata if not using Chroma's defaults
            # For now, assume the collection can handle the embeddings generated by our embedder.
            # Consider adding distance function setting (e.g., cosine) if needed: metadata={"hnsw:space": "cosine"}
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # metadata={"hnsw:space": "cosine"} # Example: Specify distance metric if needed
            )
            self._progress(f"Connected to ChromaDB collection: '{self.collection_name}'")
        except Exception as e:
            self._progress(f"[ERROR] Failed to initialize ChromaDB: {e}")
            logger.error("ChromaDB Initialization Failed", exc_info=True)
            raise # Re-raise the exception

        self._progress(f"KnowledgeBase initialized with embedder: {type(embedder).__name__} and ChromaDB collection: '{self.collection_name}'")

    def build_from_directory(self, corpus_dir: str):
        """
        Loads, parses, embeds, and adds documents from a directory to the ChromaDB collection.
        Uses upsert to avoid duplicates based on file path.
        """
        if not corpus_dir or not os.path.isdir(corpus_dir):
            if corpus_dir: # Only warn if a dir was actually provided
                self._progress(f"[WARN] Corpus directory not found or invalid: {corpus_dir}. Skipping build.")
            return

        self._progress(f"Building knowledge base from directory: {corpus_dir}")
        all_files = [
            os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir)
            if os.path.isfile(os.path.join(corpus_dir, f))
        ]
        total_files = len(all_files)
        self._progress(f"Found {total_files} files in {corpus_dir}. Starting processing...")

        # Process files in batches for potentially better performance with ChromaDB add/upsert
        batch_size = 50 # Configurable?
        ids_batch = []
        embeddings_batch = []
        metadatas_batch = []
        documents_batch = [] # Chroma can optionally store the text itself
        processed_count = 0

        for i, file_path in enumerate(all_files):
            filename = os.path.basename(file_path)
            self._progress(f"Processing file {i+1}/{total_files}: {filename}")

            parser = parser_factory.get_parser(file_path)
            if not parser:
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

            # --- Prepare data for ChromaDB ---
            doc_id = _generate_doc_id(file_path) # Use file path to generate ID
            snippet = text[:200].replace('\n', ' ').strip() + "..." # Slightly longer snippet
            metadata = {
                "file_path": file_path,
                "type": "local",
                "snippet": snippet,
                "source": filename # Add filename as source for easier identification
            }

            ids_batch.append(doc_id)
            embeddings_batch.append(emb.cpu().numpy().tolist()) # Chroma expects list or numpy array
            metadatas_batch.append(metadata)
            documents_batch.append(text) # Store the full text

            # Add batch to ChromaDB if batch size is reached
            if len(ids_batch) >= batch_size:
                try:
                    self.collection.upsert(
                        ids=ids_batch,
                        embeddings=embeddings_batch,
                        metadatas=metadatas_batch,
                        documents=documents_batch
                    )
                    processed_count += len(ids_batch)
                    self._progress(f"Upserted batch of {len(ids_batch)} documents to ChromaDB.")
                    ids_batch, embeddings_batch, metadatas_batch, documents_batch = [], [], [], [] # Reset batch
                except Exception as e:
                    self._progress(f"[ERROR] Failed to upsert batch to ChromaDB: {e}")
                    logger.error("ChromaDB upsert failed for batch", exc_info=True)
                    # Decide whether to skip the batch or halt the process
                    ids_batch, embeddings_batch, metadatas_batch, documents_batch = [], [], [], [] # Reset batch anyway

        # Add any remaining documents in the last batch
        if ids_batch:
            try:
                self.collection.upsert(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    metadatas=metadatas_batch,
                    documents=documents_batch
                )
                processed_count += len(ids_batch)
                self._progress(f"Upserted final batch of {len(ids_batch)} documents to ChromaDB.")
            except Exception as e:
                self._progress(f"[ERROR] Failed to upsert final batch to ChromaDB: {e}")
                logger.error("ChromaDB upsert failed for final batch", exc_info=True)

        self._progress(f"Knowledge base build complete. Processed {processed_count} documents from {corpus_dir}.")

    def add_documents(self, entries: List[Dict[str, Any]]):
        """
        Adds pre-computed document entries to the ChromaDB collection.
        Assumes entries contain 'embedding' and 'metadata'.
        Metadata should ideally contain a unique identifier like 'file_path' or 'source_url'.
        Uses upsert based on generated IDs.
        """
        ids_batch = []
        embeddings_batch = []
        metadatas_batch = []
        documents_batch = [] # Assuming text might be in metadata['text'] or similar
        count_added = 0
        count_skipped = 0

        for entry in entries:
            metadata = entry.get('metadata', {})
            embedding = entry.get('embedding')

            # Determine identifier for ID generation
            identifier = metadata.get('file_path') or metadata.get('source_url') or metadata.get('source')
            if not identifier:
                logger.warning("Skipping entry due to missing identifier (file_path/source_url/source) in metadata.")
                count_skipped += 1
                continue

            if embedding is None:
                logger.warning(f"Skipping entry for '{identifier}' due to missing embedding.")
                count_skipped += 1
                continue

            # Convert embedding
            if isinstance(embedding, torch.Tensor):
                embedding_list = embedding.cpu().numpy().tolist()
            elif isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, list):
                embedding_list = embedding
            else:
                logger.warning(f"Skipping entry for '{identifier}' due to unsupported embedding type: {type(embedding)}")
                count_skipped += 1
                continue

            doc_id = _generate_doc_id(str(identifier)) # Ensure identifier is string
            # Optionally store the original text if available
            doc_text = metadata.get('text', metadata.get('snippet', '')) # Get text if available

            ids_batch.append(doc_id)
            embeddings_batch.append(embedding_list)
            metadatas_batch.append(metadata)
            documents_batch.append(doc_text)

        if ids_batch:
            try:
                self.collection.upsert(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    metadatas=metadatas_batch,
                    documents=documents_batch
                )
                count_added = len(ids_batch)
                self._progress(f"Upserted {count_added} pre-computed document entries to ChromaDB.")
            except Exception as e:
                self._progress(f"[ERROR] Failed to upsert pre-computed batch to ChromaDB: {e}")
                logger.error("ChromaDB upsert failed for pre-computed batch", exc_info=True)
                count_skipped += len(ids_batch) # Count them as skipped if upsert failed

        if count_skipped > 0:
             self._progress(f"[WARN] Skipped {count_skipped} pre-computed entries due to errors or missing data.")

    def add_scraped_content(self, url: str, markdown_content: str):
        """
        Embeds scraped Markdown content and adds it to the ChromaDB collection.
        Uses upsert based on URL.
        """
        if not markdown_content or not markdown_content.strip():
            self._progress(f"[WARN] Skipping empty scraped content from {url}.")
            return False

        self._progress(f"Embedding scraped content from: {url}")
        try:
            emb = self.embedder.embed(markdown_content)

            if emb is None:
                self._progress(f"[WARN] Failed to generate embedding for scraped content from {url}. Skipping.")
                return False

            # Create metadata
            snippet = markdown_content[:200].replace('\n', ' ').strip() + "..." # Slightly longer snippet
            metadata = {
                "source_url": url,
                "type": "web_scrape",
                "snippet": snippet,
                "source": url # Use URL as source identifier
            }
            doc_id = _generate_doc_id(url)

            # Add to ChromaDB using upsert
            self.collection.upsert(
                ids=[doc_id],
                embeddings=[emb.cpu().numpy().tolist()],
                metadatas=[metadata],
                documents=[markdown_content] # Store the full text
            )
            self._progress(f"Successfully upserted scraped content from {url} to knowledge base.")
            return True

        except Exception as e:
            self._progress(f"[ERROR] Failed to process scraped content from {url}: {e}")
            logger.error(f"Error processing scraped content from {url}:", exc_info=True)
            return False

    def search(self, query: str, top_k: int = 3, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Searches the ChromaDB collection for the top_k most relevant documents.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filter_metadata (Optional[Dict[str, Any]]): A dictionary for metadata filtering (e.g., {"type": "local"}).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the metadata of a relevant document.
                                  Returns empty list on failure or if no results found.
        """
        self._progress(f"Searching knowledge base for query: '{query[:50]}...'")
        try:
            # Check if collection is empty (optional, query might handle it)
            if self.collection.count() == 0:
                self._progress("Knowledge base collection is empty. Cannot perform search.")
                return []

            query_embedding = self.embedder.embed(query)

            if query_embedding is None:
                self._progress("[ERROR] Could not generate query embedding. Search failed.")
                return []

            # Perform the query
            # Note: ChromaDB expects query_embeddings as a list of lists or ndarray
            query_emb_list = [query_embedding.cpu().numpy().tolist()]

            results = self.collection.query(
                query_embeddings=query_emb_list,
                n_results=top_k,
                where=filter_metadata, # Add metadata filter if provided
                include=['metadatas', 'distances'] # Request metadata and distances
            )

            # Process results
            # Results is a dict containing lists for ids, distances, metadatas, etc.
            # Each list corresponds to a query embedding (we only have one)
            if not results or not results.get('ids') or not results['ids'][0]:
                self._progress("Search returned no results.")
                return []

            # Extract metadata and potentially distances for the first query
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0] # Lower distance is better (e.g., L2)

            # Format results as list of metadata dictionaries
            # Optionally add distance/score if needed downstream
            formatted_results = []
            for i, meta in enumerate(metadatas):
                result_item = {'metadata': meta}
                if i < len(distances):
                    # Add distance (lower is better) or convert to similarity score (higher is better)
                    # Cosine distance = 1 - cosine similarity. Similarity = 1 - distance.
                    similarity_score = 1.0 - distances[i]
                    result_item['score'] = similarity_score # Add similarity score
                    result_item['distance'] = distances[i] # Add raw distance
                formatted_results.append(result_item)

            # Sort by similarity score (descending) if available
            if formatted_results and 'score' in formatted_results[0]:
                 formatted_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)

            self._progress(f"Search found {len(formatted_results)} results.")
            return formatted_results

        except Exception as e:
            self._progress(f"[ERROR] Error during ChromaDB search: {e}")
            logger.error("ChromaDB search failed", exc_info=True)
            return []

    def get_collection_count(self) -> int:
        """Returns the number of items in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            self._progress(f"[ERROR] Failed to get collection count: {e}")
            logger.error("Failed to get ChromaDB collection count", exc_info=True)
            return 0

    def clear_collection(self):
        """Deletes and recreates the collection."""
        self._progress(f"Attempting to clear collection: '{self.collection_name}'")
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self._progress(f"Collection '{self.collection_name}' cleared and recreated.")
        except Exception as e:
            self._progress(f"[ERROR] Failed to clear ChromaDB collection: {e}")
            logger.error("Failed to clear ChromaDB collection", exc_info=True)
            # Attempt to recreate just in case deletion failed partially
            try:
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
            except Exception as e2:
                 self._progress(f"[ERROR] Failed to recreate ChromaDB collection after clear error: {e2}")
                 logger.error("Failed to recreate ChromaDB collection", exc_info=True)
                 # Raise the original error?
                 raise e
