# search_session.py

import os
import uuid
import asyncio
import random
import traceback # For more detailed error logging

# Import necessary components
from knowledge_base import KnowledgeBase # embed_text is removed, late_interaction_score unused here
from web_search import download_webpages_ddg, parse_html_to_text, group_web_results_by_domain, sanitize_filename
from aggregator import aggregate_results # Keep for save_report call initially
# Import specific functions from LLM providers and utils
from llm_providers.tasks import chain_of_thought_query_enhancement
from llm_providers.utils import clean_search_query
import toc_tree
# Import new modules/factories
from embeddings.factory import create_embedder # Import the embedder factory
from search_logic import web_recursive, subquery as subquery_logic, summarization, reporting

#########################################################
# The "SearchSession" class: orchestrate the entire pipeline,
# including optional Monte Carlo subquery sampling, recursive web search,
# TOC tracking, and relevance scoring.
#########################################################

class SearchSession:
    # Modified __init__ to accept resolved_settings dictionary
    def __init__(self, query, config, resolved_settings, progress_callback=None):
        """
        Initializes the SearchSession.

        :param query: The initial user query.
        :param config: The raw configuration dictionary (from YAML).
        :param resolved_settings: A dictionary containing the final settings after considering CLI args, config file, and defaults.
        :param progress_callback: Optional function to call with status updates.
        """
        self.query = query
        self.config = config # Keep raw config if needed elsewhere (e.g., save_report)
        self.resolved_settings = resolved_settings # Store the resolved settings
        self.progress_callback = progress_callback or (lambda msg: None)

        # --- Use resolved_settings for initialization ---
        self.query_id = str(uuid.uuid4())[:8]
        # Use resolved_settings for base directory, falling back to default 'results'
        # Note: Assuming 'results_base_dir' might be in the raw 'config' under an 'advanced' section or similar
        results_base_dir = self.config.get('advanced', {}).get("results_base_dir", "results")
        self.base_result_dir = os.path.join(results_base_dir, self.query_id)
        os.makedirs(self.base_result_dir, exist_ok=True)

        self.progress_callback(f"Initializing SearchSession (ID: {self.query_id})...")
        print(f"[INFO] Initializing SearchSession for query_id={self.query_id}")

        # Enhance the query via chain-of-thought using resolved settings
        self.progress_callback("Enhancing query using chain-of-thought...")
        # Assemble llm_config for the enhancement task
        provider = self.resolved_settings.get('rag_model', 'gemma')
        model_id = self.resolved_settings.get(f"{provider}_model_id") # e.g., gemini_model_id
        api_key = self.resolved_settings.get(f"{provider}_api_key") # e.g., gemini_api_key
        llm_config_for_enhancement = {
            "provider": provider,
            "model_id": model_id,
            "api_key": api_key,
            "personality": self.resolved_settings.get('personality')
        }
        self.enhanced_query = chain_of_thought_query_enhancement(
            self.query,
            llm_config=llm_config_for_enhancement
        )
        self.progress_callback(f"Enhanced Query: {self.enhanced_query}")

        # --- Initialize Embedder ---
        try:
            self.progress_callback(f"Creating embedder for model='{self.resolved_settings['embedding_model']}' on device='{self.resolved_settings['device']}'...")
            self.embedder = create_embedder(
                embedding_model_name=self.resolved_settings['embedding_model'],
                device=self.resolved_settings['device']
            )
        except (ValueError, ImportError) as e:
            self.progress_callback(f"[CRITICAL] Failed to create embedder: {e}. Aborting.")
            print(f"[ERROR] Failed to create embedder: {e}\n{traceback.format_exc()}")
            raise # Re-raise the critical error

        # --- Initialize KnowledgeBase with the embedder ---
        self.progress_callback(f"Initializing KnowledgeBase...")
        self.kb = KnowledgeBase(
            embedder=self.embedder,
            progress_callback=self.progress_callback
        )
        # Removed self.model_type, self.model, self.processor storage

        # Compute the overall enhanced query embedding using the embedder
        self.progress_callback("Computing embedding for enhanced query...")
        print("[INFO] Computing embedding for enhanced query...")
        try:
            self.enhanced_query_embedding = self.embedder.embed(self.enhanced_query)
            if self.enhanced_query_embedding is None:
                # Handle embedding failure
                self.progress_callback("[CRITICAL] Failed to compute initial query embedding (returned None). Aborting.")
                raise ValueError("Failed to compute initial query embedding.")
        except Exception as e:
             self.progress_callback(f"[CRITICAL] Exception during initial query embedding: {e}. Aborting.")
             print(f"[ERROR] Exception during initial query embedding: {e}\n{traceback.format_exc()}")
             raise ValueError(f"Failed to compute initial query embedding: {e}") from e


        # Build local corpus if directory is provided using resolved setting
        if self.resolved_settings['corpus_dir']:
            # KnowledgeBase now uses its embedder internally, no need to pass keys/models
            self.kb.build_from_directory(self.resolved_settings['corpus_dir'])
        else:
            self.progress_callback("No local corpus directory specified.")

        # Placeholders for web search results and TOC tree.
        self.web_results = []
        self.grouped_web_results = {}
        self.local_results = []
        self.toc_tree = []  # List of toc_tree.TOCNode objects for the initial subqueries
        # Store reference links found during summarization
        self._reference_links = []

    async def run_session(self):
        """
        Main entry point: perform recursive web search (if enabled) and then local retrieval.
        """
        # Use resolved max_depth
        self.progress_callback(f"Starting search session (Max Depth: {self.resolved_settings['max_depth']})...")
        print(f"[INFO] Starting session with query_id={self.query_id}, max_depth={self.resolved_settings['max_depth']}")
        plain_enhanced_query = clean_search_query(self.enhanced_query) # Use clean_search_query from utils

        # 1) Generate subqueries from the enhanced query
        self.progress_callback("Generating initial subqueries...")
        initial_subqueries = subquery_logic.generate_initial_subqueries(self.enhanced_query, self.config)
        self.progress_callback(f"Generated {len(initial_subqueries)} initial subqueries.")

        # 2) Optionally do a Monte Carlo approach to sample subqueries
        if self.config.get('advanced', {}).get("monte_carlo_search", True):
            self.progress_callback("Performing Monte Carlo subquery sampling...")
            initial_subqueries = subquery_logic.perform_monte_carlo_subqueries(
                parent_query=plain_enhanced_query,
                subqueries=initial_subqueries,
                config=self.config,
                resolved_settings=self.resolved_settings,
                enhanced_query_embedding=self.enhanced_query_embedding,
                progress_callback=self.progress_callback,
                embedder=self.embedder # Pass embedder instance
                # Removed model, processor, model_type
            )
            self.progress_callback(f"Selected {len(initial_subqueries)} subqueries via Monte Carlo.")

        # 3) If web search is enabled and max_depth >= 1, do the recursive expansion
        if self.resolved_settings['web_search'] and self.resolved_settings['max_depth'] >= 1:
            self.progress_callback("Starting recursive web search...")
            web_results, web_entries, grouped, toc_nodes = await web_recursive.perform_recursive_web_searches(
                subqueries=initial_subqueries,
                current_depth=1,
                max_depth=self.resolved_settings['max_depth'],
                base_result_dir=self.base_result_dir,
                enhanced_query_embedding=self.enhanced_query_embedding,
                resolved_settings=self.resolved_settings,
                config=self.config,
                progress_callback=self.progress_callback,
                embedder=self.embedder # Pass embedder instance
                # Removed model, processor, model_type
            )
            self.web_results = web_results
            self.grouped_web_results = grouped
            self.toc_tree = toc_nodes
            # Add new entries to the knowledge base
            self.progress_callback(f"Adding {len(web_entries)} web entries to knowledge base...")
            self.kb.add_documents(web_entries) # KB handles API keys internally now
            self.progress_callback("Finished recursive web search.")
        else:
            self.progress_callback("Web search disabled or max_depth < 1, skipping.")
            print("[INFO] Web search is disabled or max_depth < 1, skipping web expansion.")

        # 4) Local retrieval (uses the KB's search method)
        # Use resolved top_k
        self.progress_callback(f"Retrieving top {self.resolved_settings['top_k']} documents from knowledge base...")
        print(f"[INFO] Retrieving top {self.resolved_settings['top_k']} documents for final answer.")
        # KB.search now uses its internal embedder, no need for keys/models
        self.local_results = self.kb.search(
            self.enhanced_query,
            top_k=self.resolved_settings['top_k']
        )
        self.progress_callback(f"Retrieved {len(self.local_results)} documents from knowledge base.")

        # 5) Summaries and final RAG generation
        self.progress_callback("Summarizing web results...")
        # Summarization function now returns summary and reference links
        summarized_web, self._reference_links = summarization.summarize_web_results(
            self.web_results, self.config, self.resolved_settings, self.progress_callback
        )
        self.progress_callback("Summarizing local results...")
        summarized_local = summarization.summarize_local_results(
            self.local_results, self.config, self.resolved_settings, self.progress_callback
        )
        self.progress_callback("Building final report using RAG...")
        final_answer = reporting.build_final_answer(
            enhanced_query=self.enhanced_query,
            toc_tree_nodes=self.toc_tree,
            summarized_web=summarized_web,
            summarized_local=summarized_local,
            reference_links=self._reference_links, # Pass stored links
            resolved_settings=self.resolved_settings,
            progress_callback=self.progress_callback
            # Pass previous_results_content, follow_up_convo if needed
        )
        self.progress_callback("Finished building final report.")
        print("[INFO] Finished building final advanced report.")
        return final_answer

    # --- Removed perform_monte_carlo_subqueries ---

    # --- Removed perform_recursive_web_searches ---

    # --- Removed _summarize_web_results ---

    # --- Removed _summarize_local_results ---

    # --- Removed _build_final_answer ---

    def save_report(self, final_answer, previous_results=None, follow_up_convo=None):
        """Saves the report using the reporting module."""
        # Call the function from the reporting module
        output_path = reporting.save_report(
            query_id=self.query_id,
            enhanced_query=self.enhanced_query,
            web_results=self.web_results,
            local_results=self.local_results,
            final_answer=final_answer,
            config=self.config,
            grouped_web_results=self.grouped_web_results,
            progress_callback=self.progress_callback,
            previous_results=previous_results,
            follow_up_convo=follow_up_convo
        )
        return output_path
