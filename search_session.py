# search_session.py

import os
import uuid
import asyncio
import random
import traceback # For more detailed error logging
import time # For potential delays

# Import necessary components
from knowledge_base import KnowledgeBase # Removed Document import
from web_search import download_webpages_ddg, parse_html_to_text, group_web_results_by_domain, sanitize_filename
from aggregator import aggregate_results # Keep for save_report call initially
# Import specific functions from LLM providers and utils
from llm_providers.tasks import chain_of_thought_query_enhancement, generate_followup_queries # Added generate_followup_queries
from llm_providers.utils import clean_search_query
import toc_tree
# Import new modules/factories
from embeddings.factory import create_embedder # Import the embedder factory
# Import send_progress helper
from search_logic.web_recursive import send_progress
from search_logic import web_recursive, subquery as subquery_logic, summarization, reporting

#########################################################
# The "SearchSession" class: orchestrate the entire pipeline,
# including optional Monte Carlo subquery sampling, recursive web search,
# TOC tracking, and relevance scoring.
#########################################################

from typing import Optional # Added for type hinting
from cache_manager import CacheManager # Added import

class SearchSession:
    # Modified __init__ to accept resolved_settings dictionary and cache_manager
    def __init__(self, query, resolved_settings, config=None, progress_callback=None, cache_manager: Optional[CacheManager] = None): # Added cache_manager
        """
        Initializes the SearchSession.

        :param query: The initial user query.
        :param resolved_settings: A dictionary containing the final settings after considering CLI args, config file, and defaults.
        :param config: The raw configuration dictionary (from YAML). Optional.
        :param progress_callback: Optional function to call with status updates.
        :param cache_manager: Optional instance of CacheManager.
        """
        self.query = query
        self.config = config or {} # Store raw config if provided, else empty dict
        self.resolved_settings = resolved_settings # Store the resolved settings
        self.progress_callback = progress_callback or (lambda msg: None)
        self.cache_manager = cache_manager # Store the cache manager instance

        # --- Use resolved_settings for initialization ---
        self.query_id = str(uuid.uuid4())[:8]
        # Use resolved_settings for base directory, falling back to default 'results'
        # Note: Assuming 'results_base_dir' might be in the raw 'config' under an 'advanced' section or similar
        results_base_dir = self.config.get('advanced', {}).get("results_base_dir", "results")
        self.base_result_dir = os.path.join(results_base_dir, self.query_id)
        os.makedirs(self.base_result_dir, exist_ok=True)

        # Start Initialization Phase
        send_progress(self.progress_callback, "phase_start", {"phase": "initialization", "message": f"Initializing SearchSession (ID: {self.query_id})..."})
        print(f"[INFO] Initializing SearchSession for query_id={self.query_id}")

        # --- Query Enhancement Removed from __init__ ---
        # The query enhancement step is now handled in the GUI before starting the SearchWorker,
        # allowing for a preview without modifying the original query unless explicitly requested
        # via the "Extract Text (Keywords)" checkbox.
        # The SearchSession now receives the query it should use directly.
        self.enhanced_query = self.query # Use the provided query directly
        send_progress(self.progress_callback, "log", {"level": "info", "message": f"Using Query: {self.enhanced_query}"})
        # --- End Query Enhancement Removal ---


        # --- Initialize Embedder ---
        try:
            embed_msg = f"Creating embedder for model='{self.resolved_settings['embedding_model']}' on device='{self.resolved_settings['device']}'..."
            send_progress(self.progress_callback, "status", {"message": embed_msg})
            send_progress(self.progress_callback, "log", {"level": "info", "message": embed_msg})
            self.embedder = create_embedder(
                embedding_model_name=self.resolved_settings['embedding_model'],
                device=self.resolved_settings['device'],
                cache_manager=self.cache_manager # Pass cache_manager here
            )
        except (ValueError, ImportError) as e:
            err_msg = f"Failed to create embedder: {e}. Aborting."
            send_progress(self.progress_callback, "error", {"message": err_msg, "details": traceback.format_exc()})
            print(f"[ERROR] {err_msg}\n{traceback.format_exc()}")
            raise # Re-raise the critical error

        # --- Initialize KnowledgeBase with the embedder and DB settings ---
        kb_config = self.config.get('knowledge_base', {})
        kb_provider = kb_config.get('provider', 'chromadb') # Default to chromadb if not specified

        if kb_provider == 'chromadb':
            chroma_settings = kb_config.get('chromadb', {})
            db_path = chroma_settings.get('path', DEFAULT_CONFIG['knowledge_base']['chromadb']['path'])
            collection_name = chroma_settings.get('collection_name', DEFAULT_CONFIG['knowledge_base']['chromadb']['collection_name'])
            kb_msg = f"Initializing ChromaDB KnowledgeBase (Path: {db_path}, Collection: {collection_name})..."
            send_progress(self.progress_callback, "status", {"message": kb_msg})
            send_progress(self.progress_callback, "log", {"level": "info", "message": kb_msg})
            try:
                self.kb = KnowledgeBase(
                    embedder=self.embedder,
                    db_path=db_path,
                    collection_name=collection_name,
                    progress_callback=self.progress_callback
                )
            except Exception as e:
                 err_msg = f"Failed to initialize ChromaDB KnowledgeBase: {e}. Aborting."
                 send_progress(self.progress_callback, "error", {"message": err_msg, "details": traceback.format_exc()})
                 print(f"[ERROR] {err_msg}\n{traceback.format_exc()}")
                 raise # Re-raise critical error
        else:
            # Handle other providers or fallback to in-memory if needed
            # For now, raise an error if provider is not chromadb
            err_msg = f"Unsupported knowledge base provider specified in config: '{kb_provider}'. Aborting."
            send_progress(self.progress_callback, "error", {"message": err_msg})
            print(f"[ERROR] {err_msg}")
            raise ValueError(err_msg)
        # Removed self.model_type, self.model, self.processor storage

        # Compute the overall enhanced query embedding using the embedder
        query_embed_msg = "Computing embedding for enhanced query..."
        send_progress(self.progress_callback, "status", {"message": query_embed_msg})
        send_progress(self.progress_callback, "log", {"level": "info", "message": query_embed_msg})
        print("[INFO] Computing embedding for enhanced query...")
        try:
            self.enhanced_query_embedding = self.embedder.embed(self.enhanced_query)
            if self.enhanced_query_embedding is None:
                # Handle embedding failure
                err_msg = "Failed to compute initial query embedding (returned None). Aborting."
                send_progress(self.progress_callback, "error", {"message": err_msg})
                raise ValueError(err_msg)
        except Exception as e:
             err_msg = f"Exception during initial query embedding: {e}. Aborting."
             send_progress(self.progress_callback, "error", {"message": err_msg, "details": traceback.format_exc()})
             print(f"[ERROR] {err_msg}\n{traceback.format_exc()}")
             raise ValueError(err_msg) from e


        # Build local corpus if directory is provided using resolved setting
        if self.resolved_settings['corpus_dir']:
            # KnowledgeBase now uses its embedder internally, no need to pass keys/models
            # KB.build_from_directory should ideally use send_progress internally
            send_progress(self.progress_callback, "phase_start", {"phase": "local_corpus", "message": f"Building knowledge base from local directory: {self.resolved_settings['corpus_dir']}"})
            self.kb.build_from_directory(self.resolved_settings['corpus_dir'])
            send_progress(self.progress_callback, "phase_end", {"phase": "local_corpus", "message": "Finished building from local directory."})
        else:
            send_progress(self.progress_callback, "log", {"level": "info", "message": "No local corpus directory specified."})

        # End Initialization Phase
        send_progress(self.progress_callback, "phase_end", {"phase": "initialization", "message": "Initialization complete."})

        # Placeholders for web search results and TOC tree.
        self.web_results = []
        self.grouped_web_results = {}
        self.local_results = []
        self.toc_tree = []  # List of toc_tree.TOCNode objects for the initial subqueries
        # Store reference links found during summarization
        self._reference_links = []
        # Store include_visuals setting
        self.include_visuals = self.resolved_settings.get('include_visuals', False)

    async def run_session(self, cancellation_check_callback=None): # Added callback parameter
        """
        Main entry point: perform recursive web search (if enabled) and then local retrieval.
        """
        # Use resolved max_depth
        start_msg = f"Starting search session (Max Depth: {self.resolved_settings['max_depth']})..."
        send_progress(self.progress_callback, "status", {"message": start_msg})
        send_progress(self.progress_callback, "log", {"level": "info", "message": start_msg})
        print(f"[INFO] Starting session with query_id={self.query_id}, max_depth={self.resolved_settings['max_depth']}")
        plain_enhanced_query = clean_search_query(self.enhanced_query) # Use clean_search_query from utils

        # 1) Generate subqueries from the enhanced query
        send_progress(self.progress_callback, "phase_start", {"phase": "subquery_generation", "message": "Generating initial subqueries..."})
        initial_subqueries = subquery_logic.generate_initial_subqueries(self.enhanced_query, self.config)
        send_progress(self.progress_callback, "log", {"level": "info", "message": f"Generated {len(initial_subqueries)} initial subqueries."})
        send_progress(self.progress_callback, "phase_end", {"phase": "subquery_generation", "message": "Finished generating initial subqueries."})


        # 2) Optionally do a Monte Carlo approach to sample subqueries
        if self.config.get('advanced', {}).get("monte_carlo_search", True):
            send_progress(self.progress_callback, "phase_start", {"phase": "monte_carlo", "message": "Performing Monte Carlo subquery sampling..."})
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
            send_progress(self.progress_callback, "log", {"level": "info", "message": f"Selected {len(initial_subqueries)} subqueries via Monte Carlo."})
            send_progress(self.progress_callback, "phase_end", {"phase": "monte_carlo", "message": "Finished Monte Carlo sampling."})


        # 3) If web search is enabled and max_depth >= 1, do the recursive expansion
        if self.resolved_settings['web_search'] and self.resolved_settings['max_depth'] >= 1:
            send_progress(self.progress_callback, "phase_start", {"phase": "web_search", "message": "Starting recursive web search..."})
            web_results, web_entries, grouped, toc_nodes = await web_recursive.perform_recursive_web_searches(
                subqueries=initial_subqueries,
                current_depth=1,
                max_depth=self.resolved_settings['max_depth'],
                base_result_dir=self.base_result_dir,
                enhanced_query_embedding=self.enhanced_query_embedding,
                resolved_settings=self.resolved_settings,
                config=self.config,
                progress_callback=self.progress_callback,
                embedder=self.embedder, # Pass embedder instance
                cancellation_check_callback=cancellation_check_callback # Pass callback down
                # Removed model, processor, model_type
            )
            self.web_results = web_results
            self.grouped_web_results = grouped
            self.toc_tree = toc_nodes
            # Assign anchor IDs after the tree is built
            if self.toc_tree:
                send_progress(self.progress_callback, "log", {"level": "info", "message": "Assigning anchor IDs to TOC nodes..."})
                toc_tree.assign_anchor_ids(self.toc_tree)
            # Add new entries to the knowledge base
            # KB.add_documents should ideally use send_progress internally
            send_progress(self.progress_callback, "phase_start", {"phase": "kb_add_web", "message": f"Adding {len(web_entries)} web entries to knowledge base..."})
            self.kb.add_documents(web_entries) # KB handles API keys internally now
            send_progress(self.progress_callback, "phase_end", {"phase": "kb_add_web", "message": "Finished adding web entries."})
            send_progress(self.progress_callback, "phase_end", {"phase": "web_search", "message": "Finished recursive web search."})
        else:
            skip_msg = "Web search disabled or max_depth < 1, skipping."
            send_progress(self.progress_callback, "log", {"level": "info", "message": skip_msg})
            print(f"[INFO] {skip_msg}")

        # 4) Local retrieval (uses the KB's search method)
        # Use resolved top_k
        local_retrieval_msg = f"Retrieving top {self.resolved_settings['top_k']} documents from knowledge base..."
        send_progress(self.progress_callback, "phase_start", {"phase": "local_retrieval", "message": local_retrieval_msg})
        print(f"[INFO] Retrieving top {self.resolved_settings['top_k']} documents for final answer.")
        # KB.search now uses its internal embedder, no need for keys/models
        # KB.search should ideally use send_progress internally
        self.local_results = self.kb.search(
            self.enhanced_query,
            top_k=self.resolved_settings['top_k']
        )
        send_progress(self.progress_callback, "log", {"level": "info", "message": f"Retrieved {len(self.local_results)} initial documents from knowledge base."})
        send_progress(self.progress_callback, "phase_end", {"phase": "local_retrieval", "message": "Finished initial local retrieval."})


        # --- NEW: Iterative Sub-Query Generation (Agentic Step) ---
        if self.resolved_settings.get('enable_iterative_search', False) and self.resolved_settings.get('rag_model') != 'None':
            iterative_start_msg = "Iterative search enabled. Analyzing initial results..."
            send_progress(self.progress_callback, "phase_start", {"phase": "iterative_search", "message": iterative_start_msg})
            print("[INFO] Starting iterative sub-query generation step.")

            # a) Prepare Context Summary (Simple concatenation for now)
            # Combine text from top local results and web results
            context_texts = [res['metadata']['snippet'] for res in self.local_results[:self.resolved_settings['top_k']] if 'metadata' in res and 'snippet' in res['metadata']]
            # Add web result text (consider limiting length or number)
            for web_res in self.web_results[:5]: # Limit web results used for context
                 if 'text' in web_res and web_res['text']:
                     context_texts.append(web_res['text'][:1000]) # Limit length per web result

            context_summary_for_llm = "\n\n".join(context_texts).strip()

            if not context_summary_for_llm:
                warn_msg = "No context found from initial results to generate follow-up queries."
                send_progress(self.progress_callback, "log", {"level": "warning", "message": warn_msg})
            else:
                # b) Generate Follow-up Queries
                # Prepare llm_config for the generation task (using RAG model settings)
                llm_config_for_followup = {
                    "provider": self.resolved_settings.get('rag_model'),
                    "model_id": self.resolved_settings.get(f"{self.resolved_settings.get('rag_model')}_model_id"),
                    "api_key": self.resolved_settings.get(f"{self.resolved_settings.get('rag_model')}_api_key"),
                    "personality": self.resolved_settings.get('personality')
                }
                gen_followup_msg = "Generating follow-up queries using LLM..."
                send_progress(self.progress_callback, "status", {"message": gen_followup_msg})
                send_progress(self.progress_callback, "log", {"level": "info", "message": gen_followup_msg})
                followup_queries = generate_followup_queries(
                    initial_query=self.enhanced_query,
                    context_summary=context_summary_for_llm[:8000], # Limit context length for LLM
                    llm_config=llm_config_for_followup,
                    max_queries=3 # Limit to 3 follow-up queries
                )

                if followup_queries:
                    gen_msg = f"Generated {len(followup_queries)} follow-up queries: {', '.join(followup_queries)}"
                    send_progress(self.progress_callback, "log", {"level": "info", "message": gen_msg})
                    followup_docs_added = 0
                    # Start phase for follow-up searches
                    send_progress(self.progress_callback, "phase_start", {"phase": "iterative_followup", "message": f"Executing {len(followup_queries)} follow-up searches..."})
                    for fq_idx, f_query in enumerate(followup_queries):
                        # Check cancellation before each sub-query search
                        if cancellation_check_callback and cancellation_check_callback():
                            send_progress(self.progress_callback, "log", {"level": "info", "message": "Cancellation requested during follow-up search."})
                            raise asyncio.CancelledError("Search cancelled by user.")

                        # Report progress within follow-up searches
                        followup_search_msg = f"Executing follow-up search {fq_idx+1}/{len(followup_queries)}: '{f_query}'"
                        send_progress(self.progress_callback, "progress_update", {"phase": "iterative_followup", "current": fq_idx + 1, "total": len(followup_queries), "unit": "Queries", "message": followup_search_msg})
                        send_progress(self.progress_callback, "log", {"level": "info", "message": followup_search_msg})
                        try:
                            # c) Execute Follow-up Queries (Simple DDG search)
                            # Use a smaller limit for follow-up searches
                            followup_search_limit = max(1, self.resolved_settings.get('search_max_results', 5) // 2)
                            followup_results = await download_webpages_ddg(f_query, limit=followup_search_limit)
                            time.sleep(0.5) # Small delay between DDG calls

                            # d) Parse and Add to KB
                            for res in followup_results:
                                if cancellation_check_callback and cancellation_check_callback(): raise asyncio.CancelledError("Search cancelled.")
                                try:
                                     # Use 'file_path' key instead of 'body'
                                     parsed_text = parse_html_to_text(res['file_path'])
                                     if parsed_text:
                                         # Embed the parsed text using the session's embedder
                                         followup_emb = self.embedder.embed(parsed_text)
                                         if followup_emb is not None:
                                             # Create the dictionary entry expected by add_documents
                                             entry = {
                                                 'embedding': followup_emb.cpu(), # Ensure CPU tensor
                                                 'metadata': {
                                                     'source': res.get('href', 'unknown_followup_url'),
                                                     'query': f_query,
                                                     'type': 'followup_web',
                                                     'snippet': parsed_text[:150].replace('\n', ' ').strip() + "..."
                                                 }
                                             }
                                             # Add the single entry (as a list) to the KB
                                             # KB.add_documents should ideally use send_progress
                                             self.kb.add_documents([entry])
                                             followup_docs_added += 1
                                         else:
                                             warn_msg = f"Failed to embed follow-up content from: {res.get('href', 'unknown URL')}"
                                             send_progress(self.progress_callback, "log", {"level": "warning", "message": warn_msg})
                                         # Optional: Log added doc source
                                        # send_progress(self.progress_callback, "log", {"level": "debug", "message": f"  Added follow-up content from: {res['href']}"})
                                except Exception as parse_err:
                                    warn_msg = f"Failed to parse follow-up result from {res.get('href', 'unknown URL')}: {parse_err}"
                                    print(f"[WARN] {warn_msg}")
                                    send_progress(self.progress_callback, "log", {"level": "warning", "message": warn_msg})

                        except Exception as search_err:
                            err_msg = f"Failed to execute follow-up search for '{f_query}': {search_err}"
                            print(f"[ERROR] {err_msg}")
                            send_progress(self.progress_callback, "log", {"level": "error", "message": err_msg})

                    # End phase for follow-up searches
                    send_progress(self.progress_callback, "phase_end", {"phase": "iterative_followup", "message": "Finished follow-up searches."})

                    if followup_docs_added > 0:
                        add_msg = f"Added {followup_docs_added} documents from follow-up searches to knowledge base."
                        send_progress(self.progress_callback, "log", {"level": "info", "message": add_msg})
                        # e) Re-run Local Retrieval
                        rerun_msg = "Re-running local retrieval with updated knowledge base..."
                        send_progress(self.progress_callback, "phase_start", {"phase": "local_retrieval_rerun", "message": rerun_msg})
                        # KB.search uses its internal embedder
                        self.local_results = self.kb.search(
                            self.enhanced_query,
                            top_k=self.resolved_settings['top_k'] # Use the original top_k
                        )
                        send_progress(self.progress_callback, "log", {"level": "info", "message": f"Retrieved {len(self.local_results)} documents after follow-up searches."})
                        send_progress(self.progress_callback, "phase_end", {"phase": "local_retrieval_rerun", "message": "Finished re-running local retrieval."})
                    else:
                         send_progress(self.progress_callback, "log", {"level": "info", "message": "No new documents added from follow-up searches."})

                else:
                    send_progress(self.progress_callback, "log", {"level": "info", "message": "LLM did not generate any follow-up queries."})
            # End iterative search phase
            send_progress(self.progress_callback, "phase_end", {"phase": "iterative_search", "message": "Finished iterative search step."})
        else:
             if self.resolved_settings.get('enable_iterative_search', False):
                  skip_msg = "Iterative search enabled, but RAG model is 'None'. Skipping follow-up query generation."
                  send_progress(self.progress_callback, "log", {"level": "info", "message": skip_msg})
             # else: # Iterative search not enabled, do nothing extra


        # 5) Summaries and final RAG generation (Now uses potentially updated local_results)
        send_progress(self.progress_callback, "phase_start", {"phase": "summarization", "message": "Summarizing results..."})
        # Summarization functions should ideally use send_progress internally
        send_progress(self.progress_callback, "status", {"message": "Summarizing web results..."})
        summarized_web, self._reference_links = summarization.summarize_web_results(
            self.web_results, self.config, self.resolved_settings, self.progress_callback, self.cache_manager # Pass cache_manager
        )
        send_progress(self.progress_callback, "status", {"message": "Summarizing local results..."})
        summarized_local = summarization.summarize_local_results(
            self.local_results, self.config, self.resolved_settings, self.progress_callback, self.cache_manager # Pass cache_manager
        )
        send_progress(self.progress_callback, "phase_end", {"phase": "summarization", "message": "Finished summarizing results."})

        send_progress(self.progress_callback, "phase_start", {"phase": "reporting", "message": "Building final report using RAG..."})
        # Reporting function should ideally use send_progress internally
        final_answer = reporting.build_final_answer(
            enhanced_query=self.enhanced_query,
            toc_tree_nodes=self.toc_tree,
            summarized_web=summarized_web,
            summarized_local=summarized_local,
            reference_links=self._reference_links, # Pass stored links
            resolved_settings=self.resolved_settings,
            progress_callback=self.progress_callback,
            include_visuals=self.include_visuals # Pass the stored setting
            # Pass previous_results_content, follow_up_convo if needed
        )
        send_progress(self.progress_callback, "phase_end", {"phase": "reporting", "message": "Finished building final report."})
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
