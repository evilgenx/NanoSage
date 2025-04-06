import os
import os
import asyncio
import traceback # For error handling
from llm_providers.utils import clean_search_query, split_query
from llm_providers.tasks import summarize_text, chain_of_thought_query_enhancement
from toc_tree import TOCNode # Import the class directly
from knowledge_base import late_interaction_score # Removed embed_text
# Import both download functions
from web_search import download_webpages_ddg, download_webpages_searxng, parse_html_to_text, group_web_results_by_domain, sanitize_filename
from embeddings.base import BaseEmbedder # Added import
from cache_manager import CacheManager # Added import
from typing import Optional, List, Dict, Any # Added import

# Helper function for sending structured progress updates
def send_progress(callback, message_type: str, data: Dict[str, Any]):
    if callback:
        try:
            callback({"type": message_type, **data})
        except Exception as e:
            # Avoid crashing the backend if the callback fails (e.g., GUI closed)
            print(f"[WARN] Failed to send progress update ({message_type}): {e}")

async def perform_recursive_web_searches(
    subqueries: List[str],
    current_depth: int,
    max_depth: int,
    base_result_dir: str,
    enhanced_query_embedding: Any, # Type depends on embedder
    resolved_settings: Dict[str, Any],
    config: Dict[str, Any], # Pass raw config for advanced settings
    progress_callback: Optional[callable],
    embedder: BaseEmbedder, # Changed signature to use embedder
    cache_manager: Optional[CacheManager] = None, # Added cache_manager
    cancellation_check_callback: Optional[callable] = None, # Added cancellation callback
    parent_node_id: Optional[str] = None # Added parent_node_id
):
    """
    Recursively perform web searches for each subquery up to max_depth.
    Sends progress updates for TOC visualization.
    Returns:
      aggregated_web_results, aggregated_corpus_entries, grouped_results, toc_nodes
    """
    aggregated_web_results = []
    aggregated_corpus_entries = []
    toc_nodes = []
    # Use resolved min_relevance from config (assuming it might be under 'advanced')
    min_relevance = config.get('advanced', {}).get("min_relevance", 0.5)

    for sq in subqueries:
        toc_node = None # Initialize toc_node to None for error handling
        try:
            # --- Cancellation Check ---
            if cancellation_check_callback and cancellation_check_callback():
                send_progress(progress_callback, "status", {"message": "Cancellation requested during web recursion."})
                print("[INFO] Cancellation requested in web_recursive loop.")
                raise asyncio.CancelledError("Search cancelled by user during web recursion")
            # --- End Cancellation Check ---

            sq_clean = clean_search_query(sq) # Use clean_search_query from utils
            if not sq_clean:
                continue

            # Create a TOC node and send initial add message
            toc_node = TOCNode(query_text=sq_clean, depth=current_depth, parent_id=parent_node_id)
            send_progress(progress_callback, "toc_add", {"node_data": toc_node.to_dict()})

            # Embed subquery for relevance check using the provided embedder
            node_embedding = embedder.embed(sq_clean) # Use embedder.embed

            if node_embedding is None:
                 print(f"[WARN] Failed to embed subquery '{sq_clean[:50]}...' for relevance check. Skipping branch.")
                 toc_node.status = TOCNode.STATUS_SKIPPED # Update status
                 send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status, "message": "Embedding failed"})
                 continue # Skip this subquery

            relevance = late_interaction_score(enhanced_query_embedding, node_embedding)
            toc_node.relevance_score = relevance
            # Send update with relevance score
            send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "relevance": f"{relevance:.2f}"})


            if relevance < min_relevance:
                send_progress(progress_callback, "status", {"message": f"Skipping branch (low relevance {relevance:.2f}): '{sq_clean[:50]}...'"})
                print(f"[INFO] Skipping branch '{sq_clean}' due to low relevance ({relevance:.2f} < {min_relevance}).")
                toc_node.status = TOCNode.STATUS_SKIPPED # Update status
                send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status, "message": f"Low relevance {relevance:.2f}"})
                continue # Skip this subquery

            # Create subdirectory
            safe_subquery = sanitize_filename(sq_clean)[:30]
            subquery_dir = os.path.join(base_result_dir, f"web_{safe_subquery}")
            os.makedirs(subquery_dir, exist_ok=True)

            # Update status to Searching
            toc_node.status = TOCNode.STATUS_SEARCHING
            send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status})
            send_progress(progress_callback, "status", {"message": f"Searching web (Depth {current_depth}, Rel: {relevance:.2f}): '{sq_clean[:50]}...'"})
            print(f"[DEBUG] Searching web for subquery '{sq_clean}' at depth={current_depth}...")

            # --- Cancellation Check ---
            if cancellation_check_callback and cancellation_check_callback():
                raise asyncio.CancelledError("Search cancelled by user before web download.")
            # --- End Cancellation Check ---

            # --- Select and call the appropriate web search function ---
            search_provider = resolved_settings.get('search_provider', 'duckduckgo')
            # search_limit is now read inside download_webpages_searxng from config
            pages = []

            if search_provider == 'searxng':
                # Check if base_url exists in config before calling
                if config.get('search', {}).get('searxng', {}).get('base_url'):
                    pages = await download_webpages_searxng(
                        keyword=sq_clean,
                        config=config, # Pass the whole config object
                        output_dir=subquery_dir,
                        progress_callback=progress_callback, # Pass generic callback
                        cache_manager=cache_manager # Pass cache_manager
                        # pageno defaults to 1 if not specified
                    )
                else:
                    send_progress(progress_callback, "status", {"message": "[WARN] SearXNG provider selected but base_url not found in config. Skipping web search."})
                    print("[WARN] SearXNG provider selected but base_url not found in config.")
            elif search_provider == 'duckduckgo':
                 # DDG still uses limit argument directly
                 search_limit = resolved_settings.get('search_max_results', 5)
                 pages = await download_webpages_ddg(
                     keyword=sq_clean,
                     limit=search_limit, # Pass limit here
                     output_dir=subquery_dir,
                     progress_callback=progress_callback, # Pass generic callback
                      cache_manager=cache_manager # Pass cache_manager
                 )
            else:
                 send_progress(progress_callback, "status", {"message": f"[WARN] Unknown search provider '{search_provider}'. Skipping web search."})
                 print(f"[WARN] Unknown search provider '{search_provider}'. Skipping web search.")
            # --- End search function selection ---

            # --- Cancellation Check ---
            if cancellation_check_callback and cancellation_check_callback():
                raise asyncio.CancelledError("Search cancelled by user after web download.")
            # --- End Cancellation Check ---

            send_progress(progress_callback, "status", {"message": f"Downloaded {len(pages)} pages via {search_provider} for '{sq_clean[:50]}...'"})
            branch_web_results = []
            branch_corpus_entries = []
            for i, page in enumerate(pages):
                if not page:
                    continue
                file_path = page.get("file_path")
                url = page.get("url")
                if not file_path or not url:
                    continue

                # --- Cancellation Check ---
                if cancellation_check_callback and cancellation_check_callback():
                    raise asyncio.CancelledError("Search cancelled by user during page processing loop.")
                # --- End Cancellation Check ---

                raw_text = parse_html_to_text(file_path)
                if not raw_text or not raw_text.strip():
                    print(f"[INFO] Skipping empty page content from {url}")
                    continue

                snippet = raw_text[:100].replace('\n', ' ') + "..."
                # Use resolved summarization_chunk_size from config for limiting text length (assuming it might be under 'advanced')
                max_embed_chars = config.get('advanced', {}).get("summarization_chunk_size", 2048) # Reuse summarization chunk size as a proxy
                limited_text = raw_text[:max_embed_chars] # Limit text length for embedding if needed

                # Embed the web page content using the provided embedder
                emb = embedder.embed(limited_text) # Use embedder.embed

                if emb is not None:
                    entry = {
                        "embedding": emb, # embed_text now returns tensor on CPU
                        "metadata": {
                            "file_path": file_path,
                            "type": "webhtml",
                            "snippet": snippet,
                            "url": url
                        }
                    }
                    branch_corpus_entries.append(entry)
                    branch_web_results.append({"url": url, "snippet": snippet})
                else:
                    send_progress(progress_callback, "status", {"message": f"[WARN] Failed to embed page content from '{url}'. Skipping."})
                    print(f"[WARN] Failed to embed page content from '{url}'. Skipping.")


            # Summarize using resolved settings
            # Update status to Summarizing
            toc_node.status = TOCNode.STATUS_SUMMARIZING
            send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status})
            send_progress(progress_callback, "status", {"message": f"Summarizing {len(branch_web_results)} web results for '{sq_clean[:50]}...'"})

            # --- Cancellation Check ---
            if cancellation_check_callback and cancellation_check_callback():
                raise asyncio.CancelledError("Search cancelled by user before summarization.")
            # --- End Cancellation Check ---

            branch_snippets = " ".join([r.get("snippet", "") for r in branch_web_results])
            # Assemble llm_config for the summarization task
            provider = resolved_settings.get('rag_model', 'gemma')
            model_id = resolved_settings.get(f"{provider}_model_id") # e.g., gemini_model_id
            api_key = resolved_settings.get(f"{provider}_api_key") # e.g., gemini_api_key
            llm_config_for_summary = {
                "provider": provider,
                "model_id": model_id,
                "api_key": api_key,
                "personality": resolved_settings.get('personality')
            }
            # Use resolved summarization_chunk_size from config (assuming it might be under 'advanced')
            max_chars = config.get('advanced', {}).get("summarization_chunk_size", 6000)
            toc_node.summary = summarize_text(
                branch_snippets,
                llm_config=llm_config_for_summary,
                max_chars=max_chars
            )

            # --- Cancellation Check ---
            if cancellation_check_callback and cancellation_check_callback():
                raise asyncio.CancelledError("Search cancelled by user after summarization.")
            # --- End Cancellation Check ---

            toc_node.web_results = branch_web_results
            toc_node.content_relevance_score = None # Initialize content relevance score
            # Send update with summary snippet
            send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "summary_snippet": (toc_node.summary[:50] + "...") if toc_node.summary else ""})


            # Note: We don't store corpus entries directly in TOC node anymore, they go straight to KB

            additional_subqueries = []
            proceed_with_recursion = True # Flag to control recursion

            # --- Content Relevance Check ---
            if toc_node.summary and toc_node.summary.strip():
                summary_embedding = embedder.embed(toc_node.summary)
                if summary_embedding is not None:
                    content_relevance = late_interaction_score(enhanced_query_embedding, summary_embedding)
                    toc_node.content_relevance_score = content_relevance
                    # Send update with content relevance
                    send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "content_relevance": f"{content_relevance:.2f}"})

                    if content_relevance < min_relevance:
                        send_progress(progress_callback, "status", {"message": f"Skipping deeper search (low content relevance {content_relevance:.2f}): '{sq_clean[:50]}...'"})
                        print(f"[INFO] Skipping deeper search for '{sq_clean}' due to low content relevance ({content_relevance:.2f} < {min_relevance}).")
                        proceed_with_recursion = False
                    else:
                         send_progress(progress_callback, "status", {"message": f"Content relevance check passed ({content_relevance:.2f}) for '{sq_clean[:50]}...'"})
                else:
                    send_progress(progress_callback, "status", {"message": f"[WARN] Failed to embed summary for content relevance check: '{sq_clean[:50]}...'"})
                    print(f"[WARN] Failed to embed summary for content relevance check: '{sq_clean}'.")
                    # Decide if failure to embed summary should halt recursion (optional, currently allows recursion)
            else:
                send_progress(progress_callback, "status", {"message": f"Skipping content relevance check (empty summary): '{sq_clean[:50]}...'"})
                # Decide if empty summary should halt recursion (optional, currently allows recursion)
            # --- End Content Relevance Check ---


            # Use resolved max_depth and proceed_with_recursion flag
            if proceed_with_recursion and current_depth < max_depth:
                # Update status to Expanding
                toc_node.status = TOCNode.STATUS_EXPANDING
                send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status})
                send_progress(progress_callback, "status", {"message": f"Generating potential sub-subqueries for '{sq_clean[:50]}...'"})

                # --- Cancellation Check ---
                if cancellation_check_callback and cancellation_check_callback():
                    raise asyncio.CancelledError("Search cancelled by user before query enhancement.")
                # --- End Cancellation Check ---

                # Assemble llm_config for the enhancement task
                provider = resolved_settings.get('rag_model', 'gemma')
                model_id = resolved_settings.get(f"{provider}_model_id") # e.g., gemini_model_id
                api_key = resolved_settings.get(f"{provider}_api_key") # e.g., gemini_api_key
                llm_config_for_enhancement = {
                    "provider": provider,
                    "model_id": model_id,
                    "api_key": api_key,
                    "personality": resolved_settings.get('personality')
                }
                # Use the chain_of_thought_query_enhancement function directly
                additional_query = chain_of_thought_query_enhancement(
                    sq_clean,
                    llm_config=llm_config_for_enhancement
                )

                # --- Cancellation Check ---
                if cancellation_check_callback and cancellation_check_callback():
                    raise asyncio.CancelledError("Search cancelled by user after query enhancement.")
                # --- End Cancellation Check ---

                if additional_query and additional_query != sq_clean:
                    # Use resolved max_query_length from config (assuming it might be under 'advanced')
                    max_query_length = config.get('advanced', {}).get("max_query_length", 200)
                    additional_subqueries = split_query(additional_query, max_len=max_query_length) # Use split_query from utils
                    send_progress(progress_callback, "status", {"message": f"Generated {len(additional_subqueries)} sub-subqueries."})

            if additional_subqueries:
                # --- Cancellation Check ---
                if cancellation_check_callback and cancellation_check_callback():
                    send_progress(progress_callback, "status", {"message": "Cancellation requested before recursive web call."})
                    print("[INFO] Cancellation requested before recursive web call.")
                    raise asyncio.CancelledError("Search cancelled by user before recursive web call")
                # --- End Cancellation Check ---

                # Recursive call returns results, entries, grouped, and child nodes
                # Need to pass all required arguments down, including the current node's ID as parent_id
                deeper_web_results, deeper_corpus_entries, _, deeper_toc_nodes = await perform_recursive_web_searches(
                    additional_subqueries,
                    current_depth=current_depth+1,
                    max_depth=max_depth,
                    base_result_dir=base_result_dir,
                    enhanced_query_embedding=enhanced_query_embedding,
                    resolved_settings=resolved_settings,
                    config=config,
                    progress_callback=progress_callback,
                    embedder=embedder, # Pass embedder down
                    cache_manager=cache_manager, # Pass cache_manager down
                    cancellation_check_callback=cancellation_check_callback, # Pass callback down
                    parent_node_id=toc_node.node_id # Pass current node ID as parent
                )
                # Extend the current branch's results and entries
                branch_web_results.extend(deeper_web_results)
                branch_corpus_entries.extend(deeper_corpus_entries)
                # Add child nodes to the current TOC node
                for child_node in deeper_toc_nodes:
                    toc_node.add_child(child_node)

            # Update status to Done after all processing for this node (including recursion)
            toc_node.status = TOCNode.STATUS_DONE
            send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status})

            # Aggregate results from this branch
            aggregated_web_results.extend(branch_web_results)
            aggregated_corpus_entries.extend(branch_corpus_entries)
            toc_nodes.append(toc_node)

        except asyncio.CancelledError:
             # Propagate cancellation upwards
             print(f"[INFO] Cancellation caught in web_recursive loop for '{sq_clean}'.")
             if toc_node: # If node was created before cancellation
                 toc_node.status = TOCNode.STATUS_SKIPPED # Or maybe a new 'Cancelled' status?
                 send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status, "message": "Cancelled"})
             raise # Re-raise to stop further processing
        except Exception as e:
            print(f"[ERROR] Error processing subquery '{sq_clean}': {e}\n{traceback.format_exc()}")
            send_progress(progress_callback, "status", {"message": f"[ERROR] Failed processing branch: '{sq_clean[:50]}...'"})
            if toc_node: # If node was created before error
                toc_node.status = TOCNode.STATUS_ERROR
                send_progress(progress_callback, "toc_update", {"node_id": toc_node.node_id, "status": toc_node.status, "message": str(e)})
            # Optionally decide whether to continue with other subqueries or raise the error
            # For now, we continue with the next subquery

    # Grouping happens once after all recursion is done
    grouped = group_web_results_by_domain(
        [{"url": r["url"], "file_path": e["metadata"]["file_path"], "content_type": e["metadata"].get("type", "")} # Use type from metadata
         for r, e in zip(aggregated_web_results, aggregated_corpus_entries) if "metadata" in e and "file_path" in e["metadata"]]
    )
    return aggregated_web_results, aggregated_corpus_entries, grouped, toc_nodes
