import os
import os
import asyncio
from llm_providers.utils import clean_search_query, split_query
from llm_providers.tasks import summarize_text, chain_of_thought_query_enhancement
import toc_tree
from knowledge_base import late_interaction_score # Removed embed_text
# Import both download functions
from web_search import download_webpages_ddg, download_webpages_searxng, parse_html_to_text, group_web_results_by_domain, sanitize_filename
from embeddings.base import BaseEmbedder # Added import

async def perform_recursive_web_searches(
    subqueries,
    current_depth,
    max_depth,
    base_result_dir,
    enhanced_query_embedding,
    resolved_settings,
    config, # Pass raw config for advanced settings
    progress_callback,
    embedder: BaseEmbedder # Changed signature to use embedder
    # Removed model, processor, model_type
):
    """
    Recursively perform web searches for each subquery up to max_depth.
    Returns:
      aggregated_web_results, aggregated_corpus_entries, grouped_results, toc_nodes
    """
    aggregated_web_results = []
    aggregated_corpus_entries = []
    toc_nodes = []
    # Use resolved min_relevance from config (assuming it might be under 'advanced')
    min_relevance = config.get('advanced', {}).get("min_relevance", 0.5)

    for sq in subqueries:
        sq_clean = clean_search_query(sq) # Use clean_search_query from utils
        if not sq_clean:
            continue

        # Create a TOC node
        toc_node = toc_tree.TOCNode(query_text=sq_clean, depth=current_depth) # Use toc_tree

        # Embed subquery for relevance check using the provided embedder
        node_embedding = embedder.embed(sq_clean) # Use embedder.embed

        if node_embedding is None:
             print(f"[WARN] Failed to embed subquery '{sq_clean[:50]}...' for relevance check. Skipping branch.")
             continue

        relevance = late_interaction_score(enhanced_query_embedding, node_embedding)
        toc_node.relevance_score = relevance

        if relevance < min_relevance:
            progress_callback(f"Skipping branch (low relevance {relevance:.2f}): '{sq_clean[:50]}...'")
            print(f"[INFO] Skipping branch '{sq_clean}' due to low relevance ({relevance:.2f} < {min_relevance}).")
            continue

        # Create subdirectory
        safe_subquery = sanitize_filename(sq_clean)[:30]
        subquery_dir = os.path.join(base_result_dir, f"web_{safe_subquery}")
        os.makedirs(subquery_dir, exist_ok=True)
        progress_callback(f"Searching web (Depth {current_depth}, Rel: {relevance:.2f}): '{sq_clean[:50]}...'")
        print(f"[DEBUG] Searching web for subquery '{sq_clean}' at depth={current_depth}...")

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
                    progress_callback=progress_callback
                    # pageno defaults to 1 if not specified
                )
            else:
                progress_callback("[WARN] SearXNG provider selected but base_url not found in config. Skipping web search for this branch.")
                print("[WARN] SearXNG provider selected but base_url not found in config.")
        elif search_provider == 'duckduckgo':
             # DDG still uses limit argument directly
             search_limit = resolved_settings.get('search_max_results', 5)
             pages = await download_webpages_ddg(
                 keyword=sq_clean,
                 limit=search_limit, # Pass limit here
                 output_dir=subquery_dir,
                 progress_callback=progress_callback
             )
        else:
             progress_callback(f"[WARN] Unknown search provider '{search_provider}'. Skipping web search for this branch.")
             print(f"[WARN] Unknown search provider '{search_provider}'. Skipping web search.")
        # --- End search function selection ---

        progress_callback(f"Downloaded {len(pages)} pages via {search_provider} for '{sq_clean[:50]}...'")
        branch_web_results = []
        branch_corpus_entries = []
        for i, page in enumerate(pages):
            if not page:
                continue
            file_path = page.get("file_path")
            url = page.get("url")
            if not file_path or not url:
                continue

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
                progress_callback(f"[WARN] Failed to embed page content from '{url}'. Skipping.")
                print(f"[WARN] Failed to embed page content from '{url}'. Skipping.")


        # Summarize using resolved settings
        progress_callback(f"Summarizing {len(branch_web_results)} web results for '{sq_clean[:50]}...'")
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
        toc_node.web_results = branch_web_results
        # Note: We don't store corpus entries directly in TOC node anymore, they go straight to KB

        additional_subqueries = []
        # Use resolved max_depth
        if current_depth < max_depth:
            progress_callback(f"Generating potential sub-subqueries for '{sq_clean[:50]}...'")
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
            if additional_query and additional_query != sq_clean:
                # Use resolved max_query_length from config (assuming it might be under 'advanced')
                max_query_length = config.get('advanced', {}).get("max_query_length", 200)
                additional_subqueries = split_query(additional_query, max_len=max_query_length) # Use split_query from utils
                progress_callback(f"Generated {len(additional_subqueries)} sub-subqueries.")

        if additional_subqueries:
            # Recursive call returns results, entries, grouped, and child nodes
            # Need to pass all required arguments down
            deeper_web_results, deeper_corpus_entries, _, deeper_toc_nodes = await perform_recursive_web_searches(
                additional_subqueries,
                current_depth=current_depth+1,
                max_depth=max_depth,
                base_result_dir=base_result_dir,
                enhanced_query_embedding=enhanced_query_embedding,
                resolved_settings=resolved_settings,
                config=config,
                progress_callback=progress_callback,
                embedder=embedder # Pass embedder down
                # Removed model, processor, model_type
            )
            # Extend the current branch's results and entries
            branch_web_results.extend(deeper_web_results)
            branch_corpus_entries.extend(deeper_corpus_entries)
            # Add child nodes to the current TOC node
            for child_node in deeper_toc_nodes:
                toc_node.add_child(child_node)

        # Aggregate results from this branch
        aggregated_web_results.extend(branch_web_results)
        aggregated_corpus_entries.extend(branch_corpus_entries)
        toc_nodes.append(toc_node)

    # Grouping happens once after all recursion is done
    grouped = group_web_results_by_domain(
        [{"url": r["url"], "file_path": e["metadata"]["file_path"], "content_type": e["metadata"].get("type", "")} # Use type from metadata
         for r, e in zip(aggregated_web_results, aggregated_corpus_entries) if "metadata" in e and "file_path" in e["metadata"]]
    )
    return aggregated_web_results, aggregated_corpus_entries, grouped, toc_nodes
