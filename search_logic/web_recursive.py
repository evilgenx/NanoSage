import os
import asyncio
from llm_providers.utils import clean_search_query, split_query
from llm_providers.tasks import summarize_text, chain_of_thought_query_enhancement
import toc_tree
from knowledge_base import embed_text, late_interaction_score
from web_search import download_webpages_ddg, parse_html_to_text, group_web_results_by_domain, sanitize_filename

async def perform_recursive_web_searches(
    subqueries,
    current_depth,
    max_depth,
    base_result_dir,
    enhanced_query_embedding,
    resolved_settings,
    config, # Pass raw config for advanced settings
    progress_callback,
    model, # Pass model/processor/type explicitly
    processor,
    model_type
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

        # Embed subquery for relevance check using resolved settings
        node_embedding = embed_text(
            text=sq_clean,
            model=model,
            processor=processor,
            model_type=model_type,
            embedding_model_name=resolved_settings['embedding_model'], # Use resolved embedding model
            device=resolved_settings['device'], # Use resolved device
            # Pass API keys explicitly if embed_text requires them
            gemini_api_key=resolved_settings.get('gemini_api_key'),
            openrouter_api_key=resolved_settings.get('openrouter_api_key')
        )
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

        # Pass the progress_callback to the download function
        # Use resolved web_search_limit from config (assuming it might be under 'advanced')
        web_search_limit = config.get('advanced', {}).get("web_search_limit", 5)
        pages = await download_webpages_ddg(
            sq_clean,
            limit=web_search_limit,
            output_dir=subquery_dir,
            progress_callback=progress_callback # Pass the callback here
        )
        progress_callback(f"Downloaded {len(pages)} pages for '{sq_clean[:50]}...'")
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

            # Embed the web page content using resolved settings
            emb = embed_text(
                text=limited_text,
                model=model,
                processor=processor,
                model_type=model_type,
                embedding_model_name=resolved_settings['embedding_model'], # Use resolved embedding model
                device=resolved_settings['device'], # Use resolved device
                # Pass API keys explicitly if embed_text requires them
                gemini_api_key=resolved_settings.get('gemini_api_key'),
                openrouter_api_key=resolved_settings.get('openrouter_api_key')
            )

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
        # Pass all resolved model params to summarize_text
        # Use resolved summarization_chunk_size from config (assuming it might be under 'advanced')
        max_chars = config.get('advanced', {}).get("summarization_chunk_size", 6000)
        toc_node.summary = summarize_text(
            branch_snippets,
            max_chars=max_chars,
            personality=resolved_settings['personality'],
            rag_model=resolved_settings['rag_model'],
            selected_gemini_model=resolved_settings['gemini_model_id'],
            selected_openrouter_model=resolved_settings['openrouter_model_id'],
            # Pass API keys explicitly if summarize_text requires them
            gemini_api_key=resolved_settings.get('gemini_api_key'),
            openrouter_api_key=resolved_settings.get('openrouter_api_key')
        )
        toc_node.web_results = branch_web_results
        # Note: We don't store corpus entries directly in TOC node anymore, they go straight to KB

        additional_subqueries = []
        # Use resolved max_depth
        if current_depth < max_depth:
            progress_callback(f"Generating potential sub-subqueries for '{sq_clean[:50]}...'")
            # Use the chain_of_thought_query_enhancement function directly, passing all resolved model params
            additional_query = chain_of_thought_query_enhancement(
                sq_clean,
                personality=resolved_settings['personality'],
                rag_model=resolved_settings['rag_model'],
                selected_gemini_model=resolved_settings['gemini_model_id'],
                selected_openrouter_model=resolved_settings['openrouter_model_id'],
                # Pass API keys explicitly if function requires them
                gemini_api_key=resolved_settings.get('gemini_api_key'),
                openrouter_api_key=resolved_settings.get('openrouter_api_key')
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
                model=model,
                processor=processor,
                model_type=model_type
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
