from llm_providers.tasks import summarize_text

def summarize_web_results(web_results, config, resolved_settings, progress_callback):
    """Summarizes the collected web results."""
    lines = []
    reference_urls = []
    progress_callback(f"Preparing {len(web_results)} web results for summarization...")
    for w in web_results:
        url = w.get('url')
        snippet = w.get('snippet')
        lines.append(f"URL: {url} - snippet: {snippet}")
        if url: # Only add valid URLs
             reference_urls.append(url)
    text = "\n".join(lines)
    # Store reference URLs for the final prompt (return them)
    reference_links = sorted(list(set(reference_urls)))  # unique and sorted
    if not text.strip():
         progress_callback("No web results to summarize.")
         return "No web results found or summarized.", reference_links # Return empty links too

    progress_callback("Calling LLM to summarize web results...")
    # Pass all resolved model params to summarize_text
    # Use resolved summarization_chunk_size from config (assuming it might be under 'advanced')
    max_chars = config.get('advanced', {}).get("summarization_chunk_size", 6000)
    summary = summarize_text(
        text,
        max_chars=max_chars,
        personality=resolved_settings['personality'],
        rag_model=resolved_settings['rag_model'],
        selected_gemini_model=resolved_settings['gemini_model_id'],
        selected_openrouter_model=resolved_settings['openrouter_model_id'],
        # Pass API keys explicitly if summarize_text requires them
        gemini_api_key=resolved_settings.get('gemini_api_key'),
        openrouter_api_key=resolved_settings.get('openrouter_api_key')
    )
    progress_callback("Finished summarizing web results.")
    return summary, reference_links

def summarize_local_results(local_results, config, resolved_settings, progress_callback):
    """Summarizes the collected local results."""
    lines = []
    progress_callback(f"Preparing {len(local_results)} local results for summarization...")
    for doc in local_results:
        meta = doc.get('metadata', {})
        file_path = meta.get('file_path')
        snippet = meta.get('snippet', '')
        lines.append(f"File: {file_path} snippet: {snippet}")
    text = "\n".join(lines)
    if not text.strip():
         progress_callback("No local results to summarize.")
         return "No local documents found or summarized."

    progress_callback("Calling LLM to summarize local results...")
    # Pass all resolved model params to summarize_text
    # Use resolved summarization_chunk_size from config (assuming it might be under 'advanced')
    max_chars = config.get('advanced', {}).get("summarization_chunk_size", 6000)
    summary = summarize_text(
        text,
        max_chars=max_chars,
        personality=resolved_settings['personality'],
        rag_model=resolved_settings['rag_model'],
        selected_gemini_model=resolved_settings['gemini_model_id'],
        selected_openrouter_model=resolved_settings['openrouter_model_id'],
        # Pass API keys explicitly if summarize_text requires them
        gemini_api_key=resolved_settings.get('gemini_api_key'),
        openrouter_api_key=resolved_settings.get('openrouter_api_key')
    )
    progress_callback("Finished summarizing local results.")
    return summary
