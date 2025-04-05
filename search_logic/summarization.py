from llm_providers.tasks import summarize_text
from cache_manager import CacheManager # Added import
from typing import Optional # Added import

def summarize_web_results(web_results, config, resolved_settings, progress_callback, cache_manager: Optional[CacheManager] = None): # Added cache_manager
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
    summary = summarize_text(
        text,
        llm_config=llm_config_for_summary,
        max_chars=max_chars,
        cache_manager=cache_manager # Pass cache_manager
    )
    progress_callback("Finished summarizing web results.")
    return summary, reference_links

def summarize_local_results(local_results, config, resolved_settings, progress_callback, cache_manager: Optional[CacheManager] = None): # Added cache_manager
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
    summary = summarize_text(
        text,
        llm_config=llm_config_for_summary,
        max_chars=max_chars,
        cache_manager=cache_manager # Pass cache_manager
    )
    progress_callback("Finished summarizing local results.")
    return summary
