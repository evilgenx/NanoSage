import logging # Added logging
from llm_providers.tasks import summarize_text
from cache_manager import CacheManager # Added import
from typing import Optional # Added import
from web_scraper import scrape_url_to_markdown # Use the correct function
from document_parsers.factory import get_parser # Assume this helps parse local files

logger = logging.getLogger(__name__) # Added logger

def summarize_web_results(web_results, config, resolved_settings, progress_callback, cache_manager: Optional[CacheManager] = None): # Added cache_manager
    """Summarizes the full content of collected web results."""
    content_blocks = []
    reference_urls = []
    progress_callback(f"Fetching and preparing content from {len(web_results)} web results for summarization...")
    for i, w in enumerate(web_results):
        url = w.get('url')
        if not url:
            continue
        reference_urls.append(url)
        progress_callback(f"Fetching web content {i+1}/{len(web_results)}: {url}")
        try:
            # Call the correct function which returns (content, error)
            # Consider adding respect_robots=True/False based on config if needed
            content, error = scrape_url_to_markdown(url) # timeout can also be configured

            if error:
                logger.warning(f"Failed to scrape {url}: {error}")
                content_blocks.append(f"--- Content from {url} ---\n[Scraping failed: {error}]\n--- End Content from {url} ---")
            elif content:
                content_blocks.append(f"--- Content from {url} ---\n{content}\n--- End Content from {url} ---")
            else:
                # Handle case where scrape returns (None, None) or ("", None)
                logger.warning(f"No content extracted or returned from URL: {url}")
                content_blocks.append(f"--- Content from {url} ---\n[No content extracted or returned]\n--- End Content from {url} ---")
        except Exception as e:
            # Catch any unexpected errors during the scraping call itself
            logger.error(f"Unexpected error calling scrape_url_to_markdown for {url}: {e}", exc_info=True)
            content_blocks.append(f"--- Content from {url} ---\n[Unexpected error during scraping: {e}]\n--- End Content from {url} ---")

    text = "\n\n".join(content_blocks) # Join blocks with double newline for separation
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
    """Summarizes the full content of collected local results."""
    content_blocks = []
    progress_callback(f"Parsing and preparing content from {len(local_results)} local results for summarization...")
    for i, doc in enumerate(local_results):
        meta = doc.get('metadata', {})
        file_path = meta.get('file_path')
        if not file_path:
            continue
        progress_callback(f"Parsing local content {i+1}/{len(local_results)}: {file_path}")
        try:
            # Use document_parsers factory to get the appropriate parser and parse content
            parser = get_parser(file_path)
            if parser:
                # Assuming parser.parse() returns the text content
                # This might need adjustment based on parser implementation (e.g., handling binary data, errors)
                content = parser.parse(file_path)
                if content:
                    content_blocks.append(f"--- Content from {file_path} ---\n{content}\n--- End Content from {file_path} ---")
                else:
                    logger.warning(f"No content extracted from file: {file_path}")
                    content_blocks.append(f"--- Content from {file_path} ---\n[No content extracted]\n--- End Content from {file_path} ---")
            else:
                logger.warning(f"No suitable parser found for file: {file_path}")
                content_blocks.append(f"--- Content from {file_path} ---\n[Unsupported file type or parser error]\n--- End Content from {file_path} ---")
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}", exc_info=True)
            content_blocks.append(f"--- Content from {file_path} ---\n[Error parsing content: {e}]\n--- End Content from {file_path} ---")

    text = "\n\n".join(content_blocks) # Join blocks with double newline
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
