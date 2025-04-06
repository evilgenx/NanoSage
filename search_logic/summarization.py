import logging # Added logging
from llm_providers.tasks import summarize_text
from cache_manager import CacheManager # Added import
from typing import Optional, Dict, Any # Added import
from web_scraper import scrape_url_to_markdown # Use the correct function
from document_parsers.factory import get_parser # Assume this helps parse local files
# Import send_progress helper
from .web_recursive import send_progress # Use relative import

logger = logging.getLogger(__name__)

# Define a separator for clarity between individual summaries
SUMMARY_SEPARATOR = "\n\n---\n\n"

def summarize_web_results(web_results, config, resolved_settings, progress_callback, cache_manager: Optional[CacheManager] = None):
    """
    Fetches content for each web result, summarizes it individually,
    and returns a combined string of all summaries.
    """
    individual_summaries = []
    reference_urls = []
    start_msg = f"Fetching and summarizing content from {len(web_results)} web results individually..."
    send_progress(progress_callback, "phase_start", {"phase": "summarize_web", "message": start_msg})

    # Assemble llm_config for the summarization task once
    provider = resolved_settings.get('rag_model', 'gemma')
    model_id = resolved_settings.get(f"{provider}_model_id")
    api_key = resolved_settings.get(f"{provider}_api_key")
    llm_config_for_summary = {
        "provider": provider,
        "model_id": model_id,
        "api_key": api_key,
        "personality": resolved_settings.get('personality')
    }
    max_chars = config.get('advanced', {}).get("summarization_chunk_size", 6000)

    for i, w in enumerate(web_results):
        url = w.get('url')
        if not url:
            continue
        reference_urls.append(url)
        process_msg = f"Processing web result {i+1}/{len(web_results)}: {url}"
        send_progress(progress_callback, "progress_update", {"phase": "summarize_web", "current": i + 1, "total": len(web_results), "unit": "Web Results", "message": process_msg})
        send_progress(progress_callback, "log", {"level": "debug", "message": process_msg})

        content = None
        error_msg = None
        summary = None

        try:
            # 1. Scrape Content
            fetch_msg = f"Fetching content for {url}..."
            send_progress(progress_callback, "log", {"level": "debug", "message": fetch_msg})
            content, scrape_error = scrape_url_to_markdown(url) # timeout can be configured

            if scrape_error:
                warn_msg = f"Failed to scrape {url}: {scrape_error}"
                logger.warning(warn_msg) # Keep logger call
                send_progress(progress_callback, "log", {"level": "warning", "message": warn_msg})
                error_msg = f"[Scraping failed: {scrape_error}]"
            elif not content:
                warn_msg = f"No content extracted or returned from URL: {url}"
                logger.warning(warn_msg) # Keep logger call
                send_progress(progress_callback, "log", {"level": "warning", "message": warn_msg})
                error_msg = "[No content extracted or returned]"

            # 2. Summarize Content (if successfully scraped)
            if content:
                summarize_msg = f"Summarizing content for {url}..."
                send_progress(progress_callback, "log", {"level": "debug", "message": summarize_msg})
                summary = summarize_text(
                    content,
                    llm_config=llm_config_for_summary,
                    max_chars=max_chars,
                    cache_manager=cache_manager # Pass cache_manager for internal caching
                )
                if not summary or summary.startswith("Error:"):
                    err_msg = f"Failed to summarize content from {url}: {summary}"
                    logger.error(err_msg) # Keep logger call
                    send_progress(progress_callback, "log", {"level": "error", "message": err_msg})
                    error_msg = f"[Summarization failed: {summary}]"
                    summary = None # Ensure summary is None if it failed

        except Exception as e:
            err_msg = f"Unexpected error processing {url}: {e}"
            logger.error(err_msg, exc_info=True) # Keep logger call
            send_progress(progress_callback, "log", {"level": "error", "message": err_msg, "details": traceback.format_exc()})
            error_msg = f"[Unexpected error during processing: {e}]"
            content = None # Ensure content is None on unexpected error
            summary = None # Ensure summary is None on unexpected error

        # 3. Append Summary or Error Message
        if summary:
            individual_summaries.append(f"--- Summary from {url} ---\n{summary}\n--- End Summary from {url} ---")
        elif error_msg:
            individual_summaries.append(f"--- Summary from {url} ---\n{error_msg}\n--- End Summary from {url} ---")
        # If no content and no error (shouldn't happen with current logic, but safe)
        # else:
        #     individual_summaries.append(f"--- Summary from {url} ---\n[No content to summarize]\n--- End Summary from {url} ---")


    # Combine individual summaries
    combined_summaries = SUMMARY_SEPARATOR.join(individual_summaries)
    reference_links = sorted(list(set(reference_urls))) # unique and sorted

    if not combined_summaries.strip():
         warn_msg = "No web results could be summarized."
         send_progress(progress_callback, "log", {"level": "warning", "message": warn_msg})
         # Return empty string for summary, but still return collected links
         send_progress(progress_callback, "phase_end", {"phase": "summarize_web", "message": "Finished summarizing web results (none summarized)."})
         return "", reference_links

    end_msg = "Finished summarizing all web results."
    send_progress(progress_callback, "phase_end", {"phase": "summarize_web", "message": end_msg})
    return combined_summaries, reference_links


def summarize_local_results(local_results, config, resolved_settings, progress_callback, cache_manager: Optional[CacheManager] = None):
    """
    Parses content for each local result, summarizes it individually,
    and returns a combined string of all summaries.
    """
    individual_summaries = []
    start_msg = f"Parsing and summarizing content from {len(local_results)} local results individually..."
    send_progress(progress_callback, "phase_start", {"phase": "summarize_local", "message": start_msg})

    # Assemble llm_config for the summarization task once
    provider = resolved_settings.get('rag_model', 'gemma')
    model_id = resolved_settings.get(f"{provider}_model_id")
    api_key = resolved_settings.get(f"{provider}_api_key")
    llm_config_for_summary = {
        "provider": provider,
        "model_id": model_id,
        "api_key": api_key,
        "personality": resolved_settings.get('personality')
    }
    max_chars = config.get('advanced', {}).get("summarization_chunk_size", 6000)

    for i, doc in enumerate(local_results):
        meta = doc.get('metadata', {})
        file_path = meta.get('file_path')
        if not file_path:
            continue
        process_msg = f"Processing local result {i+1}/{len(local_results)}: {file_path}"
        send_progress(progress_callback, "progress_update", {"phase": "summarize_local", "current": i + 1, "total": len(local_results), "unit": "Local Results", "message": process_msg})
        send_progress(progress_callback, "log", {"level": "debug", "message": process_msg})

        content = None
        error_msg = None
        summary = None

        try:
            # 1. Parse Content
            parse_msg = f"Parsing content for {file_path}..."
            send_progress(progress_callback, "log", {"level": "debug", "message": parse_msg})
            parser = get_parser(file_path)
            if parser:
                content = parser.parse(file_path)
                if not content:
                    warn_msg = f"No content extracted from file: {file_path}"
                    logger.warning(warn_msg) # Keep logger call
                    send_progress(progress_callback, "log", {"level": "warning", "message": warn_msg})
                    error_msg = "[No content extracted]"
            else:
                warn_msg = f"No suitable parser found for file: {file_path}"
                logger.warning(warn_msg) # Keep logger call
                send_progress(progress_callback, "log", {"level": "warning", "message": warn_msg})
                error_msg = "[Unsupported file type or parser error]"

            # 2. Summarize Content (if successfully parsed)
            if content:
                summarize_msg = f"Summarizing content for {file_path}..."
                send_progress(progress_callback, "log", {"level": "debug", "message": summarize_msg})
                summary = summarize_text(
                    content,
                    llm_config=llm_config_for_summary,
                    max_chars=max_chars,
                    cache_manager=cache_manager # Pass cache_manager for internal caching
                )
                if not summary or summary.startswith("Error:"):
                    err_msg = f"Failed to summarize content from {file_path}: {summary}"
                    logger.error(err_msg) # Keep logger call
                    send_progress(progress_callback, "log", {"level": "error", "message": err_msg})
                    error_msg = f"[Summarization failed: {summary}]"
                    summary = None # Ensure summary is None if it failed

        except Exception as e:
            err_msg = f"Error processing file {file_path}: {e}"
            logger.error(err_msg, exc_info=True) # Keep logger call
            send_progress(progress_callback, "log", {"level": "error", "message": err_msg, "details": traceback.format_exc()})
            error_msg = f"[Error processing content: {e}]"
            content = None # Ensure content is None on unexpected error
            summary = None # Ensure summary is None on unexpected error

        # 3. Append Summary or Error Message
        if summary:
            individual_summaries.append(f"--- Summary from {file_path} ---\n{summary}\n--- End Summary from {file_path} ---")
        elif error_msg:
            individual_summaries.append(f"--- Summary from {file_path} ---\n{error_msg}\n--- End Summary from {file_path} ---")
        # If no content and no error
        # else:
        #     individual_summaries.append(f"--- Summary from {file_path} ---\n[No content to summarize]\n--- End Summary from {file_path} ---")


    # Combine individual summaries
    combined_summaries = SUMMARY_SEPARATOR.join(individual_summaries)

    if not combined_summaries.strip():
         warn_msg = "No local results could be summarized."
         send_progress(progress_callback, "log", {"level": "warning", "message": warn_msg})
         send_progress(progress_callback, "phase_end", {"phase": "summarize_local", "message": "Finished summarizing local results (none summarized)."})
         return "" # Return empty string

    end_msg = "Finished summarizing all local results."
    send_progress(progress_callback, "phase_end", {"phase": "summarize_local", "message": end_msg})
    return combined_summaries
