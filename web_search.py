import os
import os
import asyncio
from urllib.parse import urlparse, quote_plus, urlencode # Added urlencode
import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import fitz  # PyMuPDF
import json
import logging # Added logging
from typing import Optional # Added Optional
from cache_manager import CacheManager # Added CacheManager import

logger = logging.getLogger(__name__) # Added logger

def sanitize_filename(filename):
    """Sanitize a filename by allowing only alphanumerics, dot, underscore, and dash."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in filename)

def sanitize_path(path):
    """
    Sanitize a full filesystem path by splitting it into components, sanitizing each component,
    and then rejoining them. This helps avoid Windows invalid characters in any folder names.
    """
    parts = path.split(os.sep)
    sanitized_parts = [sanitize_filename(part) for part in parts if part]
    if path.startswith(os.sep):
        return os.sep + os.sep.join(sanitized_parts)
    else:
        return os.sep.join(sanitized_parts)

# Added cache_manager parameter
async def download_page(session, url, headers, timeout, file_path, progress_callback, cache_manager: Optional[CacheManager] = None):
    # --- Cache Check ---
    if cache_manager:
        cached_content = cache_manager.get_web_content(url)
        if cached_content is not None:
            logger.info(f"Cache hit for '{url}'. Saving to '{file_path}'")
            try:
                # Assume cached content is text for now, write as UTF-8
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cached_content)
                # Return a dict indicating it came from cache (content_type might be inaccurate)
                return {'url': url, 'file_path': file_path, 'content_type': 'text/html; charset=utf-8 (cached)'}
            except Exception as e:
                logger.warning(f"Error writing cached content for {url} to {file_path}: {e}")
                # Proceed to download if writing cache fails
    # --- End Cache Check ---

    try:
        logger.debug(f"Downloading URL: {url}")
        async with session.get(url, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            is_binary = ('application/pdf' in content_type) or file_path.lower().endswith('.pdf') or ('image/' in content_type)

            if is_binary:
                content = await response.read()
                mode = 'wb'
                open_kwargs = {}
                content_to_cache = None # Don't cache binary content for now
            else:
                # Try decoding with utf-8 first, fallback to response encoding or latin-1
                try:
                    content = await response.text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        content = await response.text(encoding=response.get_encoding())
                    except UnicodeDecodeError:
                        logger.warning(f"UTF-8 and response encoding failed for {url}, falling back to latin-1")
                        content_bytes = await response.read()
                        content = content_bytes.decode('latin-1', errors='replace')

                mode = 'w'
                open_kwargs = {'encoding': 'utf-8'} # Always write text as UTF-8
                content_to_cache = content # Cache the text content

            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, mode, **open_kwargs) as f:
                f.write(content)
            logger.info(f"Saved '{url}' -> '{file_path}'")

            # --- Store in Cache (if text content and cache enabled) ---
            if content_to_cache and cache_manager:
                cache_manager.store_web_content(url, content_to_cache)
            # --- End Store in Cache ---

            return {'url': url, 'file_path': file_path, 'content_type': content_type}
    except asyncio.TimeoutError:
        error_message = f"Timeout fetching '{url}'"
        logger.warning(error_message)
        if progress_callback: progress_callback(f"[WARN] {error_message}")
        return None
    except aiohttp.ClientResponseError as e:
        error_message = f"HTTP Error {e.status} fetching '{url}': {e.message}"
        logger.warning(error_message)
        if progress_callback: progress_callback(f"[WARN] HTTP Error {e.status} for: {url}")
        return None
    except aiohttp.ClientError as e: # Catch other client errors like connection issues
        error_message = f"Client error fetching '{url}': {e}"
        logger.warning(error_message)
        if progress_callback: progress_callback(f"[WARN] Network error for: {url}")
        return None
    except Exception as e:
        error_message = f"Couldn't fetch or save '{url}': {repr(e)}"
        logger.warning(error_message)
        if progress_callback: progress_callback(f"[WARN] Error processing: {url}")
        # Log traceback for unexpected errors
        if not isinstance(e, (asyncio.TimeoutError, aiohttp.ClientError)):
            logger.exception(f"Unexpected error during download_page for {url}")
        return None


# Added cache_manager parameter
async def download_webpages_ddg(keyword, limit=5, output_dir='downloaded_webpages', progress_callback=None, cache_manager: Optional[CacheManager] = None):
    """
    Perform a DuckDuckGo text search and download pages asynchronously.
    Returns a list of dicts with 'url', 'file_path', and optionally 'content_type'.
    """
    # Sanitize the output directory
    output_dir = sanitize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    results_info = []
    if not keyword.strip():
        logger.warning("Empty keyword provided to DuckDuckGo search; skipping search.") # <<< Use logger
        return []
    
    with DDGS() as ddgs:
        results = ddgs.text(keyword, max_results=limit)
    if not results:
        logger.warning(f"No results found for '{keyword}'.") # <<< Use logger
        return []
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for idx, result in enumerate(results):
            url = result.get("href")
            if not url:
                continue
            # Determine file extension from URL
            ext = ".html"
            if ".pdf" in url.lower():
                ext = ".pdf"
            # Limit the sanitized keyword length for the filename
            short_keyword = sanitize_filename(keyword)[:50]  # up to 50 chars
            filename = f"{short_keyword}_{idx}{ext}"
            file_path = os.path.join(output_dir, filename)
            # Pass progress_callback and cache_manager to download_page
            tasks.append(download_page(session, url, headers, timeout, file_path, progress_callback, cache_manager))
        pages = await asyncio.gather(*tasks)
        for page in pages:
            if page:
                results_info.append(page)
    return results_info

async def download_webpages_searxng(
    keyword,
    config, # Accept the full config dictionary
    output_dir='downloaded_webpages',
    progress_callback=None,
    cache_manager: Optional[CacheManager] = None, # Added cache_manager
    pageno=1 # Keep pageno as it's query-specific
):
    """
    Perform a SearXNG search using its API directly and download pages asynchronously.
    Supports additional parameters like language, time_range, safesearch, etc., read from config.
    Returns a list of dicts with 'url', 'file_path', and optionally 'content_type'.
    """
    # --- Extract SearXNG settings from config ---
    searxng_config = config.get('search', {}).get('searxng', {})
    base_url = searxng_config.get('base_url')
    limit = searxng_config.get('max_results', 5) # Get limit from config
    language = searxng_config.get('language')
    safesearch = searxng_config.get('safesearch') # Can be None or 0, 1, 2
    time_range = searxng_config.get('time_range')
    categories = searxng_config.get('categories')
    engines = searxng_config.get('engines')
    enabled_plugins = searxng_config.get('enabled_plugins') # Read enabled plugins
    disabled_plugins = searxng_config.get('disabled_plugins') # Read disabled plugins
    # api_key = searxng_config.get('api_key') # If you needed API key auth

    if not base_url:
        logger.error("SearXNG base_url is not configured in config.yaml.") # <<< Use logger
        if progress_callback:
            progress_callback("[ERROR] SearXNG base_url is not configured.")
        return []

    # Sanitize the output directory
    output_dir = sanitize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results_info = []
    if not keyword.strip():
        logger.warning("Empty keyword provided to SearXNG search; skipping search.") # <<< Use logger
        return []

    # --- Direct API Call Implementation ---
    params = {
        'q': keyword,
        'format': 'json',
        'pageno': pageno,
    }
    # Add optional parameters if they are provided and not None/empty
    if language:
        params['language'] = language
    if time_range:
        params['time_range'] = time_range
    if safesearch is not None: # Check for None explicitly as 0 is a valid value
        params['safesearch'] = safesearch
    if categories:
        params['categories'] = categories
    if engines and isinstance(engines, list) and len(engines) > 0:
        # Join the list of engine keys into a comma-separated string
        params['engines'] = ','.join(engines)
    elif engines and isinstance(engines, str): # Keep backward compatibility if it's already a string
         params['engines'] = engines
    # Add plugins if they are configured and are lists
    if enabled_plugins and isinstance(enabled_plugins, list) and len(enabled_plugins) > 0:
        params['enabled_plugins'] = ','.join(enabled_plugins)
    if disabled_plugins and isinstance(disabled_plugins, list) and len(disabled_plugins) > 0:
        params['disabled_plugins'] = ','.join(disabled_plugins)

    # Note: 'limit' (from max_results) is not a direct SearXNG API param for the query itself,
    # but we'll use it later to limit the number of downloads.
    # SearXNG might return more results than 'limit' based on its internal pagination.

    query_string = urlencode(params, quote_via=quote_plus)
    search_url = f"{base_url.rstrip('/')}/search?{query_string}"

    logger.info(f"Querying SearXNG API: {search_url}") # <<< Use logger
    if progress_callback:
        progress_callback(f"Querying SearXNG API for: '{keyword[:50]}...' (Page {pageno})")

    search_results_list = []
    try:
        timeout = aiohttp.ClientTimeout(total=60) # Increased timeout for API query to 60s
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(search_url, headers={'Accept': 'application/json'}) as response:
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = await response.json()

                # Extract results - adjust keys based on actual SearXNG JSON response structure
                # Common keys are 'results', 'answers', 'infoboxes'. We primarily want 'results'.
                if 'results' in data and isinstance(data['results'], list):
                    search_results_list = data['results']
                else:
                    logger.warning(f"SearXNG JSON response missing 'results' list or not a list. URL: {search_url}") # <<< Use logger
                    logger.debug(f"Response data: {data}") # <<< Use logger
                if not search_results_list:
                    logger.warning(f"No results found via SearXNG API for '{keyword}'. URL: {search_url}") # <<< Use logger
                    if progress_callback:
                        progress_callback(f"[WARN] No SearXNG results for: '{keyword[:50]}...' (Page {pageno})")
                    return []

    except aiohttp.ClientError as e:
        error_message = f"Network error querying SearXNG API '{search_url}': {e}"
        logger.error(error_message, exc_info=True) # <<< Use logger with traceback
        if progress_callback:
            progress_callback(f"[ERROR] Network error querying SearXNG.")
        return []
    except json.JSONDecodeError as e:
        error_message = f"Error decoding JSON response from SearXNG API '{search_url}': {e}"
        logger.error(error_message, exc_info=True) # <<< Use logger with traceback
        if progress_callback:
            progress_callback(f"[ERROR] Invalid JSON from SearXNG.")
        return []
    except Exception as e:
        # Catch other potential errors (e.g., connection issues, unexpected response structure)
        error_message = f"Error during SearXNG API query for '{keyword}': {repr(e)}"
        logger.error(error_message, exc_info=True) # <<< Use logger with traceback
        if progress_callback:
            progress_callback(f"[ERROR] SearXNG query failed: {repr(e)}") # Also add repr(e) here
        return []

    # --- Download Content of Found URLs ---
    if not search_results_list:
        return [] # Return early if API query failed or yielded no results

    headers = {'User-Agent': 'Mozilla/5.0'}
    download_timeout = aiohttp.ClientTimeout(total=10) # Keep original timeout for downloads
    async with aiohttp.ClientSession(timeout=download_timeout) as session:
        tasks = []
        count = 0
        # Iterate through results up to the specified limit
        for result in search_results_list:
            if count >= limit:
                break # Stop processing once we reach the desired number of downloads

            # Extract URL - common keys are 'url', 'link'. Prioritize 'url'.
            url = result.get("url") or result.get("link")
            if not url:
                logger.warning(f"SearXNG result missing 'url' or 'link': {result}") # <<< Use logger
                continue

            # Determine file extension from URL
            ext = ".html"
            parsed_url = urlparse(url)
            if parsed_url.path.lower().endswith('.pdf'):
                ext = ".pdf"
            # Consider other potential file types if needed (e.g., .txt, .docx)

            short_keyword = sanitize_filename(keyword)[:50]
            # Include page number in filename if not page 1 to avoid overwrites
            page_suffix = f"_p{pageno}" if pageno > 1 else ""
            filename = f"{short_keyword}{page_suffix}_{count}{ext}"
            file_path = os.path.join(output_dir, filename)

            # Pass progress_callback and cache_manager to download_page
            tasks.append(download_page(session, url, headers, download_timeout, file_path, progress_callback, cache_manager))
            count += 1 # Increment count based on successfully processed results with a URL

        if not tasks:
            logger.info("No valid URLs found in SearXNG results to download.") # <<< Use logger
            return []

        pages = await asyncio.gather(*tasks)
        for page in pages:
            if page:
                results_info.append(page)

    return results_info


def parse_pdf_to_text(pdf_file_path, max_pages=10):
    """
    Extract text from a PDF using PyMuPDF.
    If text is found (even partially), return it.
    Otherwise, convert up to max_pages pages to images and save them.
    """
    try:
        doc = fitz.open(pdf_file_path)
        text = ""
        for i in range(min(max_pages, doc.page_count)):
            page = doc.load_page(i)
            page_text = page.get_text().strip()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            logger.info(f"Extracted text from PDF: {pdf_file_path}") # <<< Use logger
            return text
        else:
            logger.info(f"No text found in PDF: {pdf_file_path}, converting pages to images") # <<< Use logger
            for i in range(min(max_pages, doc.page_count)):
                page = doc.load_page(i)
                pix = page.get_pixmap()
                image_file = pdf_file_path.replace('.pdf', f'_page_{i+1}.png')
                pix.save(image_file)
                logger.info(f"Saved page {i+1} as image: {image_file}") # <<< Use logger
            return ""
    except Exception as e:
        logger.warning(f"Failed to parse PDF {pdf_file_path}: {e}", exc_info=True) # <<< Use logger with traceback
        return ""

def parse_html_to_text(file_path, max_pdf_pages=10):
    """
    If the file is HTML, parse it and return its plain text.
    If it's a PDF, attempt to extract text with PyMuPDF.
    If the PDF has little or no text, convert up to max_pdf_pages to images.
    """
    try:
        if file_path.lower().endswith('.pdf'):
            return parse_pdf_to_text(file_path, max_pages=max_pdf_pages)
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            html_data = f.read()
        soup = BeautifulSoup(html_data, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logger.warning(f"Failed to parse HTML {file_path}: {e}", exc_info=True) # <<< Use logger with traceback
        return ""

def group_web_results_by_domain(web_results):
    """
    Takes a list of dicts, each with 'url', 'file_path', 'content_type', and groups them by domain.
    """
    grouped = {}
    for item in web_results:
        url = item.get('url')
        if not url:
            continue
        domain = urlparse(url).netloc
        grouped.setdefault(domain, []).append(item)
    return grouped
