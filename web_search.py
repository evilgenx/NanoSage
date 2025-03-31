import os
import os
import asyncio
from urllib.parse import urlparse, quote_plus # Added quote_plus
import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import fitz  # PyMuPDF
import json # Added json import
from langchain_community.utilities import SearxSearchWrapper # Added LangChain import

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

# Added progress_callback parameter
async def download_page(session, url, headers, timeout, file_path, progress_callback):
    try:
        async with session.get(url, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            # If it's a PDF or image (or other binary content), read as binary.
            if ('application/pdf' in content_type) or file_path.lower().endswith('.pdf') or \
               ('image/' in content_type):
                content = await response.read()
                mode = 'wb'
                open_kwargs = {}
            else:
                content = await response.text()
                mode = 'w'
                open_kwargs = {'encoding': 'utf-8'}  # write text as UTF-8 to avoid charmap errors
            with open(file_path, mode, **open_kwargs) as f:
                f.write(content)
            print(f"[INFO] Saved '{url}' -> '{file_path}'")
            return {'url': url, 'file_path': file_path, 'content_type': content_type}
    except Exception as e:
        error_message = f"Couldn't fetch '{url}': {e}"
        print(f"[WARN] {error_message}")
        # Call the progress callback with the warning
        if progress_callback:
            progress_callback(f"[WARN] {error_message}")
        return None

# Added progress_callback parameter
async def download_webpages_ddg(keyword, limit=5, output_dir='downloaded_webpages', progress_callback=None):
    """
    Perform a DuckDuckGo text search and download pages asynchronously.
    Returns a list of dicts with 'url', 'file_path', and optionally 'content_type'.
    """
    # Sanitize the output directory
    output_dir = sanitize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    results_info = []
    if not keyword.strip():
        print("[WARN] Empty keyword provided to DuckDuckGo search; skipping search.")
        return []
    
    with DDGS() as ddgs:
        results = ddgs.text(keyword, max_results=limit)
    if not results:
        print(f"[WARN] No results found for '{keyword}'.")
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
            # Pass progress_callback to download_page
            tasks.append(download_page(session, url, headers, timeout, file_path, progress_callback))
        pages = await asyncio.gather(*tasks)
        for page in pages:
            if page:
                results_info.append(page)
    return results_info

async def download_webpages_searxng(keyword, limit=5, base_url="http://127.0.0.1:8080", output_dir='downloaded_webpages', progress_callback=None):
    """
    Perform a SearXNG search and download pages asynchronously.
    Returns a list of dicts with 'url', 'file_path', and optionally 'content_type'.
    """
    if not base_url:
        print("[ERROR] SearXNG base_url is not configured.")
        if progress_callback:
            progress_callback("[ERROR] SearXNG base_url is not configured.")
        return []

    # Sanitize the output directory
    output_dir = sanitize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results_info = []
    if not keyword.strip():
        print("[WARN] Empty keyword provided to SearXNG search; skipping search.")
        return []

    # Construct the SearXNG query URL
    encoded_query = quote_plus(keyword)
    search_url = f"{base_url.rstrip('/')}/search?q={encoded_query}&format=json"
    print(f"[INFO] Querying SearXNG via LangChain wrapper: {base_url}")
    if progress_callback:
        progress_callback(f"Querying SearXNG (LangChain) for: '{keyword[:50]}...'")

    try:
        # Use LangChain wrapper to get search results list
        wrapper = SearxSearchWrapper(searx_host=base_url)
        # Note: The wrapper runs synchronously. Consider running in an executor if performance becomes an issue.
        # The .results() method returns more info like title and snippet if needed later.
        searx_results_list = wrapper.results(keyword, num_results=limit)

        if not searx_results_list:
            print(f"[WARN] No results found via SearXNG wrapper for '{keyword}'. Host: {base_url}")
            if progress_callback:
                progress_callback(f"[WARN] No SearXNG results for: '{keyword[:50]}...'")
            return []

        # Now, asynchronously download the content of the found URLs
        headers = {'User-Agent': 'Mozilla/5.0'}
        timeout = aiohttp.ClientTimeout(total=10) # Keep timeout for downloads
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            count = 0
            for result in searx_results_list:
                # The wrapper returns 'link' for the URL
                url = result.get("link")
                if not url:
                    print(f"[WARN] SearXNG wrapper result missing 'link': {result}")
                    continue

                # Determine file extension from URL (same logic as before)
                ext = ".html"
                if ".pdf" in url.lower():
                    ext = ".pdf"
                short_keyword = sanitize_filename(keyword)[:50]
                filename = f"{short_keyword}_{count}{ext}"
                file_path = os.path.join(output_dir, filename)

                # Pass progress_callback to download_page
                tasks.append(download_page(session, url, headers, timeout, file_path, progress_callback))
                count += 1 # Increment count based on successfully processed results

            pages = await asyncio.gather(*tasks)
            for page in pages:
                if page:
                    results_info.append(page)

    # Catch potential errors from the SearxSearchWrapper or aiohttp downloads
    except ImportError:
         error_message = "LangChain community package not found. Please install it: pip install langchain-community"
         print(f"[ERROR] {error_message}")
         if progress_callback:
             progress_callback(f"[ERROR] {error_message}")
    except Exception as e:
        # Catch errors from wrapper (e.g., connection issues) or download phase
        error_message = f"Error during SearXNG search/download for '{keyword}': {e}"
        print(f"[ERROR] {error_message}")
        if progress_callback:
            progress_callback(f"[ERROR] {error_message}")

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
            print(f"[INFO] Extracted text from PDF: {pdf_file_path}")
            return text
        else:
            print(f"[INFO] No text found in PDF: {pdf_file_path}, converting pages to images")
            for i in range(min(max_pages, doc.page_count)):
                page = doc.load_page(i)
                pix = page.get_pixmap()
                image_file = pdf_file_path.replace('.pdf', f'_page_{i+1}.png')
                pix.save(image_file)
                print(f"[INFO] Saved page {i+1} as image: {image_file}")
            return ""
    except Exception as e:
        print(f"[WARN] Failed to parse PDF {pdf_file_path}: {e}")
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
        print(f"[WARN] Failed to parse HTML {file_path}: {e}")
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
