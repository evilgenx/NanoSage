import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_USER_AGENT = "NanoSageScraper/1.0"

def can_fetch(url: str, user_agent: str = DEFAULT_USER_AGENT, respect_robots: bool = True) -> bool:
    """Checks if scraping the URL is allowed by robots.txt."""
    if not respect_robots:
        return True

    parsed_url = urlparse(url)
    robots_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", "/robots.txt")

    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        # Add a small delay to avoid overwhelming servers with robots.txt requests
        time.sleep(0.5)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        logging.warning(f"Could not fetch or parse robots.txt from {robots_url}: {e}")
        # Be cautious: if robots.txt is inaccessible or malformed, assume fetching is not allowed.
        # Alternatively, you could default to True, but that's less polite.
        return False

def scrape_url_to_markdown(
    url: str,
    respect_robots: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 15
) -> tuple[str | None, str | None]:
    """
    Scrapes a single URL, extracts main content, and converts it to Markdown.

    Args:
        url: The URL to scrape.
        respect_robots: Whether to check and obey robots.txt rules.
        user_agent: The User-Agent string to use for requests.
        timeout: Request timeout in seconds.

    Returns:
        A tuple containing (markdown_content, error_message).
        If successful, markdown_content is the extracted text as Markdown, and error_message is None.
        If failed, markdown_content is None, and error_message contains the reason.
    """
    if respect_robots:
        logging.info(f"Checking robots.txt for {url}")
        if not can_fetch(url, user_agent, respect_robots):
            error_msg = f"Scraping disallowed by robots.txt for {url}"
            logging.warning(error_msg)
            return None, error_msg

    headers = {'User-Agent': user_agent}
    logging.info(f"Attempting to fetch URL: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type - only process HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            error_msg = f"Skipping non-HTML content ({content_type}) at {url}"
            logging.warning(error_msg)
            return None, error_msg

        logging.info(f"Successfully fetched {url}. Parsing HTML...")
        soup = BeautifulSoup(response.content, 'lxml') # Use lxml for speed

        # --- Content Extraction Heuristic ---
        # Try common main content tags first
        main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main')

        # If specific tags aren't found, fall back to the body, but remove common noise
        if not main_content:
            main_content = soup.body
            if main_content:
                # Remove common non-content elements
                for tag in main_content.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style', 'form', 'button']):
                    tag.decompose()
            else:
                # If no body tag, something is very wrong
                 error_msg = f"Could not find <body> tag in HTML from {url}"
                 logging.error(error_msg)
                 return None, error_msg

        # Convert the selected HTML content to Markdown
        # Use options for cleaner output if needed (e.g., strip=['script', 'style'])
        markdown_content = md(str(main_content), heading_style="ATX", strip=['script', 'style'])

        if not markdown_content.strip():
             error_msg = f"Extracted content is empty after processing {url}"
             logging.warning(error_msg)
             return None, error_msg

        logging.info(f"Successfully extracted and converted content from {url} to Markdown.")
        return markdown_content.strip(), None

    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed for {url}: {e}"
        logging.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while processing {url}: {e}"
        logging.error(error_msg, exc_info=True) # Include traceback for unexpected errors
        return None, error_msg

if __name__ == '__main__':
    # Example Usage (for testing)
    test_url = "https://example.com" # Replace with a real URL for testing
    print(f"Attempting to scrape: {test_url}")
    markdown, error = scrape_url_to_markdown(test_url, respect_robots=True)

    if error:
        print(f"\nError: {error}")
    elif markdown:
        print("\n--- Scraped Markdown ---")
        print(markdown)
        print("------------------------")
    else:
        print("\nNo content returned and no specific error.")
