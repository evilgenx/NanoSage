# llm_providers/base.py
import time
import google.api_core.exceptions
from requests.exceptions import HTTPError
import logging # <<< Import logging

logger = logging.getLogger(__name__) # <<< Get logger

# --- Backoff Handler for Rate Limits ---

def handle_rate_limit(details):
    """Custom handler for backoff to specifically wait on 429 errors."""
    exc_type, exc_value, _ = details['exception']
    wait_time = 30 # Seconds to wait for 429 errors (Increased from 15)

    is_rate_limit_error = False
    if isinstance(exc_value, google.api_core.exceptions.ResourceExhausted):
        logger.warning(f"Gemini API rate limit hit (ResourceExhausted). Waiting {wait_time}s before retry {details['tries']}...") # <<< Use logger
        is_rate_limit_error = True
    elif isinstance(exc_value, HTTPError) and exc_value.response.status_code == 429:
        logger.warning(f"OpenRouter API rate limit hit (429). Waiting {wait_time}s before retry {details['tries']}...") # <<< Use logger
        is_rate_limit_error = True

    if is_rate_limit_error:
        time.sleep(wait_time)
    else:
        # For other errors, let backoff use its default exponential delay
        # We can still log them if needed
        logger.warning(f"Retrying after error ({exc_type.__name__}): {exc_value}. Attempt {details['tries']}.") # <<< Use logger
        # No extra sleep needed here, backoff handles the delay
