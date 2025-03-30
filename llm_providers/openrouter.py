# llm_providers/openrouter.py
import os
import requests
import json
import backoff
from requests.exceptions import HTTPError, RequestException

# Import the custom backoff handler
from .base import handle_rate_limit

# --- Model Listing Functions ---

def list_openrouter_models():
    """Lists available and free models from OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    # No API key needed for listing models according to OpenRouter docs as of late 2023/early 2024
    # Re-add key check if required by API in the future.
    # if not api_key:
    #     print("[ERROR] OPENROUTER_API_KEY environment variable not set (needed for model listing).")
    #     return None

    headers = {
        # Add Authorization header if API key becomes required for listing
        # "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=30 # Timeout for listing models
        )
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            print("[WARN] Unexpected response format from OpenRouter /models endpoint.")
            return None

        # Filter for models often considered "free tier" (pricing is 0)
        # Note: OpenRouter's definition of "free" might change. This checks for $0 cost.
        free_models = []
        for model_info in data["data"]:
            pricing = model_info.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", "1")) # Default to non-zero if missing
            completion_cost = float(pricing.get("completion", "1")) # Default to non-zero if missing
            # Some models might have null price - treat as non-free unless explicitly 0
            is_free = prompt_cost == 0.0 and completion_cost == 0.0

            if is_free:
                # Include context length for potential future use/display
                context = model_info.get("context_length", "N/A")
                model_id = model_info.get("id")
                if model_id:
                    # Optionally format for display: f"{model_id} (Context: {context})"
                    free_models.append(model_id)

        if not free_models:
            print("[WARN] No free models found on OpenRouter (based on pricing info).")
            return [] # Return empty list, not None

        return sorted(free_models) # Return sorted list of model IDs

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch OpenRouter models: {e}")
        return None # Indicate failure
    except Exception as e:
        print(f"[ERROR] Failed to parse OpenRouter models response: {e}")
        return None # Indicate failure

# Removed list_openrouter_embedding_models function

# --- Generative Model Calling Functions ---

# Add backoff decorator for OpenRouter chat completions
@backoff.on_exception(backoff.expo,
                      HTTPError,
                      max_tries=8, # Increased from 5
                      on_backoff=handle_rate_limit) # Use custom handler for 429s
@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=3) # Retry basic connection errors quickly
# Added openrouter_api_key parameter
def call_openrouter(prompt, model="openai/gpt-3.5-turbo", personality=None, openrouter_api_key=None):
    """Calls the OpenRouter API with a specified model, with retry logic."""
    # Prioritize passed key, fallback to environment variable
    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OpenRouter API key not provided via parameter or environment variable.")
        return "Error: OpenRouter API key not configured."

    system_message_content = f"You are a helpful assistant."
    if personality:
         system_message_content = f"You are a {personality} assistant."

    messages = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": prompt}
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Recommended headers by OpenRouter
        "HTTP-Referer": "http://localhost", # Replace with your actual app URL if applicable
        "X-Title": "NanoSage", # Replace with your app name
    }

    data = {
        "model": model,
        "messages": messages,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=180 # Set a reasonable timeout (e.g., 3 minutes)
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()

        if "choices" in response_data and len(response_data["choices"]) > 0:
            # Check message structure (can vary slightly)
            if "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                 return response_data["choices"][0]["message"]["content"].strip()
            # Handle potential variations if needed
            # elif ... other possible structures ...
            else:
                 print(f"[WARN] Unexpected OpenRouter response structure: {response_data}")
                 return "Error: Unexpected response structure from OpenRouter."
        elif "error" in response_data:
             error_message = response_data["error"].get("message", "Unknown error")
             print(f"[ERROR] OpenRouter API error: {error_message}")
             return f"Error: OpenRouter API error - {error_message}"
        else:
            print(f"[WARN] OpenRouter response missing expected 'choices' or 'error': {response_data}")
            return "Error: Invalid response from OpenRouter."

    except HTTPError as e:
        # Re-raise HTTPError so backoff can catch it
        print(f"[ERROR] OpenRouter API HTTP error: {e.response.status_code} - {e}")
        raise e
    except requests.exceptions.RequestException as e:
        # Re-raise RequestException so backoff can catch it
        print(f"[ERROR] OpenRouter API request failed: {e}")
        raise e
    except Exception as e:
        # Catch other potential errors during processing
        print(f"[ERROR] Failed to process OpenRouter response: {e}")
        # Don't re-raise generic exceptions unless necessary for backoff
        return f"Error: Failed processing OpenRouter response. Details: {e}"

# Removed call_openrouter_embedding function
