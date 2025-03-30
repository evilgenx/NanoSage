# llm_utils.py

import os
import time
import re
import requests # Added for OpenRouter
import json # Added for OpenRouter payload
import google.generativeai as genai
import google.api_core.exceptions # Added for specific Gemini error handling
from requests.exceptions import HTTPError # Added for specific OpenRouter error handling
# from google.generativeai.types import EmbedContentResponse # Removed problematic import
from ollama import chat, ChatResponse
import backoff # For retry logic on API calls
import time # For sleep in retry

# --- Backoff Handler for Rate Limits ---

def handle_rate_limit(details):
    """Custom handler for backoff to specifically wait on 429 errors."""
    exc_type, exc_value, _ = details['exception']
    wait_time = 30 # Seconds to wait for 429 errors (Increased from 15)

    is_rate_limit_error = False
    if isinstance(exc_value, google.api_core.exceptions.ResourceExhausted):
        print(f"[WARN] Gemini API rate limit hit (ResourceExhausted). Waiting {wait_time}s before retry {details['tries']}...")
        is_rate_limit_error = True
    elif isinstance(exc_value, HTTPError) and exc_value.response.status_code == 429:
        print(f"[WARN] OpenRouter API rate limit hit (429). Waiting {wait_time}s before retry {details['tries']}...")
        is_rate_limit_error = True

    if is_rate_limit_error:
        time.sleep(wait_time)
    else:
        # For other errors, let backoff use its default exponential delay
        # We can still log them if needed
        print(f"[WARN] Retrying after error ({exc_type.__name__}): {exc_value}. Attempt {details['tries']}.")
        # No extra sleep needed here, backoff handles the delay

# --- Model Listing Functions ---

# Function to list available Gemini models supporting generateContent
def list_gemini_models():
    """Lists available Gemini models that support generateContent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not set.")
        return None # Indicate failure due to missing key
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Filter for models supporting 'generateContent'
        supported_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        if not supported_models:
            print("[WARN] No Gemini models found supporting 'generateContent'.")
            return []
        return supported_models
    except Exception as e:
        print(f"[ERROR] Failed to list Gemini models: {e}")
        return None # Indicate failure

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


# --- Embedding Model Listing ---

def list_gemini_embedding_models():
    """Lists available Gemini models that support embedContent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not set.")
        return None
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Filter for models supporting 'embedContent'
        supported_models = [m.name for m in models if 'embedContent' in m.supported_generation_methods]
        if not supported_models:
            print("[WARN] No Gemini models found supporting 'embedContent'.")
            return []
        # Often, there's a primary one like 'models/embedding-001'
        # Let's prioritize that if found, otherwise return all supported
        preferred_model = "models/embedding-001"
        if preferred_model in supported_models:
            return [preferred_model] + [m for m in supported_models if m != preferred_model]
        return supported_models
    except Exception as e:
        print(f"[ERROR] Failed to list Gemini embedding models: {e}")
        return None

def list_openrouter_embedding_models():
    """Lists available and free embedding models from OpenRouter."""
    # No API key needed for listing models as of early 2024
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            print("[WARN] Unexpected response format from OpenRouter /models endpoint.")
            return None

        free_embedding_models = []
        # Common embedding model IDs often proxied by OpenRouter (check their docs for updates)
        # We also check the pricing info like before.
        known_embedding_prefixes = [
            "openai/text-embedding-",
            "sentence-transformers/",
            "cohere/embed-",
            # Add others as needed based on OpenRouter offerings
        ]

        for model_info in data["data"]:
            model_id = model_info.get("id")
            if not model_id:
                continue

            # Check if it's likely an embedding model by ID prefix
            is_embedding_type = any(model_id.startswith(prefix) for prefix in known_embedding_prefixes)
            # Add specific known ones if prefixes aren't enough
            # if model_id == "jinaai/jina-embeddings-v2-base-en": is_embedding_type = True

            if not is_embedding_type:
                continue # Skip non-embedding models

            # Check pricing
            pricing = model_info.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", "1")) # Embedding cost often listed under 'prompt'
            completion_cost = float(pricing.get("completion", "1")) # Usually 0 for embeddings
            is_free = prompt_cost == 0.0 and completion_cost == 0.0

            if is_free:
                free_embedding_models.append(model_id)

        if not free_embedding_models:
            print("[WARN] No free embedding models found on OpenRouter (based on pricing and known IDs).")
            return []

        return sorted(free_embedding_models)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch OpenRouter models: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to parse OpenRouter models response: {e}")
        return None


# --- Generative Model Calling Functions ---

def call_gemma(prompt, model="gemma2:2b", personality=None):
    system_message = ""
    if personality:
        system_message = f"You are a {personality} assistant.\n\n"
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    response: ChatResponse = chat(model=model, messages=messages)
    return response.message.content

# --- OpenRouter Integration ---
# Add backoff decorator for OpenRouter chat completions
@backoff.on_exception(backoff.expo,
                      HTTPError,
                      max_tries=8, # Increased from 5
                      on_backoff=handle_rate_limit) # Use custom handler for 429s
@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=3) # Retry basic connection errors quickly
def call_openrouter(prompt, model="openai/gpt-3.5-turbo", personality=None):
    """Calls the OpenRouter API with a specified model, with retry logic."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY environment variable not set.")
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


# --- Existing Functions Modified for OpenRouter ---

# Updated retry logic for Gemini: specific handling for ResourceExhausted
@backoff.on_exception(backoff.expo,
                      google.api_core.exceptions.ResourceExhausted,
                      max_tries=8, # Increased from 5
                      on_backoff=handle_rate_limit) # Use custom handler
@backoff.on_exception(backoff.expo,
                      (google.api_core.exceptions.DeadlineExceeded,
                       google.api_core.exceptions.ServiceUnavailable), # Other retryable Gemini errors
                      max_tries=3)
# Modified to require model_name
def call_gemini(prompt, model_name):
    """Calls the Gemini API with a specified model, with retry logic."""
    if not model_name:
        print("[ERROR] No Gemini model specified for the API call.")
        return "Error: No Gemini model specified."
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not set.")
        return "Error: Gemini API key not configured."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # Handle potential safety blocks or empty responses
        if not response.parts:
             print(f"[WARN] Gemini response blocked or empty. Reason: {response.prompt_feedback}")
             # Consider returning a more informative error or fallback
             return f"Error: Gemini response blocked or empty. Reason: {response.prompt_feedback}"
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        print(f"[ERROR] Gemini API call failed: {e}")
        # Re-raise the exception for backoff to catch it
        raise e


# --- Embedding Model Calling Functions ---

# Updated retry logic for Gemini embeddings
@backoff.on_exception(backoff.expo,
                      google.api_core.exceptions.ResourceExhausted,
                      max_tries=8, # Increased from 5
                      on_backoff=handle_rate_limit) # Use custom handler
@backoff.on_exception(backoff.expo,
                      (google.api_core.exceptions.DeadlineExceeded,
                       google.api_core.exceptions.ServiceUnavailable), # Other retryable Gemini errors
                      max_tries=3)
def call_gemini_embedding(text: str, model_name: str = "models/embedding-001") -> list[float] | None:
    """Calls the Gemini API to get embeddings for a text, with retry logic."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not set for embedding.")
        return None
    try:
        # Ensure genai is configured (might be redundant if done elsewhere, but safe)
        genai.configure(api_key=api_key)
        # Use embed_content for embedding task
        # Removed type hint: EmbedContentResponse
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_document" # Or "retrieval_query" if embedding a query
        )
        # Check if embedding is present
        if 'embedding' in result:
            return result['embedding']
        else:
            print(f"[WARN] Gemini embedding response missing 'embedding' field for model {model_name}.")
            return None
    except Exception as e:
        print(f"[ERROR] Gemini embedding call failed for model {model_name}: {e}")
        # Re-raise the exception for backoff to catch it
        raise e # Re-raise for backoff

# Updated retry logic for OpenRouter embeddings
@backoff.on_exception(backoff.expo,
                      HTTPError,
                      max_tries=8, # Increased from 5
                      on_backoff=handle_rate_limit) # Use custom handler for 429s
@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=3) # Retry basic connection errors quickly
def call_openrouter_embedding(text: str, model_name: str) -> list[float] | None:
    """Calls the OpenRouter API to get embeddings for a text, with retry logic."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY environment variable not set for embedding.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Replace with your actual app URL if applicable
        "X-Title": "NanoSage", # Replace with your app name
    }
    data = {
        "model": model_name,
        "input": text,
        # Add encoding_format if needed, e.g., "encoding_format": "float"
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings", # Use the /embeddings endpoint
            headers=headers,
            data=json.dumps(data),
            timeout=60 # Timeout for embedding requests
        )
        response.raise_for_status()
        response_data = response.json()

        # Check response structure (usually contains a 'data' list with embedding objects)
        if "data" in response_data and isinstance(response_data["data"], list) and len(response_data["data"]) > 0:
            # Assuming the first result contains the embedding
            embedding_data = response_data["data"][0]
            if "embedding" in embedding_data and isinstance(embedding_data["embedding"], list):
                return embedding_data["embedding"]
            else:
                print(f"[WARN] OpenRouter embedding response structure unexpected (missing embedding list): {response_data}")
                return None
        elif "error" in response_data:
             error_message = response_data["error"].get("message", "Unknown error")
             print(f"[ERROR] OpenRouter embedding API error: {error_message}")
             # Raise specific error type if possible based on message
             raise requests.exceptions.RequestException(f"OpenRouter API Error: {error_message}")
        else:
            print(f"[WARN] OpenRouter embedding response missing expected 'data' or 'error': {response_data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] OpenRouter embedding API request failed: {e}")
        raise e # Re-raise for backoff
    except Exception as e:
        print(f"[ERROR] Failed to process OpenRouter embedding response: {e}")
        raise e # Re-raise for backoff


# --- Utility Functions ---

def extract_final_query(text):
    marker = "Final Enhanced Query:"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()

# Modified to accept rag_model, selected_gemini_model, and selected_openrouter_model
def chain_of_thought_query_enhancement(query, personality=None, rag_model="gemma", selected_gemini_model=None, selected_openrouter_model=None):
    prompt = (
        "You are an expert search strategist. Think step-by-step through the implications and nuances "
        "of the following query and produce a final, enhanced query that covers more angles.\n\n"
        f"Query: \"{query}\"\n\n"
        "After your reasoning, output only the final enhanced query on a single line - SHORT AND CONCISE.\n"
        "Provide your reasoning, and at the end output the line 'Final Enhanced Query:' followed by the enhanced query."
    )
    if rag_model == "gemini":
        print("[INFO] Using Gemini for query enhancement.")
        # Gemini doesn't use 'personality' in the same way, prompt needs to be self-contained
        # Pass the selected model name
        raw_output = call_gemini(prompt, model_name=selected_gemini_model)
    elif rag_model == "openrouter":
        print("[INFO] Using OpenRouter for query enhancement.")
        if not selected_openrouter_model:
             print("[WARN] No OpenRouter model specified for query enhancement, falling back to default.")
             # Fallback or return error - let's fallback to original query for now
             return query
        raw_output = call_openrouter(prompt, model=selected_openrouter_model, personality=personality)
    else: # Default to gemma/ollama
        print("[INFO] Using Gemma (Ollama) for query enhancement.")
        raw_output = call_gemma(prompt, personality=personality)

    if raw_output.startswith("Error:"): # Handle potential API errors
        print(f"[WARN] Query enhancement failed: {raw_output}. Falling back to original query.")
        return query # Fallback to original query if enhancement fails

    return extract_final_query(raw_output)

# Modified summarize_text to support different RAG models
def summarize_text(text, max_chars=6000, personality=None, rag_model="gemma", selected_gemini_model=None, selected_openrouter_model=None):
    """Summarizes text, potentially chunking, using the specified RAG model."""

    def _call_selected_model(prompt_text):
        """Internal helper to call the correct model based on rag_model."""
        if rag_model == "gemini":
            if not selected_gemini_model:
                print("[ERROR] Gemini selected for summarization, but no model specified.")
                return "Error: Gemini model not specified for summarization."
            return call_gemini(prompt_text, model_name=selected_gemini_model)
        elif rag_model == "openrouter":
            if not selected_openrouter_model:
                print("[ERROR] OpenRouter selected for summarization, but no model specified.")
                return "Error: OpenRouter model not specified for summarization."
            return call_openrouter(prompt_text, model=selected_openrouter_model, personality=personality)
        else: # Default to gemma/ollama
            if rag_model != "gemma":
                 print(f"[WARN] Unknown rag_model '{rag_model}' for summarization, defaulting to gemma.")
            return call_gemma(prompt_text, personality=personality)

    if len(text) <= max_chars:
        prompt = f"Please summarize the following text succinctly:\n\n{text}"
        return _call_selected_model(prompt)

    # If text is longer than max_chars, chunk it
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    summaries = []
    print(f"[INFO] Summarizing text in {len(chunks)} chunks using {rag_model}...")
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize part {i+1}/{len(chunks)}:\n\n{chunk}"
        summary = _call_selected_model(prompt)
        if summary.startswith("Error:"): # Propagate errors
            return summary
        summaries.append(summary)
        # Optional: Add delay only if needed, maybe make configurable
        # time.sleep(1)

    combined = "\n".join(summaries)
    if len(combined) > max_chars:
        print(f"[INFO] Combining {len(summaries)} summaries using {rag_model}...")
        prompt = f"Combine these summaries into one concise summary:\n\n{combined}"
        combined = _call_selected_model(prompt)

    return combined

# Added selected_gemini_model and selected_openrouter_model parameters
def rag_final_answer(aggregation_prompt, rag_model="gemma", personality=None, selected_gemini_model=None, selected_openrouter_model=None):
    print("[INFO] Performing final RAG generation using model:", rag_model)
    if rag_model == "gemma":
        return call_gemma(aggregation_prompt, personality=personality)
    elif rag_model == "pali":
        modified_prompt = f"PALI mode analysis:\n\n{aggregation_prompt}"
        return call_gemma(modified_prompt, personality=personality)
    elif rag_model == "gemini":
        # Note: Gemini doesn't have a direct 'personality' parameter like ollama's system message.
        # The prompt itself needs to guide the desired tone if needed.
        # Use the passed-in parameter
        return call_gemini(aggregation_prompt, model_name=selected_gemini_model)
    elif rag_model == "openrouter":
        print("[INFO] Using OpenRouter for final RAG generation.")
        if not selected_openrouter_model:
             print("[ERROR] No OpenRouter model specified for RAG.")
             return "Error: No OpenRouter model specified."
        # Pass the selected model name
        return call_openrouter(aggregation_prompt, model=selected_openrouter_model, personality=personality)
    else: # Default or unknown, fall back to gemma
        print(f"[WARN] Unknown rag_model '{rag_model}', defaulting to gemma.")
        return call_gemma(aggregation_prompt, personality=personality)

# Modified follow_up_conversation to support different RAG models
def follow_up_conversation(follow_up_prompt, personality=None, rag_model="gemma", selected_gemini_model=None, selected_openrouter_model=None):
    """Handles follow-up conversation prompts using the specified RAG model."""
    print(f"[INFO] Handling follow-up conversation using {rag_model}...")
    if rag_model == "gemini":
        if not selected_gemini_model:
            print("[ERROR] Gemini selected for follow-up, but no model specified.")
            return "Error: Gemini model not specified for follow-up."
        # Gemini doesn't use personality directly in the call
        return call_gemini(follow_up_prompt, model_name=selected_gemini_model)
    elif rag_model == "openrouter":
        if not selected_openrouter_model:
            print("[ERROR] OpenRouter selected for follow-up, but no model specified.")
            return "Error: OpenRouter model not specified for follow-up."
        return call_openrouter(follow_up_prompt, model=selected_openrouter_model, personality=personality)
    else: # Default to gemma/ollama
        if rag_model != "gemma":
            print(f"[WARN] Unknown rag_model '{rag_model}' for follow-up, defaulting to gemma.")
        return call_gemma(follow_up_prompt, personality=personality)


def clean_search_query(query):
    query = re.sub(r'[\*\_`]', '', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

def split_query(query, max_len=200):
    query = query.replace('"', '').replace("'", "")
    sentences = query.split('.')
    subqueries = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not any(c.isalnum() for c in sentence):
            continue
        if len(current) + len(sentence) + 1 <= max_len:
            current += (". " if current else "") + sentence
        else:
            subqueries.append(current)
            current = sentence
    if current:
        subqueries.append(current)
    return [sq for sq in subqueries if sq.strip()]
