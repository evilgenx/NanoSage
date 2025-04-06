# llm_providers/gemini.py
import os
import google.generativeai as genai
import google.api_core.exceptions
import requests.exceptions # Needed for listing models error handling
import backoff
import logging # <<< Import logging

# Import the custom backoff handler
from .base import handle_rate_limit

logger = logging.getLogger(__name__) # <<< Get logger

# --- Model Listing Functions ---

# Function to list available Gemini models supporting generateContent
# Added gemini_api_key parameter
def list_gemini_models(gemini_api_key=None):
    """Lists available Gemini models that support generateContent."""
    # Prioritize passed key, fallback to environment variable
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key not provided via parameter or environment variable for listing models.") # <<< Use logger
        return None # Indicate failure due to missing key
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Filter for models supporting 'generateContent'
        supported_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        if not supported_models:
            logger.warning("No Gemini models found supporting 'generateContent'.") # <<< Use logger
            return []
        return supported_models
    except Exception as e:
        logger.error(f"Failed to list Gemini models: {e}", exc_info=True) # <<< Use logger with traceback
        return None # Indicate failure

# Removed list_gemini_embedding_models function

# --- Generative Model Calling Functions ---

# Updated retry logic for Gemini: specific handling for ResourceExhausted
@backoff.on_exception(backoff.expo,
                      google.api_core.exceptions.ResourceExhausted,
                      max_tries=8, # Increased from 5
                      on_backoff=handle_rate_limit) # Use custom handler
@backoff.on_exception(backoff.expo,
                      (google.api_core.exceptions.DeadlineExceeded,
                       google.api_core.exceptions.ServiceUnavailable), # Other retryable Gemini errors
                      max_tries=3)
# Modified to require model_name, added gemini_api_key parameter
def call_gemini(prompt, model_name, gemini_api_key=None):
    """Calls the Gemini API with a specified model, with retry logic."""
    if not model_name:
        logger.error("No Gemini model specified for the API call.") # <<< Use logger
        return "Error: No Gemini model specified."
    # Prioritize passed key, fallback to environment variable
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key not provided via parameter or environment variable.") # <<< Use logger
        return "Error: Gemini API key not configured."
    try:
        # Configure API key for this call (safe even if called multiple times)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        logger.debug(f"Calling Gemini model '{model_name}'...") # <<< Use logger
        response = model.generate_content(prompt)
        # Handle potential safety blocks or empty responses
        if not response.parts:
             logger.warning(f"Gemini response blocked or empty. Reason: {response.prompt_feedback}") # <<< Use logger
             # Consider returning a more informative error or fallback
             return f"Error: Gemini response blocked or empty. Reason: {response.prompt_feedback}"
        logger.debug(f"Gemini call successful for model '{model_name}'.") # <<< Use logger
        return response.text
    except Exception as e:
        logger.error(f"Gemini API call failed for model '{model_name}': {e}", exc_info=True) # <<< Use logger with traceback
        # Re-raise the exception for backoff to catch it
        raise e

# Removed call_gemini_embedding function
