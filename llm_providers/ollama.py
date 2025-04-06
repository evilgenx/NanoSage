# llm_providers/ollama.py
from ollama import chat, ChatResponse
import logging # <<< Import logging

logger = logging.getLogger(__name__) # <<< Get logger

# --- Generative Model Calling Functions ---

def call_gemma(prompt, model="gemma2:2b", personality=None):
    """Calls the local Ollama service for Gemma models."""
    system_message = ""
    if personality:
        system_message = f"You are a {personality} assistant.\n\n"
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    try:
        logger.debug(f"Calling Ollama model '{model}'...") # <<< Use logger
        response: ChatResponse = chat(model=model, messages=messages)
        logger.debug(f"Ollama call successful for model '{model}'.") # <<< Use logger
        return response.message.content
    except Exception as e:
        logger.error(f"Ollama call failed for model {model}: {e}", exc_info=True) # <<< Use logger with traceback
        # Depending on desired behavior, could return an error string or raise
        return f"Error: Ollama call failed - {e}"
