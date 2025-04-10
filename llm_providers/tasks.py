# llm_providers/tasks.py
import time # Needed for summarize_text chunking delay (optional)
import logging # Added logging
from typing import Optional, Dict, Any # Added Optional, Dict, Any
from cache_manager import CacheManager # Added CacheManager import

# Import functions from the new provider modules and utils
from .gemini import call_gemini
from .openrouter import call_openrouter
from .ollama import call_gemma
from .utils import extract_final_query

logger = logging.getLogger(__name__) # Added logger

# Modified to accept llm_config dictionary
def chain_of_thought_query_enhancement(query, llm_config: dict = {}):
    """Enhances a query using a step-by-step thinking process with the specified LLM."""
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id") # Specific model ID for the provider
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality")

    prompt = (
        "You are an expert search strategist. Think step-by-step through the implications and nuances "
        "of the following query and produce a final, enhanced query that covers more angles.\n\n"
        f"Query: \"{query}\"\n\n"
        "After your reasoning, output only the final enhanced query on a single line - SHORT AND CONCISE.\n"
        "Provide your reasoning, and at the end output the line 'Final Enhanced Query:' followed by the enhanced query."
    )
    raw_output = ""
    logger.info(f"Attempting query enhancement using {provider}...") # <<< Use logger
    try:
        if provider == "gemini":
            # logger.info("Using Gemini for query enhancement.") # Redundant with above
            if not model_id:
                logger.error("Gemini selected for query enhancement, but no model specified.") # <<< Use logger
                return query # Fallback
            raw_output = call_gemini(prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            # logger.info("Using OpenRouter for query enhancement.") # Redundant with above
            if not model_id:
                 logger.warning("No OpenRouter model specified for query enhancement, falling back to default.") # <<< Use logger
                 return query # Fallback
            raw_output = call_openrouter(prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default to gemma/ollama
            if provider != "gemma":
                logger.warning(f"Unknown provider '{provider}' for query enhancement, defaulting to gemma.") # <<< Use logger
            # logger.info("Using Gemma (Ollama) for query enhancement.") # Redundant with above
            # Pass model_id to gemma if provided, otherwise it uses its default
            gemma_model = model_id or "gemma2:2b" # Example default if not specified
            raw_output = call_gemma(prompt, model=gemma_model, personality=personality)

        if not raw_output or raw_output.startswith("Error:"): # Handle potential API errors or empty output
            logger.warning(f"Query enhancement failed or returned error: {raw_output}. Falling back to original query.") # <<< Use logger
            return query # Fallback to original query if enhancement fails

        return extract_final_query(raw_output)

    except Exception as e:
        logger.error(f"Exception during query enhancement with {provider}: {e}. Falling back to original query.", exc_info=True) # <<< Use logger with traceback
        # Log the exception details if needed
        return query # Fallback in case of unexpected errors during the call

def extract_topics_from_text(text: str, llm_config: dict = {}, max_topics=5) -> str:
    """
    Extracts key topics or phrases from a given text using the specified LLM.

    Args:
        text (str): The input text to analyze.
        llm_config (dict): Configuration for the LLM provider (provider, model_id, api_key, personality).
        max_topics (int): The maximum number of topics/phrases to extract.

    Returns:
        str: A string containing the extracted topics/phrases, likely separated by newlines, or an error message.
    """
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality")

    prompt = (
        f"Analyze the following text and identify the main topics or key phrases. "
        f"Extract up to {max_topics} distinct topics/phrases that best represent the core subject matter. "
        f"Focus on nouns, noun phrases, or concepts discussed.\n\n"
        f"Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
        f"Output ONLY the topics/phrases, one per line. Do not include numbering, bullets, explanations, or any other text."
    )

    raw_output = ""
    logger.info(f"Extracting topics from text using {provider}...") # <<< Use logger
    try:
        if provider == "gemini":
            if not model_id:
                logger.error("Gemini selected for topic extraction, but no model specified.") # <<< Use logger
                return "Error: Gemini model not specified."
            raw_output = call_gemini(prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            if not model_id:
                logger.error("OpenRouter selected for topic extraction, but no model specified.") # <<< Use logger
                return "Error: OpenRouter model not specified."
            raw_output = call_openrouter(prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default to gemma/ollama
            if provider != "gemma":
                logger.warning(f"Unknown provider '{provider}' for topic extraction, defaulting to gemma.") # <<< Use logger
            gemma_model = model_id or "gemma2:2b" # Example default
            raw_output = call_gemma(prompt, model=gemma_model, personality=personality)

        if not raw_output or raw_output.startswith("Error:"):
            error_msg = raw_output if raw_output else "Error: Topic extraction failed (empty response)."
            logger.warning(f"Topic extraction failed or returned error: {error_msg}") # <<< Use logger
            return error_msg # Return the error message

        # Return the raw output, assuming the LLM followed the prompt (one topic per line)
        # Further parsing could be added here if needed (e.g., stripping extra whitespace)
        return raw_output.strip()

    except Exception as e:
        logger.error(f"Exception during topic extraction ({provider}): {e}", exc_info=True) # <<< Use logger with traceback
        return f"Error: Failed to extract topics with {provider} - {e}"


# Modified summarize_text to accept llm_config dictionary and cache_manager
def summarize_text(text: str, llm_config: Dict[str, Any] = {}, max_chars: int = 6000, cache_manager: Optional[CacheManager] = None) -> str:
    """Summarizes text, potentially chunking, using the specified RAG model configured in llm_config, with caching."""

    if not text or not text.strip():
        logger.warning("Attempted to summarize empty text.")
        return "" # Return empty string for empty input

    # --- Prepare Cacheable Config ---
    # Extract relevant keys that affect summary output for hashing
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    personality = llm_config.get("personality")
    # Add other relevant keys if they influence the summary prompt/output significantly
    cacheable_config = {
        'provider': provider,
        'model_id': model_id,
        'personality': personality,
        'task': 'summarize' # Add task type to differentiate from other potential uses
    }
    # --- End Prepare Cacheable Config ---

    # --- Cache Check for Full Text ---
    if cache_manager:
        cached_summary = cache_manager.get_summary(text, cacheable_config)
        if cached_summary is not None:
            return cached_summary
    # --- End Cache Check for Full Text ---

    # --- Internal LLM Call Helper ---
    def _call_selected_model(prompt_text: str) -> str:
        """Internal helper to call the correct model based on pre-extracted config."""
        # Note: This helper doesn't handle caching itself, caching is done around its calls.
        try:
            if provider == "gemini":
                if not model_id:
                    logger.error("Gemini selected for summarization, but no model specified.")
                    return "Error: Gemini model not specified for summarization."
                # Assuming call_gemini takes api_key from llm_config if needed
                return call_gemini(prompt_text, model_name=model_id, gemini_api_key=llm_config.get("api_key"))
            elif provider == "openrouter":
                if not model_id:
                    logger.error("OpenRouter selected for summarization, but no model specified.")
                    return "Error: OpenRouter model not specified for summarization."
                return call_openrouter(prompt_text, model=model_id, personality=personality, openrouter_api_key=llm_config.get("api_key"))
            else: # Default to gemma/ollama
                if provider != "gemma":
                     logger.warning(f"Unknown provider '{provider}' for summarization, defaulting to gemma.")
                gemma_model = model_id or "gemma2:2b" # Example default
                return call_gemma(prompt_text, model=gemma_model, personality=personality)
        except Exception as e:
            logger.error(f"Exception during _call_selected_model ({provider}): {e}", exc_info=True)
            return f"Error: Failed to call {provider} model - {e}"
    # --- End Internal LLM Call Helper ---


    # --- Handle Text <= Max Chars ---
    if len(text) <= max_chars:
        logger.debug(f"Summarizing text (<= {max_chars} chars) using {provider}...")
        # Changed prompt to ask for more detail and key points
        prompt = (
            f"Analyze the following text and provide a detailed summary. "
            f"Extract the key information, main arguments, and supporting details. "
            f"Present the summary in a comprehensive paragraph format.\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\""
        )
        summary = _call_selected_model(prompt)

        # Store in cache if successful
        if summary and not summary.startswith("Error:") and cache_manager:
            cache_manager.store_summary(text, cacheable_config, summary)

        return summary if summary else "Error: Summarization failed (empty response)."
    # --- End Handle Text <= Max Chars ---


    # --- Handle Text > Max Chars (Chunking) ---
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    summaries = []
    logger.info(f"Summarizing text in {len(chunks)} chunks using {provider}...")

    # Prepare cache config specific to chunk summarization (might differ slightly if needed)
    chunk_cacheable_config = cacheable_config.copy()
    chunk_cacheable_config['task'] = 'summarize_chunk'

    for i, chunk in enumerate(chunks):
        # --- Cache Check for Chunk ---
        cached_chunk_summary = None
        if cache_manager:
            cached_chunk_summary = cache_manager.get_summary(chunk, chunk_cacheable_config)

        if cached_chunk_summary is not None:
            summary = cached_chunk_summary
            logger.debug(f"Cache hit for summary chunk {i+1}/{len(chunks)}")
        else:
            # --- Generate Chunk Summary (Cache Miss) ---
            logger.debug(f"Summarizing chunk {i+1}/{len(chunks)} (cache miss)...")
            # Changed prompt to ask for detailed extraction from the chunk
            prompt = (
                f"Analyze the following text chunk, which is part {i+1} of {len(chunks)}. "
                f"Extract the key information, main arguments, and supporting details from this specific chunk. "
                f"Present the findings in a detailed paragraph format.\n\n"
                f"Text Chunk:\n\"\"\"\n{chunk}\n\"\"\""
            )
            summary = _call_selected_model(prompt)

            if not summary or summary.startswith("Error:"): # Propagate errors or handle empty summaries
                error_msg = summary if summary else "Error: Summarization failed for chunk (empty response)."
                logger.error(f"Failed to summarize chunk {i+1}/{len(chunks)}: {error_msg}")
                return error_msg # Return the error immediately

            # --- Store Chunk Summary in Cache ---
            if cache_manager:
                cache_manager.store_summary(chunk, chunk_cacheable_config, summary)
            # --- End Store Chunk Summary ---

        summaries.append(summary)
        # Optional: Add delay only if needed, maybe make configurable
        # time.sleep(0.5) # Consider if rate limits are hit without this

    combined = "\n".join(summaries)

    # --- Handle Combined Summary ---
    if len(combined) > max_chars:
        # Prepare cache config specific to combined summarization
        combine_cacheable_config = cacheable_config.copy()
        combine_cacheable_config['task'] = 'summarize_combined'

        # --- Cache Check for Combined Summary ---
        cached_final_summary = None
        if cache_manager:
            # Use 'combined' text as the key for the final summary cache
            cached_final_summary = cache_manager.get_summary(combined, combine_cacheable_config)

        if cached_final_summary is not None:
            logger.debug("Cache hit for final combined summary.")
            return cached_final_summary
        else:
            # --- Generate Final Combined Summary (Cache Miss) ---
            logger.info(f"Combining {len(summaries)} summaries into a final summary using {provider} (cache miss)...")
            # Changed prompt to ask for detailed synthesis
            prompt = (
                f"Synthesize the following individual summaries (derived from text chunks) into a single, coherent, and detailed overview. "
                f"Ensure the final summary integrates the key information, arguments, and details from all parts, preserving nuance.\n\n"
                f"Individual Summaries:\n\"\"\"\n{combined}\n\"\"\"\n\n"
                f"Final Synthesized Summary:"
            )
            final_summary = _call_selected_model(prompt)

            # --- Store Final Combined Summary in Cache ---
            if final_summary and not final_summary.startswith("Error:") and cache_manager:
                # Use 'combined' text as the key
                cache_manager.store_summary(combined, combine_cacheable_config, final_summary)
            # --- End Store Final Combined Summary ---

            return final_summary if final_summary else "Error: Final combination summarization failed (empty response)."
    else:
        # If combined summary is within limits, it means the chunk summaries were already short enough.
        # We already cached the individual chunks. We could potentially cache the combined result here too,
        # using the original full text as the key, but it might be redundant if the full text check passed initially.
        # For simplicity, just return the combined text.
        return combined
    # --- End Handle Combined Summary ---


# Modified rag_final_answer to accept llm_config dictionary
def rag_final_answer(aggregation_prompt, llm_config: dict = {}):
    """Generates the final answer using the specified RAG model configured in llm_config."""
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality")

    logger.info(f"Performing final RAG generation using provider: {provider}") # <<< Use logger
    try:
        if provider == "gemma":
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(aggregation_prompt, model=gemma_model, personality=personality)
        elif provider == "pali": # Note: 'pali' seems to just modify the prompt for gemma here
            modified_prompt = f"PALI mode analysis:\n\n{aggregation_prompt}"
            logger.info("Using Gemma (Ollama) with PALI mode prompt.") # <<< Use logger
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(modified_prompt, model=gemma_model, personality=personality)
        elif provider == "gemini":
            if not model_id:
                logger.error("Gemini selected for RAG, but no model specified.") # <<< Use logger
                return "Error: Gemini model not specified for RAG."
            return call_gemini(aggregation_prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            # logger.info("Using OpenRouter for final RAG generation.") # Redundant
            if not model_id:
                 logger.error("No OpenRouter model specified for RAG.") # <<< Use logger
                 return "Error: No OpenRouter model specified."
            return call_openrouter(aggregation_prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default or unknown, fall back to gemma
            logger.warning(f"Unknown provider '{provider}', defaulting to gemma.") # <<< Use logger
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(aggregation_prompt, model=gemma_model, personality=personality)
    except Exception as e:
        logger.error(f"Exception during RAG final answer generation ({provider}): {e}", exc_info=True) # <<< Use logger with traceback
        return f"Error: Failed to generate final answer with {provider} - {e}"

# Modified follow_up_conversation to accept llm_config dictionary
def follow_up_conversation(follow_up_prompt, llm_config: dict = {}):
    """Handles follow-up conversation prompts using the specified RAG model configured in llm_config."""
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality")

    logger.info(f"Handling follow-up conversation using {provider}...") # <<< Use logger
    try:
        if provider == "gemini":
            if not model_id:
                logger.error("Gemini selected for follow-up, but no model specified.") # <<< Use logger
                return "Error: Gemini model not specified for follow-up."
            return call_gemini(follow_up_prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            if not model_id:
                logger.error("OpenRouter selected for follow-up, but no model specified.") # <<< Use logger
                return "Error: OpenRouter model not specified for follow-up."
            return call_openrouter(follow_up_prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default to gemma/ollama
            if provider != "gemma":
                logger.warning(f"Unknown provider '{provider}' for follow-up, defaulting to gemma.") # <<< Use logger
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(follow_up_prompt, model=gemma_model, personality=personality)
    except Exception as e:
        logger.error(f"Exception during follow-up conversation ({provider}): {e}", exc_info=True) # <<< Use logger with traceback
        return f"Error: Failed to handle follow-up with {provider} - {e}"


def generate_followup_queries(initial_query: str, context_summary: str, llm_config: dict = {}, max_queries=3) -> list[str]:
    """
    Generates potential follow-up search queries based on the initial query and a summary of the context found so far.

    Args:
        initial_query (str): The original (or enhanced) user query.
        context_summary (str): A summary of the information gathered in the first pass (web + local).
        llm_config (dict): Configuration for the LLM provider (provider, model_id, api_key, personality).
        max_queries (int): The maximum number of follow-up queries to generate.

    Returns:
        list[str]: A list of generated follow-up query strings.
    """
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality") # Personality might influence query style

    prompt = (
        f"You are a research assistant analyzing search results.\n"
        f"Initial Query: \"{initial_query}\"\n\n"
        f"Summary of Information Found So Far:\n\"\"\"\n{context_summary}\n\"\"\"\n\n"
        f"Based on the initial query and the summary, identify key gaps, ambiguities, or areas needing deeper investigation. "
        f"Generate up to {max_queries} specific, concise search queries that would help address these points. "
        f"Output ONLY the search queries, one per line. Do not include numbering, bullets, or any other text."
    )

    raw_output = ""
    logger.info(f"Generating follow-up queries using {provider}...") # <<< Use logger
    try:
        if provider == "gemini":
            if not model_id:
                logger.error("Gemini selected for follow-up query generation, but no model specified.") # <<< Use logger
                return []
            raw_output = call_gemini(prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            if not model_id:
                logger.error("OpenRouter selected for follow-up query generation, but no model specified.") # <<< Use logger
                return []
            raw_output = call_openrouter(prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default to gemma/ollama
            if provider != "gemma":
                logger.warning(f"Unknown provider '{provider}' for follow-up query generation, defaulting to gemma.") # <<< Use logger
            gemma_model = model_id or "gemma2:2b" # Example default
            raw_output = call_gemma(prompt, model=gemma_model, personality=personality)

        if not raw_output or raw_output.startswith("Error:"):
            logger.warning(f"Follow-up query generation failed or returned error: {raw_output}") # <<< Use logger
            return []

        # Parse the output: split by lines, strip whitespace, filter empty lines
        queries = [line.strip() for line in raw_output.splitlines() if line.strip()]

        # Limit the number of queries
        return queries[:max_queries]

    except Exception as e:
        logger.error(f"Exception during follow-up query generation ({provider}): {e}", exc_info=True) # <<< Use logger with traceback
        return [] # Return empty list on error


def refine_text_section(section_content: str, instruction: str, llm_config: dict = {}) -> str:
    """
    Refines a given section of text based on a user-provided instruction using the specified LLM.

    Args:
        section_content (str): The text content (potentially HTML/Markdown) of the section to refine.
        instruction (str): The user's instruction for refinement (e.g., "Summarize this", "Make it simpler").
        llm_config (dict): Configuration for the LLM provider (provider, model_id, api_key, personality).

    Returns:
        str: The refined text content, or an error message starting with "Error:".
    """
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality") # Personality might influence refinement style

    # Construct the prompt
    # Instruct the LLM to maintain the original format (e.g., Markdown) if possible,
    # unless the instruction implies a format change (like 'convert to bullet points').
    prompt = (
        f"You are tasked with refining a section of a document based on a specific instruction.\n"
        f"User Instruction: \"{instruction}\"\n\n"
        f"Original Section Content:\n\"\"\"\n{section_content}\n\"\"\"\n\n"
        f"Please refine the original section content strictly according to the user's instruction. "
        f"Try to maintain the original formatting (e.g., Markdown headings, lists) unless the instruction requires changing it. "
        f"Output ONLY the refined section content. Do not include explanations, apologies, or introductions like 'Here is the refined section:'."
    )

    raw_output = ""
    logger.info(f"Refining text section using {provider}...") # <<< Use logger
    try:
        if provider == "gemini":
            if not model_id:
                logger.error("Gemini selected for refinement, but no model specified.") # <<< Use logger
                return "Error: Gemini model not specified for refinement."
            raw_output = call_gemini(prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            if not model_id:
                logger.error("OpenRouter selected for refinement, but no model specified.") # <<< Use logger
                return "Error: OpenRouter model not specified for refinement."
            raw_output = call_openrouter(prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default to gemma/ollama
            if provider != "gemma":
                logger.warning(f"Unknown provider '{provider}' for refinement, defaulting to gemma.") # <<< Use logger
            gemma_model = model_id or "gemma2:2b" # Example default
            raw_output = call_gemma(prompt, model=gemma_model, personality=personality)

        if not raw_output or raw_output.startswith("Error:"):
            error_msg = raw_output if raw_output else "Error: Refinement failed (empty response)."
            logger.warning(f"Refinement failed or returned error: {error_msg}") # <<< Use logger
            return error_msg # Return the error message

        # Return the raw output, assuming the LLM followed the prompt
        return raw_output.strip()

    except Exception as e:
        logger.error(f"Exception during text refinement ({provider}): {e}", exc_info=True) # <<< Use logger with traceback
        return f"Error: Failed to refine text with {provider} - {e}"
