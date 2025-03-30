# llm_providers/tasks.py
import time # Needed for summarize_text chunking delay (optional)

# Import functions from the new provider modules and utils
from .gemini import call_gemini
from .openrouter import call_openrouter
from .ollama import call_gemma
from .utils import extract_final_query

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
    try:
        if provider == "gemini":
            print("[INFO] Using Gemini for query enhancement.")
            if not model_id:
                print("[ERROR] Gemini selected for query enhancement, but no model specified.")
                return query # Fallback
            raw_output = call_gemini(prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            print("[INFO] Using OpenRouter for query enhancement.")
            if not model_id:
                 print("[WARN] No OpenRouter model specified for query enhancement, falling back to default.")
                 return query # Fallback
            raw_output = call_openrouter(prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default to gemma/ollama
            if provider != "gemma":
                print(f"[WARN] Unknown provider '{provider}' for query enhancement, defaulting to gemma.")
            print("[INFO] Using Gemma (Ollama) for query enhancement.")
            # Pass model_id to gemma if provided, otherwise it uses its default
            gemma_model = model_id or "gemma2:2b" # Example default if not specified
            raw_output = call_gemma(prompt, model=gemma_model, personality=personality)

        if not raw_output or raw_output.startswith("Error:"): # Handle potential API errors or empty output
            print(f"[WARN] Query enhancement failed or returned error: {raw_output}. Falling back to original query.")
            return query # Fallback to original query if enhancement fails

        return extract_final_query(raw_output)

    except Exception as e:
        print(f"[ERROR] Exception during query enhancement with {provider}: {e}. Falling back to original query.")
        # Log the exception details if needed
        return query # Fallback in case of unexpected errors during the call

# Modified summarize_text to accept llm_config dictionary (reordered params)
def summarize_text(text, llm_config: dict = {}, max_chars=6000):
    """Summarizes text, potentially chunking, using the specified RAG model configured in llm_config."""

    # Extract config once for the helper function
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality")

    def _call_selected_model(prompt_text):
        """Internal helper to call the correct model based on pre-extracted config."""
        try:
            if provider == "gemini":
                if not model_id:
                    print("[ERROR] Gemini selected for summarization, but no model specified.")
                    return "Error: Gemini model not specified for summarization."
                return call_gemini(prompt_text, model_name=model_id, gemini_api_key=api_key)
            elif provider == "openrouter":
                if not model_id:
                    print("[ERROR] OpenRouter selected for summarization, but no model specified.")
                    return "Error: OpenRouter model not specified for summarization."
                return call_openrouter(prompt_text, model=model_id, personality=personality, openrouter_api_key=api_key)
            else: # Default to gemma/ollama
                if provider != "gemma":
                     print(f"[WARN] Unknown provider '{provider}' for summarization, defaulting to gemma.")
                gemma_model = model_id or "gemma2:2b" # Example default
                return call_gemma(prompt_text, model=gemma_model, personality=personality)
        except Exception as e:
            print(f"[ERROR] Exception during _call_selected_model ({provider}): {e}")
            return f"Error: Failed to call {provider} model - {e}"


    if not text or not text.strip():
        print("[WARN] Attempted to summarize empty text.")
        return "" # Return empty string for empty input

    if len(text) <= max_chars:
        prompt = f"Please summarize the following text succinctly:\n\n{text}"
        summary = _call_selected_model(prompt)
        # Return summary, even if it's an error message from the LLM call
        return summary if summary else "Error: Summarization failed (empty response)."

    # If text is longer than max_chars, chunk it
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    summaries = []
    print(f"[INFO] Summarizing text in {len(chunks)} chunks using {provider}...")
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize part {i+1}/{len(chunks)}:\n\n{chunk}"
        summary = _call_selected_model(prompt)
        if not summary or summary.startswith("Error:"): # Propagate errors or handle empty summaries
            error_msg = summary if summary else "Error: Summarization failed for chunk (empty response)."
            print(f"[ERROR] Failed to summarize chunk {i+1}/{len(chunks)}: {error_msg}")
            return error_msg # Return the error immediately
        summaries.append(summary)
        # Optional: Add delay only if needed, maybe make configurable
        # time.sleep(1) # Consider if rate limits are hit without this

    combined = "\n".join(summaries)
    # Check if combined summary still needs further summarization
    if len(combined) > max_chars:
        print(f"[INFO] Combining {len(summaries)} summaries into a final summary using {provider}...")
        prompt = f"Combine these summaries into one concise final summary:\n\n{combined}"
        final_summary = _call_selected_model(prompt)
        # Return final summary, even if it's an error message
        return final_summary if final_summary else "Error: Final combination summarization failed (empty response)."
    else:
        # If combined summary is within limits, return it directly
        return combined


# Modified rag_final_answer to accept llm_config dictionary
def rag_final_answer(aggregation_prompt, llm_config: dict = {}):
    """Generates the final answer using the specified RAG model configured in llm_config."""
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality")

    print("[INFO] Performing final RAG generation using provider:", provider)
    try:
        if provider == "gemma":
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(aggregation_prompt, model=gemma_model, personality=personality)
        elif provider == "pali": # Note: 'pali' seems to just modify the prompt for gemma here
            modified_prompt = f"PALI mode analysis:\n\n{aggregation_prompt}"
            print("[INFO] Using Gemma (Ollama) with PALI mode prompt.")
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(modified_prompt, model=gemma_model, personality=personality)
        elif provider == "gemini":
            if not model_id:
                print("[ERROR] Gemini selected for RAG, but no model specified.")
                return "Error: Gemini model not specified for RAG."
            return call_gemini(aggregation_prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            print("[INFO] Using OpenRouter for final RAG generation.")
            if not model_id:
                 print("[ERROR] No OpenRouter model specified for RAG.")
                 return "Error: No OpenRouter model specified."
            return call_openrouter(aggregation_prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default or unknown, fall back to gemma
            print(f"[WARN] Unknown provider '{provider}', defaulting to gemma.")
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(aggregation_prompt, model=gemma_model, personality=personality)
    except Exception as e:
        print(f"[ERROR] Exception during RAG final answer generation ({provider}): {e}")
        return f"Error: Failed to generate final answer with {provider} - {e}"

# Modified follow_up_conversation to accept llm_config dictionary
def follow_up_conversation(follow_up_prompt, llm_config: dict = {}):
    """Handles follow-up conversation prompts using the specified RAG model configured in llm_config."""
    # Extract config with defaults
    provider = llm_config.get("provider", "gemma")
    model_id = llm_config.get("model_id")
    api_key = llm_config.get("api_key")
    personality = llm_config.get("personality")

    print(f"[INFO] Handling follow-up conversation using {provider}...")
    try:
        if provider == "gemini":
            if not model_id:
                print("[ERROR] Gemini selected for follow-up, but no model specified.")
                return "Error: Gemini model not specified for follow-up."
            return call_gemini(follow_up_prompt, model_name=model_id, gemini_api_key=api_key)
        elif provider == "openrouter":
            if not model_id:
                print("[ERROR] OpenRouter selected for follow-up, but no model specified.")
                return "Error: OpenRouter model not specified for follow-up."
            return call_openrouter(follow_up_prompt, model=model_id, personality=personality, openrouter_api_key=api_key)
        else: # Default to gemma/ollama
            if provider != "gemma":
                print(f"[WARN] Unknown provider '{provider}' for follow-up, defaulting to gemma.")
            gemma_model = model_id or "gemma2:2b" # Example default
            return call_gemma(follow_up_prompt, model=gemma_model, personality=personality)
    except Exception as e:
        print(f"[ERROR] Exception during follow-up conversation ({provider}): {e}")
        return f"Error: Failed to handle follow-up with {provider} - {e}"
