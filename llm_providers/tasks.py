# llm_providers/tasks.py
import time # Needed for summarize_text chunking delay (optional)

# Import functions from the new provider modules and utils
from .gemini import call_gemini
from .openrouter import call_openrouter
from .ollama import call_gemma
from .utils import extract_final_query

# Modified to accept rag_model, selected_gemini_model, selected_openrouter_model, and API keys
def chain_of_thought_query_enhancement(query, personality=None, rag_model="gemma", selected_gemini_model=None, selected_openrouter_model=None, gemini_api_key=None, openrouter_api_key=None):
    """Enhances a query using a step-by-step thinking process with the specified LLM."""
    prompt = (
        "You are an expert search strategist. Think step-by-step through the implications and nuances "
        "of the following query and produce a final, enhanced query that covers more angles.\n\n"
        f"Query: \"{query}\"\n\n"
        "After your reasoning, output only the final enhanced query on a single line - SHORT AND CONCISE.\n"
        "Provide your reasoning, and at the end output the line 'Final Enhanced Query:' followed by the enhanced query."
    )
    raw_output = ""
    try:
        if rag_model == "gemini":
            print("[INFO] Using Gemini for query enhancement.")
            if not selected_gemini_model:
                print("[ERROR] Gemini selected for query enhancement, but no model specified.")
                return query # Fallback
            # Pass the selected model name and API key
            raw_output = call_gemini(prompt, model_name=selected_gemini_model, gemini_api_key=gemini_api_key)
        elif rag_model == "openrouter":
            print("[INFO] Using OpenRouter for query enhancement.")
            if not selected_openrouter_model:
                 print("[WARN] No OpenRouter model specified for query enhancement, falling back to default.")
                 return query # Fallback
            # Pass the selected model name, personality, and API key
            raw_output = call_openrouter(prompt, model=selected_openrouter_model, personality=personality, openrouter_api_key=openrouter_api_key)
        else: # Default to gemma/ollama
            if rag_model != "gemma":
                print(f"[WARN] Unknown rag_model '{rag_model}' for query enhancement, defaulting to gemma.")
            print("[INFO] Using Gemma (Ollama) for query enhancement.")
            # Assuming call_gemma doesn't need API key for local Ollama
            raw_output = call_gemma(prompt, personality=personality)

        if not raw_output or raw_output.startswith("Error:"): # Handle potential API errors or empty output
            print(f"[WARN] Query enhancement failed or returned error: {raw_output}. Falling back to original query.")
            return query # Fallback to original query if enhancement fails

        return extract_final_query(raw_output)

    except Exception as e:
        print(f"[ERROR] Exception during query enhancement with {rag_model}: {e}. Falling back to original query.")
        # Log the exception details if needed
        return query # Fallback in case of unexpected errors during the call

# Modified summarize_text to support different RAG models and accept API keys
def summarize_text(text, max_chars=6000, personality=None, rag_model="gemma", selected_gemini_model=None, selected_openrouter_model=None, gemini_api_key=None, openrouter_api_key=None):
    """Summarizes text, potentially chunking, using the specified RAG model."""

    def _call_selected_model(prompt_text):
        """Internal helper to call the correct model based on rag_model."""
        try:
            if rag_model == "gemini":
                if not selected_gemini_model:
                    print("[ERROR] Gemini selected for summarization, but no model specified.")
                    return "Error: Gemini model not specified for summarization."
                # Pass API key
                return call_gemini(prompt_text, model_name=selected_gemini_model, gemini_api_key=gemini_api_key)
            elif rag_model == "openrouter":
                if not selected_openrouter_model:
                    print("[ERROR] OpenRouter selected for summarization, but no model specified.")
                    return "Error: OpenRouter model not specified for summarization."
                # Pass API key
                return call_openrouter(prompt_text, model=selected_openrouter_model, personality=personality, openrouter_api_key=openrouter_api_key)
            else: # Default to gemma/ollama
                if rag_model != "gemma":
                     print(f"[WARN] Unknown rag_model '{rag_model}' for summarization, defaulting to gemma.")
                return call_gemma(prompt_text, personality=personality)
        except Exception as e:
            print(f"[ERROR] Exception during _call_selected_model ({rag_model}): {e}")
            return f"Error: Failed to call {rag_model} model - {e}"


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
    print(f"[INFO] Summarizing text in {len(chunks)} chunks using {rag_model}...")
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
        print(f"[INFO] Combining {len(summaries)} summaries into a final summary using {rag_model}...")
        prompt = f"Combine these summaries into one concise final summary:\n\n{combined}"
        final_summary = _call_selected_model(prompt)
        # Return final summary, even if it's an error message
        return final_summary if final_summary else "Error: Final combination summarization failed (empty response)."
    else:
        # If combined summary is within limits, return it directly
        return combined


# Added selected_gemini_model, selected_openrouter_model parameters, and API keys
def rag_final_answer(aggregation_prompt, rag_model="gemma", personality=None, selected_gemini_model=None, selected_openrouter_model=None, gemini_api_key=None, openrouter_api_key=None):
    """Generates the final answer using the specified RAG model."""
    print("[INFO] Performing final RAG generation using model:", rag_model)
    try:
        if rag_model == "gemma":
            # Assuming call_gemma doesn't need API key for local Ollama
            return call_gemma(aggregation_prompt, personality=personality)
        elif rag_model == "pali": # Note: 'pali' seems to just modify the prompt for gemma here
            modified_prompt = f"PALI mode analysis:\n\n{aggregation_prompt}"
            print("[INFO] Using Gemma (Ollama) with PALI mode prompt.")
            return call_gemma(modified_prompt, personality=personality)
        elif rag_model == "gemini":
            if not selected_gemini_model:
                print("[ERROR] Gemini selected for RAG, but no model specified.")
                return "Error: Gemini model not specified for RAG."
            # Pass API key
            return call_gemini(aggregation_prompt, model_name=selected_gemini_model, gemini_api_key=gemini_api_key)
        elif rag_model == "openrouter":
            print("[INFO] Using OpenRouter for final RAG generation.")
            if not selected_openrouter_model:
                 print("[ERROR] No OpenRouter model specified for RAG.")
                 return "Error: No OpenRouter model specified."
            # Pass the selected model name and API key
            return call_openrouter(aggregation_prompt, model=selected_openrouter_model, personality=personality, openrouter_api_key=openrouter_api_key)
        else: # Default or unknown, fall back to gemma
            print(f"[WARN] Unknown rag_model '{rag_model}', defaulting to gemma.")
            # Assuming call_gemma doesn't need API key for local Ollama
            return call_gemma(aggregation_prompt, personality=personality)
    except Exception as e:
        print(f"[ERROR] Exception during RAG final answer generation ({rag_model}): {e}")
        return f"Error: Failed to generate final answer with {rag_model} - {e}"

# Modified follow_up_conversation to support different RAG models and accept API keys
def follow_up_conversation(follow_up_prompt, personality=None, rag_model="gemma", selected_gemini_model=None, selected_openrouter_model=None, gemini_api_key=None, openrouter_api_key=None):
    """Handles follow-up conversation prompts using the specified RAG model."""
    print(f"[INFO] Handling follow-up conversation using {rag_model}...")
    try:
        if rag_model == "gemini":
            if not selected_gemini_model:
                print("[ERROR] Gemini selected for follow-up, but no model specified.")
                return "Error: Gemini model not specified for follow-up."
            # Pass API key
            return call_gemini(follow_up_prompt, model_name=selected_gemini_model, gemini_api_key=gemini_api_key)
        elif rag_model == "openrouter":
            if not selected_openrouter_model:
                print("[ERROR] OpenRouter selected for follow-up, but no model specified.")
                return "Error: OpenRouter model not specified for follow-up."
            # Pass API key
            return call_openrouter(follow_up_prompt, model=selected_openrouter_model, personality=personality, openrouter_api_key=openrouter_api_key)
        else: # Default to gemma/ollama
            if rag_model != "gemma":
                print(f"[WARN] Unknown rag_model '{rag_model}' for follow-up, defaulting to gemma.")
            return call_gemma(follow_up_prompt, personality=personality)
    except Exception as e:
        print(f"[ERROR] Exception during follow-up conversation ({rag_model}): {e}")
        return f"Error: Failed to handle follow-up with {rag_model} - {e}"
