# llm_utils.py

import os
import time
import re
import google.generativeai as genai
from ollama import chat, ChatResponse

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

# Modified to require model_name
def call_gemini(prompt, model_name):
    """Calls the Gemini API with a specified model."""
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
        return f"Error: Gemini API call failed. Details: {e}"

def extract_final_query(text):
    marker = "Final Enhanced Query:"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()

# Modified to accept rag_model and selected_gemini_model
def chain_of_thought_query_enhancement(query, personality=None, rag_model="gemma", selected_gemini_model=None):
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
    else:
        print("[INFO] Using Gemma (Ollama) for query enhancement.")
        raw_output = call_gemma(prompt, personality=personality)

    if raw_output.startswith("Error:"): # Handle potential API errors
        print(f"[WARN] Query enhancement failed: {raw_output}. Falling back to original query.")
        return query # Fallback to original query if enhancement fails

    return extract_final_query(raw_output)

def summarize_text(text, max_chars=6000, personality=None):
    if len(text) <= max_chars:
        prompt = f"Please summarize the following text succinctly:\n\n{text}"
        return call_gemma(prompt, personality=personality)
    # If text is longer than max_chars, chunk it
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize part {i+1}/{len(chunks)}:\n\n{chunk}"
        summary = call_gemma(prompt, personality=personality)
        summaries.append(summary)
        time.sleep(1) # Consider making sleep configurable or removing if not strictly needed
    combined = "\n".join(summaries)
    if len(combined) > max_chars:
        prompt = f"Combine these summaries into one concise summary:\n\n{combined}"
        combined = call_gemma(prompt, personality=personality)
    return combined

# Added selected_gemini_model parameter
def rag_final_answer(aggregation_prompt, rag_model="gemma", personality=None, selected_gemini_model=None):
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
    else: # Default or unknown, fall back to gemma
        print(f"[WARN] Unknown rag_model '{rag_model}', defaulting to gemma.")
        return call_gemma(aggregation_prompt, personality=personality)

def follow_up_conversation(follow_up_prompt, personality=None):
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
