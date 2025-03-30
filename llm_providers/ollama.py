# llm_providers/ollama.py
from ollama import chat, ChatResponse

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
        response: ChatResponse = chat(model=model, messages=messages)
        return response.message.content
    except Exception as e:
        print(f"[ERROR] Ollama call failed for model {model}: {e}")
        # Depending on desired behavior, could return an error string or raise
        return f"Error: Ollama call failed - {e}"
