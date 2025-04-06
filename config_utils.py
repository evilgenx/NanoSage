# config_utils.py
import os
import yaml
import collections.abc
from gui.searxng_engines import DEFAULT_ENABLED_ENGINES # Import the default list

# --- Helper for deep merging dictionaries ---
def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

# --- Default configuration structure ---
DEFAULT_CONFIG = {
    'general': {
        'corpus_dir': None,
        'device': 'cpu',
        'max_depth': 1,
        'web_search': True
    },
    'retrieval': {
        'retrieval_model': 'colpali',
        'top_k': 3,
        'embedding_model': 'local' # Added default
    },
    'llm': {
        'rag_model': 'gemma',
        'personality': 'cheerful assistant',
        'rag_model_type': 'gemma', # Added for RAG tab selection
        'selected_gemini_model': 'models/gemini-1.5-flash-latest', # Added for RAG tab selection
        'selected_openrouter_model': 'openai/gpt-3.5-turbo', # Added for RAG tab selection
        'rag_personality': 'cheerful assistant', # Added for RAG tab input
        'selected_output_format': 'report', # Added for RAG tab selection (defaulting to 'report')
        'gemma_model_id': 'gemma2:2b', # Kept for potential backend use, though UI might override
        'gemini_model_id': 'models/gemini-1.5-flash-latest', # Kept for potential backend use
        'openrouter_model_id': 'openai/gpt-3.5-turbo', # Kept for potential backend use
        'personality': 'cheerful assistant', # Kept for potential backend use
        'rag_report_prompt_template': """You are an expert research analyst tasked with creating a comprehensive, well-structured report.
Use the provided data to generate a report of at least 3000 words on the topic.

**Formatting Guidelines:**
*   Use clear Markdown formatting throughout the report.
*   Employ headings (`## Section Title`) and subheadings (`### Subsection Title`) consistently based on the Table of Contents structure.
*   Use **bold text** for key terms or emphasis within paragraphs where appropriate.
*   Utilize bullet points (`*` or `-`) or numbered lists (`1.`, `2.`) for clarity when presenting lists or enumerated points.
*   Ensure adequate paragraph spacing and line breaks between sections for readability.
*   Cite sources immediately after the relevant statement using `(URL)` or `(Local File Path)`. Do not wait until the end.
*   Include a final "References" section listing all unique URLs and file paths cited within the report body.

**Report Content Structure:**
1.  **Table of Contents:** Use the provided structure below.
2.  **Introduction:** Provide background, context, and scope based on the query.
3.  **Main Body:** Synthesize information from the web and local summaries. Organize logically using sections/subsections from the Table of Contents. Analyze findings, discuss challenges, and explore future directions. Cite sources diligently as you write.
4.  **Conclusion:** Summarize the key findings and offer concluding remarks.
5.  **References:** Create a list of all unique URLs and file paths cited above.

---
**Input Data:**

**User Query:** {enhanced_query}

**Table of Contents Structure:**
{toc_str}

**Summarized Web Results:**
{summarized_web}

**Summarized Local Document Results:**
{summarized_local}

**Reference Links (Unique URLs/Paths Found):**
{reference_links_str}
---

**Generate the Report:**
"""
    },
    'search': {
        'provider': 'searxng',
        'duckduckgo': {
            'max_results': 5
        },
        'searxng': {
            'base_url': 'http://127.0.0.1:8080', # Default to localhost
            'max_results': 5,
            'language': 'en',
            'safesearch': 1,
            'time_range': None,
            'categories': None,
            'engines': DEFAULT_ENABLED_ENGINES, # Use the imported list as default
            # Default plugins based on SearXNG documentation
            'enabled_plugins': [
                'Hash_plugin', 'Self_Information', 'Tracker_URL_remover', 'Ahmia_blacklist'
            ],
            'disabled_plugins': [
                'Hostnames_plugin', 'Open_Access_DOI_rewrite', 'Vim-like_hotkeys', 'Tor_check_plugin'
            ]
        }
    },
    'api_keys': {
        'gemini_api_key': 'YOUR_GEMINI_API_KEY_HERE',
        'openrouter_api_key': 'YOUR_OPENROUTER_API_KEY_HERE'
    },
    'cache': {
        'enabled': False, # Default to disabled for safety/simplicity initially
        'db_path': "cache/nanosage_cache.db"
    },
    'knowledge_base': { # <<< Added section
        'provider': 'chromadb', # Default to chromadb
        'chromadb': {
            'path': "./chroma_db",
            'collection_name': "nanosage_kb",
            # 'distance_function': "cosine" # Optional
        }
    }
}

def load_config(config_path):
    """
    Loads configuration from a YAML file.
    Creates a default file if not found.
    Merges loaded config with defaults to ensure all keys are present.
    """
    # Create default config if file doesn't exist
    if not os.path.isfile(config_path):
        print(f"[INFO] Config file not found at {config_path}. Creating default config.")
        if save_config(config_path, DEFAULT_CONFIG):
            print(f"[INFO] Default configuration saved to {config_path}.")
            # Return a deep copy of defaults if created successfully
            import copy
            return copy.deepcopy(DEFAULT_CONFIG)
        else:
            print(f"[ERROR] Failed to save default config to {config_path}. Returning empty config.")
            return {} # Return empty if saving failed

    # Load existing config file
    loaded_config = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
            # Ensure we have a dict even if the file is empty or contains only 'null'
            loaded_config = loaded_data if isinstance(loaded_data, dict) else {}
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file {config_path}: {e}")
        loaded_config = {} # Start with empty on error
    except FileNotFoundError:
        print(f"[ERROR] File not found during open (should have been created): {config_path}")
        loaded_config = {}
    except Exception as e:
        print(f"[ERROR] Failed to read config file {config_path}: {e}")
        loaded_config = {}

    # Merge loaded config onto a copy of defaults
    # This ensures all default keys are present if missing in the loaded file
    import copy
    final_config = copy.deepcopy(DEFAULT_CONFIG)
    deep_update(final_config, loaded_config) # Merge loaded values into the defaults

    return final_config

def save_config(config_path, config_data):
    """Saves configuration data to a YAML file."""
    try:
        # Ensure the directory exists only if config_path includes a directory
        directory = os.path.dirname(config_path)
        if directory: # Only create directory if path is not empty
            os.makedirs(directory, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Configuration saved to: {config_path}")
        return True
    except yaml.YAMLError as e:
        print(f"[ERROR] Error writing YAML file {config_path}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to save config file {config_path}: {e}")
        return False
