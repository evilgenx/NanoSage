# config_utils.py
import os
import yaml

# Default configuration structure
DEFAULT_CONFIG = {
    'general': {
        'corpus_dir': None,
        'device': 'cpu',
        'max_depth': 1,
        'web_search': True
    },
    'retrieval': {
        'retrieval_model': 'colpali',
        'top_k': 3
    },
    'llm': {
        'rag_model': 'gemma',
        'personality': 'cheerful assistant',
        'gemma_model_id': 'gemma2:2b',
        'gemini_model_id': 'models/gemini-1.5-flash-latest',
        'openrouter_model_id': 'openai/gpt-3.5-turbo',
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
            'engines': None
        }
    },
    'api_keys': {
        'gemini_api_key': 'YOUR_GEMINI_API_KEY_HERE',
        'openrouter_api_key': 'YOUR_OPENROUTER_API_KEY_HERE'
    }
}

def load_config(config_path):
    """Loads configuration from a YAML file. Creates a default if not found."""
    if not os.path.isfile(config_path):
        print(f"[INFO] Config file not found at {config_path}. Creating default config.")
        if save_config(config_path, DEFAULT_CONFIG):
            print(f"[INFO] Default configuration saved to {config_path}.")
            return DEFAULT_CONFIG
        else:
            print(f"[ERROR] Failed to save default config to {config_path}. Returning empty config.")
            return {} # Return empty if saving failed

    # Load existing config file
    config = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
            # Ensure we return a dict even if the file is empty or contains only 'null'
            config = loaded_data if isinstance(loaded_data, dict) else {}
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file {config_path}: {e}")
        # Return empty dict on parsing error
        config = {}
    except FileNotFoundError:
        # This case is already handled by the initial check, but good practice
        print(f"[ERROR] File not found during open: {config_path}")
        config = {}
    except Exception as e:
        # Catch other potential file reading errors
        print(f"[ERROR] Failed to read config file {config_path}: {e}")
        config = {}

    return config

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
