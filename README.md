# NanoSage 🧙: Advanced Recursive Search & Report Generation  

Deep Research assistant that runs on your laptop, using tiny models. - all open source!

## How is NanoSage different than other Assistant Researchers?

It offers a structured breakdown of a multi-source, relevance-driven, recursive search pipeline. It walks through how the system refines a user query, builds a knowledge base from local and web data, and dynamically explores subqueries—tracking progress through a Table of Contents (TOC).

With Monte Carlo-based exploration, the system balances depth vs. breadth, ranking each branch’s relevance to ensure precision and avoid unrelated tangents. The result? A detailed, well-organized report generated using retrieval-augmented generation (RAG), integrating the most valuable insights.

I wanted to experiment with new research methods, so I thought, basically, when we research a topic, we randomly explore new ideas as we search, and NanoSage basically does that!
It explores and records its journey, where each (relevant) step is a node... and then sums it up to you in a neat report!
Where the table of content is basically its search graph. 🧙

---

## Example Report

You can find an example report in the following link:  
[example report output for query: "Create a structure bouldering gym workout to push my climbing from v4 to v6"](https://github.com/masterFoad/NanoSage/blob/main/example_report.md)

---

## Quick Start Guide  

### 1. Install Dependencies

1. Ensure **Python 3.8+** is installed.
2. Install required packages (including `PyQt6` for the GUI and other necessary libraries):

```bash
pip install -r requirements.txt
```

3. *(Optional)* For GPU acceleration, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*(Replace `cu118` with your CUDA version.)*


4. Make sure to update pyOpenSSL and cryptography:

```bash
pip install --upgrade pyOpenSSL cryptography
```

---

### 1.5. Launch the GUI (Recommended)

NanoSage includes a graphical user interface (GUI) built with PyQt6 for easier interaction.

1.  **Dependencies**: The required `PyQt6` library is included in `requirements.txt` and installed in the previous step.
2.  **Run the GUI**:
    ```bash
    python main.py --gui
    ```
    This will open the main application window where you can enter your query, configure settings (like web search, embedding model, RAG provider), and start the research process. Results are saved in the `results/` directory.

---


### 2. Set Up Ollama & Pull the Gemma Model

1. **Install Ollama**:

```bash
curl -fsSL https://ollama.com/install.sh | sh
pip install --upgrade ollama
```
*(Windows users: see [ollama.com](https://ollama.com) for installer.)*

2. **Pull Gemma 2B** (for RAG-based summaries):

```bash
ollama pull gemma2:2b
```

---

### 3. Run a Simple Search Query (CLI Mode)

If you prefer the command line, here's a sample command:

```bash
python main.py --query "Create a structured bouldering gym workout to push my climbing from v4 to v6" \
               --web_search \
               --max_depth 2 \
               --device cpu \
               --top_k 10 \
               --embedding_model colpali \
               --rag_model gemma
```

**Common Parameters**:
- `--query`: Main search query (required in CLI mode).
- `--gui`: Launches the graphical user interface instead of running in CLI.
- `--config`: Path to the configuration file (default: `config.yaml`).
- `--corpus_dir`: Path to a directory containing local documents for search.
- `--web_search`: Enables web-based retrieval (use `--no-web_search` to disable if default is true in config).
- `--max_depth`: Recursion depth for subqueries (default: 1).
- `--device`: Device for embedding model (`cpu` or `cuda`).
- `--top_k`: Number of local documents to retrieve.
- `--embedding_model`: Specifies the embedding model (e.g., `colpali`, `all-minilm`, `models/embedding-001`, `openai/text-embedding-ada-002`). Default is defined in `config.yaml`.
- `--rag_model`: Selects the LLM provider for summarization and report generation (`gemma`, `pali`, `gemini`, `openrouter`). Default in `config.yaml`.
- `--gemma_model_id`: Specific Ollama model ID (e.g., `gemma2:2b`, `llama3:8b`) if using `gemma` or `pali`.
- `--gemini_model_id`: Specific Gemini model ID (e.g., `models/gemini-1.5-flash-latest`) if using `gemini`.
- `--openrouter_model_id`: Specific OpenRouter model ID (e.g., `openai/gpt-3.5-turbo`, `google/gemini-flash-1.5`) if using `openrouter`.
- `--personality`: Optional personality prompt for the RAG LLM (e.g., "scientific", "concise").

*Settings Hierarchy: Command-line arguments override `config.yaml` settings, which override internal defaults.*

---

### 4. Check Results & Report

A **detailed Markdown report** will appear in `results/<query_id>/`.

**Example**:
```
results/
└── 389380e2/
    ├── Quantum_computing_in_healthcare_output.md
    ├── web_Quantum_computing/
    ├── web_results/
    └── local_results/
```

Open the `*_output.md` file (e.g., `Quantum_computing_in_healthcare_output.md`) in a Markdown viewer (VSCode, Obsidian, etc.).

---

### 5. Advanced Options

#### ✅ Using Local Files

If you have local PDFs, text files, or images:

```bash
python main.py --query "AI in finance" \
               --corpus_dir "my_local_data/" \
               --embedding_model all-minilm \
               --top_k 5 \
               --device cpu
```
*This searches local documents in `my_local_data/` using the `all-minilm` embedding model.*

Now the system searches **both** local docs and web data (if `--web_search` is enabled).

#### 🔄 RAG with Different LLM Providers

You can choose the LLM provider for generating summaries and the final report using `--rag_model`.

**Using Gemma (via Ollama):**

```bash
python main.py --query "Climate change impact on economy" \
               --rag_model gemma \
               --gemma_model_id llama3:8b \
               --personality "scientific"
```
*This uses the `llama3:8b` model (via Ollama) for RAG. If `--gemma_model_id` is omitted, it uses the default specified in `config.yaml` or the internal default (`gemma2:2b`).*

**Using Gemini API:**

1.  **Get an API Key**: Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  **Provide the Key**: You can provide the key in two ways:
    *   **Environment Variable**: Set the `GEMINI_API_KEY` environment variable.
        ```bash
        export GEMINI_API_KEY='YOUR_API_KEY' 
        # Windows: set GEMINI_API_KEY=YOUR_API_KEY
        ```
    *   **Configuration File**: Add the key to your `config.yaml` under `api_keys`:
        ```yaml
        api_keys:
          gemini_api_key: YOUR_API_KEY_HERE 
        ```
3.  **Run with Gemini**: Use the `--rag_model gemini` flag.
    ```bash
    python main.py --query "Latest advancements in renewable energy" \
                   --rag_model gemini \
                   --gemini_model_id models/gemini-1.5-pro-latest \
                   --web_search
    ```
    *If `--gemini_model_id` is omitted, it uses the default from `config.yaml` or the internal default. If no model is specified anywhere and the API key is valid, the application will list available models and prompt you to choose one interactively.*

**Using OpenRouter API:**

1.  **Get an API Key**: Obtain an OpenRouter API key from [OpenRouter.ai](https://openrouter.ai/).
2.  **Provide the Key**: Similar to Gemini, provide the key via:
    *   **Environment Variable**: Set the `OPENROUTER_API_KEY` environment variable.
        ```bash
        export OPENROUTER_API_KEY='YOUR_API_KEY'
        # Windows: set OPENROUTER_API_KEY=YOUR_API_KEY
        ```
    *   **Configuration File**: Add the key to `config.yaml` under `api_keys`:
        ```yaml
        api_keys:
          openrouter_api_key: YOUR_API_KEY_HERE 
        ```
3.  **Run with OpenRouter**: Use the `--rag_model openrouter` flag.
    ```bash
    python main.py --query "Future of AI in education" \
                   --rag_model openrouter \
                   --openrouter_model_id google/gemini-flash-1.5 \
                   --web_search
    ```
    *Specify the desired model using `--openrouter_model_id`. If omitted, it uses the default from `config.yaml` or the internal default (`openai/gpt-3.5-turbo`).*

---

### 6. Troubleshooting

- **Missing dependencies?** Rerun: `pip install -r requirements.txt`
- **Ollama not found?** Ensure it’s installed (`ollama list` shows `gemma:2b`).
- **Memory issues?** Use `--device cpu`.
- **Too many subqueries?** Lower `--max_depth` to 1.

---

### 7. Next Steps

- **Try different embedding models** (`--embedding_model all-minilm`).
- **Experiment with different RAG providers and models** (`--rag_model`, `--gemma_model_id`, etc.).
- **Tweak recursion** (`--max_depth`).
- **Tune** `config.yaml` for web search limits, `min_relevance`, API keys, or default models.

---

## Detailed Design: NanoSage Architecture

### 1. Core Input Parameters

- **User Query**: E.g. `"Quantum computing in healthcare"`.
- **CLI Flags** (in `main.py`):
  ```
  --query
  --gui
  --config
  --corpus_dir
  --device
  --embedding_model # Changed from retrieval_model
  --top_k
  --web_search / --no-web_search
  --personality
  --rag_model       # gemma, pali, gemini, openrouter
  --gemma_model_id
  --gemini_model_id
  --openrouter_model_id
  --max_depth
  ```
- **YAML Config** (e.g. `config.yaml`):
  - Sections: `general`, `retrieval`, `llm`, `api_keys`.
  - Defines defaults for most CLI flags.
  - Allows setting API keys directly (`gemini_api_key`, `openrouter_api_key`).

### 2. Configuration & Session Setup

1. **Configuration Loading**:
   - `load_config(config_path)` reads YAML.
   - Settings are resolved in order: CLI Args > Config File > Internal Defaults.
   - API keys are resolved: Config File > Environment Variable > None.

2. **Session Initialization**:
   `SearchSession.__init__()` uses the *resolved* settings:
   - Sets up `query_id`, `base_result_dir`.
   - Enhances query via `chain_of_thought_query_enhancement()`.
   - Loads the specified embedding model (`load_embedding_model()`). # Updated function name assumed
   - Embeds the query for relevance checks (`embed_text()`).
   - Loads local files (if `corpus_dir` provided) into `KnowledgeBase`.
   - Initializes the selected RAG provider (Ollama, Gemini, OpenRouter).

### 3. Recursive Web Search & TOC Tracking

1. **Subquery Generation**:  
   - The enhanced query is split with `split_query()`.
2. **Relevance Filtering**:  
   - For each subquery, compare embeddings with the main query (via `late_interaction_score()`).  
   - If `< min_relevance`, skip to avoid rabbit holes.
3. **TOCNode Creation**:  
   - Each subquery → `TOCNode`, storing the text, summary, relevance, etc.
4. **Web Data**:  
   - If relevant:  
     - `download_webpages_ddg()` to fetch results.  
     - `parse_html_to_text()` and embed them.  
     - Summarize snippets (`summarize_text()`).  
   - If `current_depth < max_depth`, optionally **expand** new sub-subqueries (chain-of-thought on the current subquery).
5. **Hierarchy**:  
   - All subqueries & expansions form a tree of TOC nodes for the final report.

### 4. Local Retrieval & Summaries

1. **Local Documents** + **Downloaded Web Entries** → appended into `KnowledgeBase`.
2. **KnowledgeBase.search(...)** for top-K relevant docs.
3. Summaries:
   - Summarize web results & local retrieval with `summarize_text()`.

### 5. Final RAG Prompt & Report Generation

1. **Build Prompt**: `_build_final_answer(...)` constructs a prompt with:
   - User query, TOC, summaries, references.
2. **Call LLM**: `rag_final_answer(...)` calls the selected RAG provider (`call_ollama()`, `call_gemini()`, `call_openrouter()`) with the prompt and specified model ID. # Updated function names assumed
3. **Save Report**: `aggregate_results(...)` saves the generated report to `results/<query_id>/`.

### 6. Balancing Exploration vs. Exploitation

- Subqueries with **relevance_score < min_relevance** are skipped.
- Depth-limited recursion ensures not to blow up on too many expansions.
- **Monte Carlo** expansions (optional) can sample random subqueries to avoid missing unexpected gems.

### 7. Final Output

- **Markdown report** summarizing relevant subqueries, local docs, and a final advanced RAG-based discussion.

---

## Summary Flow Diagram

```plaintext
User Query
    │
    ▼
main.py:
    └── load_config(config.yaml)
         └── Create SearchSession(...)
              │
              ├── chain_of_thought_query_enhancement()
              ├── load_embedding_model() # Updated
              ├── embed_text() for reference
              ├── load_corpus_from_dir() → KnowledgeBase.add_documents()
              └── run_session():
                  └── perform_recursive_web_searches():
                      ├── For each subquery:
                      │   ├─ Compute relevance_score
                      │   ├─ if relevance_score < min_relevance: skip
                      │   ├─ else:
                      │   │   ├─ download_webpages_ddg()
                      │   │   ├─ parse_html_to_text(), embed
                      │   │   ├─ summarize_text() → store in TOCNode
                      │   │   └─ if depth < max_depth:
                      │   │       └─ recursively expand
                      └── Aggregates web corpus, builds TOC
              │
              ├── KnowledgeBase.search(enhanced_query, top_k)
              ├── Summarize results
              ├── _build_final_answer() → prompt
              ├── rag_final_answer() → call_ollama()/call_gemini()/call_openrouter() # Updated
              └── aggregate_results() → saves Markdown
```


If you found **NanoSage** useful for your research or project - or saved you 1 minute of googling, please consider citing it:  

**BibTeX Citation:**  
```bibtex
@misc{NanoSage,
  author = {Foad Abo Dahood}, 
  title = {NanoSage: A Recursive, Relevance-Driven Search and RAG Pipeline},
  year = {2025},
  howpublished = {\url{https://github.com/masterFoad/NanoSage}},
  note = {Accessed: \today}
}
```

**APA Citation:**  
Foad, Abo Dahood. (2025). *NanoSage: A Recursive, Relevance-Driven Search and RAG Pipeline*. Retrieved from [https://github.com/masterFoad/NanoSage](https://github.com/masterFoad/NanoSage)
