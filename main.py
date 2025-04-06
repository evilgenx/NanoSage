# main.py

import argparse
import asyncio
import yaml
import os
import sys # Import sys for exiting
import logging # Added for logging

# Import SearchSession and related functions
from search_session import SearchSession
from llm_providers.gemini import list_gemini_models # Import from the new provider file
from config_utils import load_config, DEFAULT_CONFIG # Import from the new utility file
from cache_manager import CacheManager # Added CacheManager import
from knowledge_base import KnowledgeBase # Added for scraping
from embeddings.factory import create_embedder # Added for scraping
from web_scraper import scrape_url_to_markdown # Added for scraping

# Conditional GUI imports
GUI_ENABLED = True
try:
    from PyQt6.QtWidgets import QApplication
    # Import from the new structure
    from gui.main_window import MainWindow
except ImportError:
    GUI_ENABLED = False
    # Updated warning message
    print("[WARN] PyQt6 or GUI components (gui/main_window.py) not found. GUI mode disabled.")

# Removed load_config function definition from here

def main():
    parser = argparse.ArgumentParser(description="Multi-step RAG pipeline with depth-limited searching.")
    # Make query not strictly required initially, will check later for CLI mode
    parser.add_argument("--query", type=str, help="Initial user query (required for CLI mode)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--corpus_dir", type=str, default=None, help="Path to local corpus folder")
    parser.add_argument("--device", type=str, default="cpu", help="Device for retrieval model (cpu, cuda, or rocm)") # Updated help text
    # Renamed retrieval_model to embedding_model for consistency internally
    parser.add_argument("--embedding_model", type=str, default="colpali", help="Embedding model name (e.g., colpali, all-minilm, models/embedding-001, openai/text-embedding-ada-002)")
    parser.add_argument("--top_k", type=int, default=None, help="Number of local docs to retrieve") # Default None to detect if set
    parser.add_argument("--web_search", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable web search") # Use BooleanOptionalAction
    parser.add_argument("--personality", type=str, default=None, help="Optional personality for LLM (e.g. cheerful)")
    parser.add_argument("--rag_model", type=str, choices=["gemma", "pali", "gemini", "openrouter"], default=None, help="Which model to use for final RAG steps") # Default None
    parser.add_argument("--gemma_model_id", type=str, default=None, help="Specific model ID for Gemma/Ollama")
    parser.add_argument("--gemini_model_id", type=str, default=None, help="Specific model ID for Gemini")
    parser.add_argument("--openrouter_model_id", type=str, default=None, help="Specific model ID for OpenRouter")
    parser.add_argument("--max_depth", type=int, default=None, help="Depth limit for subquery expansions") # Default None
    # Search Provider Arguments
    parser.add_argument("--search_provider", type=str, choices=["duckduckgo", "searxng"], default=None, help="Search provider to use")
    parser.add_argument("--searxng_url", type=str, default=None, help="Base URL for your SearXNG instance")
    parser.add_argument("--search_max_results", type=int, default=None, help="Number of search results to fetch")
    # GUI Argument
    parser.add_argument("--gui", action="store_true", help="Launch the graphical user interface") # Add GUI flag
    parser.add_argument('--clear-cache', action='store_true', help='Clear the cache database before running.') # Added clear-cache flag
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level.') # Added log level arg

    # Scraping Arguments
    scrape_group = parser.add_argument_group('Scraping Options')
    scrape_group.add_argument('--scrape-url', type=str, help='URL to scrape and add to the knowledge base. If set, runs in scrape mode instead of search/GUI.')
    # scrape_group.add_argument('--scrape-depth', type=int, default=0, help='Recursion depth for scraping (0 for single page). Not yet implemented.') # Add when depth is supported
    scrape_group.add_argument('--scrape-ignore-robots', action='store_true', help='Ignore robots.txt rules when scraping.')

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Get logger for main


    # --- Load Config and Resolve Settings (Needed for both modes) ---
    # Use the load_config function which now merges with defaults
    config = load_config(args.config)
    # Defaults are now handled within load_config, but we might still need DEFAULT_CONFIG for reference if needed elsewhere
    defaults = DEFAULT_CONFIG # Use the imported defaults

    # Helper to safely get nested config values
    def get_config_value(cfg, keys, default=None):
        val = cfg
        try:
            for key in keys:
                val = val[key]
            # Return None if the config value is explicitly null/None
            return val if val is not None else default
        except (KeyError, TypeError):
            return default

    # Resolve settings: CLI > Config > Defaults
    # Simplified resolution for scraping, focusing on necessary parts
    resolved = {
        'device': args.device if args.device is not None else get_config_value(config, ['general', 'device'], defaults['general']['device']),
        'embedding_model': args.embedding_model if args.embedding_model is not None else get_config_value(config, ['retrieval', 'embedding_model'], defaults['retrieval']['embedding_model']),
        # Add other necessary resolved settings if needed by embedder/KB
    }

    # --- Scraping Mode ---
    if args.scrape_url:
        logger.info(f"--- Running in Scrape Mode for URL: {args.scrape_url} ---")

        # 1. Initialize Embedder
        try:
            # Progress callback for embedder init (optional)
            def embedder_progress(msg): logger.info(f"[Embedder] {msg}")
            embedder = create_embedder(
                model_name_or_path=resolved['embedding_model'],
                device=resolved['device'],
                progress_callback=embedder_progress
            )
            if not embedder:
                 raise ValueError("Embedder creation failed.")
            logger.info(f"Embedder '{resolved['embedding_model']}' initialized successfully on device '{resolved['device']}'.")
        except Exception as e:
            logger.critical(f"Failed to initialize embedder: {e}", exc_info=True)
            sys.exit(1)

        # 2. Initialize Knowledge Base
        try:
            # Progress callback for KB (optional)
            def kb_progress(msg): logger.info(f"[KnowledgeBase] {msg}")
            # Note: KB build_from_directory is not called here, we're just adding scraped content
            kb = KnowledgeBase(embedder=embedder, progress_callback=kb_progress)
            logger.info("KnowledgeBase initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize KnowledgeBase: {e}", exc_info=True)
            sys.exit(1)

        # 3. Perform Scraping
        logger.info(f"Starting scrape for {args.scrape_url} (Respect robots.txt: {not args.scrape_ignore_robots})")
        markdown_content, error_msg = scrape_url_to_markdown(
            url=args.scrape_url,
            respect_robots=not args.scrape_ignore_robots # Function expects respect_robots flag
        )

        # 4. Add to Knowledge Base if successful
        if error_msg:
            logger.error(f"Scraping failed: {error_msg}")
            sys.exit(1)
        elif markdown_content:
            logger.info("Scraping successful. Adding content to knowledge base...")
            success = kb.add_scraped_content(args.scrape_url, markdown_content)
            if success:
                logger.info(f"Successfully added content from {args.scrape_url} to the knowledge base.")
                # TODO: Persist the knowledge base if it's not in-memory only?
                # Currently, the KB seems in-memory unless explicitly saved/loaded elsewhere.
                # This scraped content will be lost when the script exits unless KB persistence is added.
                logger.warning("Knowledge base is currently in-memory. Scraped content will be lost on exit unless persistence is implemented.")
                sys.exit(0)
            else:
                logger.error("Failed to add scraped content to the knowledge base.")
                sys.exit(1)
        else:
            logger.error("Scraping finished, but no content was returned and no specific error message.")
            sys.exit(1)

    # --- GUI Mode ---
    elif args.gui: # Use elif since scrape mode exits
        if not GUI_ENABLED:
             # Updated error message
             print("[ERROR] GUI dependencies (PyQt6) are not installed or GUI components (gui/main_window.py) are missing. Cannot launch GUI.")
             sys.exit(1)
        # Corrected indentation starts here (8 spaces)
        print("[INFO] Launching NanoSage-EG GUI...")
        app = QApplication(sys.argv)

        # --- Apply Stylesheet ---
        style_path = os.path.join(os.path.dirname(__file__), "gui", "style.qss")
        if os.path.exists(style_path):
            try:
                with open(style_path, "r") as f:
                    stylesheet = f.read()
                    app.setStyleSheet(stylesheet)
                print(f"[INFO] Loaded stylesheet from: {style_path}")
            except Exception as e:
                print(f"[WARN] Could not load or apply stylesheet from {style_path}: {e}")
        else:
            print(f"[WARN] Stylesheet not found at: {style_path}")
        # --- End Apply Stylesheet ---

        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())

    # --- CLI Mode (Search Session) ---
    else: # Neither scrape nor GUI mode
        print("[INFO] Running in Command-Line Interface mode (Search Session).")

        # Check if query is provided for CLI mode
        if not args.query:
            parser.error("--query is required when not using --gui or --scrape-url")

        # --- Resolve Full Settings for Search Session ---
        # Re-resolve settings here, including all necessary for SearchSession
        resolved = {
            'corpus_dir': args.corpus_dir if args.corpus_dir is not None else get_config_value(config, ['general', 'corpus_dir'], defaults['general']['corpus_dir']),
            'device': args.device if args.device is not None else get_config_value(config, ['general', 'device'], defaults['general']['device']),
            'max_depth': args.max_depth if args.max_depth is not None else get_config_value(config, ['general', 'max_depth'], defaults['general']['max_depth']),
            'web_search': args.web_search if args.web_search is not None else get_config_value(config, ['general', 'web_search'], defaults['general']['web_search']),

            'embedding_model': args.embedding_model if args.embedding_model is not None else get_config_value(config, ['retrieval', 'embedding_model'], defaults['retrieval']['embedding_model']),
            'top_k': args.top_k if args.top_k is not None else get_config_value(config, ['retrieval', 'top_k'], defaults['retrieval']['top_k']),

            'rag_model': args.rag_model if args.rag_model is not None else get_config_value(config, ['llm', 'rag_model'], defaults['llm']['rag_model']),
            'personality': args.personality if args.personality is not None else get_config_value(config, ['llm', 'personality'], defaults['llm']['personality']),
            'gemma_model_id': args.gemma_model_id if args.gemma_model_id is not None else get_config_value(config, ['llm', 'gemma_model_id'], defaults['llm']['gemma_model_id']),
            'gemini_model_id': args.gemini_model_id if args.gemini_model_id is not None else get_config_value(config, ['llm', 'gemini_model_id'], defaults['llm']['gemini_model_id']),
            'openrouter_model_id': args.openrouter_model_id if args.openrouter_model_id is not None else get_config_value(config, ['llm', 'openrouter_model_id'], defaults['llm']['openrouter_model_id']),

            # API Keys: Config > Env Var > Default (None)
            'gemini_api_key': get_config_value(config, ['api_keys', 'gemini_api_key']) or os.getenv("GEMINI_API_KEY"),
            'openrouter_api_key': get_config_value(config, ['api_keys', 'openrouter_api_key']) or os.getenv("OPENROUTER_API_KEY"),

            # Search Settings Resolution
            'search_provider': args.search_provider if args.search_provider is not None else get_config_value(config, ['search', 'provider'], defaults['search']['provider']),
            'searxng_url': args.searxng_url if args.searxng_url is not None else get_config_value(config, ['search', 'searxng', 'base_url'], defaults['search']['searxng']['base_url']),
            # Resolve max_results based on the chosen provider
            'search_max_results': args.search_max_results if args.search_max_results is not None else None, # Placeholder, resolved below

            # Add cache settings resolution (CLI overrides config)
            'cache': {
                 'enabled': get_config_value(config, ['cache', 'enabled'], defaults['cache']['enabled']),
                 'db_path': get_config_value(config, ['cache', 'db_path'], defaults['cache']['db_path'])
            }
        }

        # Resolve search_max_results based on the selected provider
        provider = resolved['search_provider']
        if resolved['search_max_results'] is None: # Only resolve from config/defaults if not set by CLI
            if provider == 'searxng':
                resolved['search_max_results'] = get_config_value(config, ['search', 'searxng', 'max_results'], defaults['search']['searxng']['max_results'])
            else: # Default to duckduckgo settings
                resolved['search_max_results'] = get_config_value(config, ['search', 'duckduckgo', 'max_results'], defaults['search']['duckduckgo']['max_results'])

        # --- Validate SearXNG URL if selected ---
        if resolved['search_provider'] == 'searxng' and not resolved['searxng_url']:
            print("[ERROR] SearXNG is selected as the provider, but searxng_url is not set in config or via --searxng_url argument.")
            sys.exit(1)

        # --- Model Selection/Validation Logic (using resolved settings) ---
        # (Keep the existing model selection logic here)
        # Gemini Model Selection (if rag_model is gemini)
        if resolved['rag_model'] == "gemini":
            if not resolved['gemini_model_id']: # If no specific model ID is set via CLI or config
                print("[INFO] Gemini selected, but no specific model ID set. Fetching available models...")
                # Use resolved API key for listing
                available_models = list_gemini_models(gemini_api_key=resolved['gemini_api_key'])

                if available_models is None:
                    print("[ERROR] Could not retrieve Gemini models. Please check API key (config/env) and network connection.")
                    sys.exit(1)
                elif not available_models:
                    print("[ERROR] No Gemini models supporting 'generateContent' found.")
                    print("[WARN] Defaulting to 'models/gemini-1.5-flash-latest'. This might not work.")
                    resolved['gemini_model_id'] = defaults['llm']['gemini_model_id'] # Use default
                else:
                    print("Available Gemini Models:")
                    for i, model_name in enumerate(available_models):
                        print(f"  {i + 1}: {model_name}")
                    while True:
                        try:
                            choice = input("Please select a Gemini model number: ")
                            choice_idx = int(choice) - 1
                            if 0 <= choice_idx < len(available_models):
                                resolved['gemini_model_id'] = available_models[choice_idx]
                                break
                            else:
                                print(f"[ERROR] Invalid choice. Please enter a number between 1 and {len(available_models)}.")
                        except ValueError:
                            print("[ERROR] Invalid input. Please enter a number.")
                        except EOFError:
                             print("\n[ERROR] Input cancelled. Exiting.")
                             sys.exit(1)
            print(f"[INFO] Using Gemini model: {resolved['gemini_model_id']}")

        # OpenRouter Model Check (if rag_model is openrouter)
        elif resolved['rag_model'] == "openrouter":
            if not resolved['openrouter_model_id']:
                # If not set by CLI or config, use the default
                resolved['openrouter_model_id'] = defaults['llm']['openrouter_model_id']
                print(f"[WARN] No OpenRouter model specified via CLI or config. Defaulting to: {resolved['openrouter_model_id']}")
                # Optionally add a check here using list_openrouter_models if needed
            print(f"[INFO] Using OpenRouter model: {resolved['openrouter_model_id']}")

        # Gemma/Pali Model Check (if rag_model is gemma or pali)
        elif resolved['rag_model'] in ["gemma", "pali"]:
             if not resolved['gemma_model_id']:
                  resolved['gemma_model_id'] = defaults['llm']['gemma_model_id']
                  print(f"[WARN] No Gemma model specified via CLI or config. Defaulting to: {resolved['gemma_model_id']}")
             print(f"[INFO] Using Gemma/Ollama model: {resolved['gemma_model_id']}")


        # --- Initialize Cache Manager ---
        cache_manager = None
        cache_config = resolved.get('cache', {}) # Get cache settings from resolved dict
        if cache_config.get('enabled', False):
            db_path = cache_config.get('db_path', 'cache/nanosage_cache.db')
            try:
                cache_manager = CacheManager(db_path)
                if args.clear_cache:
                    logger.info("Clearing cache as requested by --clear-cache flag.")
                    cache_manager.clear_all_cache() # This handles closing/reopening connection
            except Exception as e:
                 logger.error(f"Failed to initialize cache manager at {db_path}: {e}. Caching disabled.")
                 cache_manager = None # Ensure it's None if init fails
        else:
            logger.info("Caching is disabled in configuration.")


        # --- Instantiate SearchSession with resolved settings and cache manager ---
        try:
            session = SearchSession(
                query=args.query,
                # config=config, # Pass raw config if needed by session, otherwise resolved_settings is better
                resolved_settings=resolved, # Pass the dictionary of resolved settings
                cache_manager=cache_manager # Pass the cache manager instance
                # progress_callback can be added here if needed for CLI mode
            )

            # --- Run Session ---
            # Use asyncio.run() for modern Python async handling
            if sys.version_info >= (3, 7):
                final_answer = asyncio.run(session.run_session())
            else:
                # Fallback for older Python versions (though 3.8+ is recommended by README)
                loop = asyncio.get_event_loop()
                final_answer = loop.run_until_complete(session.run_session())


            # --- Save Report ---
            if final_answer:
                output_path = session.save_report(final_answer)
                logger.info(f"Final report saved to: {output_path}")
            else:
                logger.warning("Session finished but no final answer was generated.")

        except Exception as e:
            logger.critical(f"An unexpected error occurred during the session: {e}", exc_info=True)
            # Optionally add more specific error handling
        finally:
            # --- Close Cache Connection ---
            if cache_manager:
                cache_manager.close()


if __name__ == "__main__":
    main()
