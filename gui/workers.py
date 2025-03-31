#!/usr/bin/env python3
# gui/workers.py

import os
import asyncio
import traceback
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QMutex, QMutexLocker # Added Mutex

# Assuming search_session, llm_utils, and config_utils are accessible from the parent directory
try:
    from search_session import SearchSession
    # Import from the new provider modules
    from llm_providers.gemini import list_gemini_models
    from llm_providers.openrouter import list_openrouter_models
    # Import task functions
    from llm_providers.tasks import extract_topics_from_text, chain_of_thought_query_enhancement
    from config_utils import load_config, save_config
except ImportError as e:
    # This error might be better handled by the main application window
    # or logged, rather than exiting here.
    print(f"Error importing from parent modules in workers.py: {e}")
    # Consider raising an exception or emitting an error signal if this happens
    # during runtime, rather than exiting the whole process.
    # sys.exit(1) # Avoid exiting from here

# --- Worker Threads ---

class SearchWorker(QThread):
    """Runs the SearchSession in a separate thread."""
    progress_updated = pyqtSignal(str)
    search_complete = pyqtSignal(str) # Emits the report path
    error_occurred = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self._progress_callback_proxy = None # To hold the proxy object
        self._mutex = QMutex()
        self._cancellation_requested = False

    def request_cancellation(self):
        """Sets the cancellation flag."""
        with QMutexLocker(self._mutex):
            self._cancellation_requested = True
            # self.progress_updated.emit("Cancellation flag set.") # Optional debug message

    def is_cancellation_requested(self):
        """Checks if cancellation has been requested."""
        with QMutexLocker(self._mutex):
            return self._cancellation_requested

    def run(self):
        """Executes the search session."""
        # Reset cancellation flag at the start of each run
        with QMutexLocker(self._mutex):
            self._cancellation_requested = False

        try:
            # Create a proxy object to safely emit signals from the asyncio loop
            class ProgressCallbackProxy(QObject):
                progress_signal = pyqtSignal(str)

                def __call__(self, message):
                    # Check cancellation before emitting progress
                    if self.parent().is_cancellation_requested():
                         # Optional: could raise a specific exception here if needed
                         # raise asyncio.CancelledError("Search cancelled by user")
                         return # Or just stop emitting
                    self.progress_signal.emit(message)

                def set_parent_worker(self, worker):
                    # Store reference to parent worker to access is_cancellation_requested
                    self._parent_worker = worker
                def parent(self):
                    return self._parent_worker


            self._progress_callback_proxy = ProgressCallbackProxy()
            self._progress_callback_proxy.set_parent_worker(self) # Give proxy access to parent
            self._progress_callback_proxy.progress_signal.connect(self.progress_updated.emit)


            # Check cancellation before starting
            if self.is_cancellation_requested():
                self.progress_updated.emit("Search cancelled before starting.")
                return

            self.progress_updated.emit("Initializing search session...")

            # Load config (assuming config.yaml exists or is handled)
            config = load_config(self.params.get("config_path", "config.yaml"))

            # Determine which model was actually selected based on rag_model type
            rag_model_type = self.params.get("rag_model")
            selected_model_name = None
            if rag_model_type == "gemini":
                selected_model_name = self.params.get("selected_gemini_model")
            elif rag_model_type == "openrouter":
                # Pass the selected OpenRouter model name via the expected parameter
                selected_model_name = self.params.get("selected_openrouter_model")
            # Other RAG models like 'gemma', 'pali', 'None' don't need a specific selected model name here

            # --- Load selected prompt template ---
            output_format_name = self.params.get("output_format", "Report") # Default to Report if not provided
            output_formats_map = config.get('llm', {}).get('output_formats', {})
            template_path = output_formats_map.get(output_format_name)
            template_content = None

            if template_path and os.path.exists(template_path):
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                    self.progress_updated.emit(f"Loaded prompt template for '{output_format_name}' from {template_path}")
                except Exception as e:
                    self.progress_updated.emit(f"[Warning] Failed to read prompt template file {template_path}: {e}")
                    # Fallback logic below handles missing content
            else:
                self.progress_updated.emit(f"[Warning] Prompt template path not found for format '{output_format_name}' in config or file missing: {template_path}.")

            # Fallback to 'Report' template if the selected one failed or wasn't found
            if template_content is None and output_format_name != "Report":
                self.progress_updated.emit("Attempting to fall back to default 'Report' template...")
                default_template_path = output_formats_map.get("Report")
                if default_template_path and os.path.exists(default_template_path):
                    try:
                        with open(default_template_path, 'r', encoding='utf-8') as f:
                            template_content = f.read()
                        self.progress_updated.emit(f"Successfully loaded fallback 'Report' template from {default_template_path}")
                    except Exception as e:
                        self.progress_updated.emit(f"[Warning] Failed to read fallback 'Report' template {default_template_path}: {e}")
                else:
                    self.progress_updated.emit("[Warning] Default 'Report' template path not found or file missing.")

            # Final check: if no template could be loaded, abort.
            if template_content is None:
                 self.error_occurred.emit(f"Could not load any prompt template (tried '{output_format_name}' and fallback 'Report'). Cannot proceed with RAG.")
                 return # Stop the worker

            # --- Prepare resolved_settings dictionary ---
            resolved_settings = {
                'rag_prompt_template_content': template_content, # Add the loaded template content
                'corpus_dir': self.params.get("corpus_dir"),
                'device': self.params.get("device", "cpu"),
                'max_depth': self.params.get("max_depth", 1),
                'web_search': self.params.get("web_search", False),
                'enable_iterative_search': self.params.get("enable_iterative_search", False), # Add the new setting
                'embedding_model': self.params.get("embedding_model_name", "colpali"), # Use 'embedding_model' key
                'top_k': self.params.get("top_k", 3),
                'rag_model': rag_model_type, # RAG model type (gemma, gemini, openrouter, etc.)
                'personality': self.params.get("personality"),
                'gemma_model_id': None, # GUI doesn't explicitly set this, SearchSession might default
                'gemini_model_id': selected_model_name if rag_model_type == "gemini" else None,
                'openrouter_model_id': selected_model_name if rag_model_type == "openrouter" else None,
                # API keys are assumed to be handled by SearchSession/llm_utils using config/env
                'gemini_api_key': config.get('api_keys', {}).get('gemini_api_key') or os.getenv("GEMINI_API_KEY"),
                'openrouter_api_key': config.get('api_keys', {}).get('openrouter_api_key') or os.getenv("OPENROUTER_API_KEY"),
                # Add search provider settings passed from MainWindow
                'search_provider': self.params.get("search_provider", "duckduckgo"),
                'search_max_results': self.params.get("search_limit", 5), # Used by DDG directly
                'searxng_url': self.params.get("searxng_url"),
                # Add the other SearXNG params passed from MainWindow
                'searxng_time_range': self.params.get("searxng_time_range"),
                'searxng_categories': self.params.get("searxng_categories"),
                'searxng_engines': self.params.get("searxng_engines"),
            }

            # --- Instantiate SearchSession correctly ---
            # NOTE: Assumes SearchSession.__init__ accepts cancellation_check_callback
            # This callback should be checked within SearchSession's async methods.
            session = SearchSession(
                query=self.params["query"],
                config=config, # Pass raw config
                resolved_settings=resolved_settings, # Pass the resolved settings dictionary
                progress_callback=self._progress_callback_proxy # Pass the callable proxy
                # Removed cancellation_check_callback from here
            )

            # Check cancellation before running session
            if self.is_cancellation_requested():
                self.progress_updated.emit("Search cancelled before running session.")
                return

            self.progress_updated.emit("Starting search process...")
            # Run the asyncio event loop within the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Pass cancellation check callback here
            final_answer = loop.run_until_complete(session.run_session(cancellation_check_callback=self.is_cancellation_requested))
            loop.close()

            # Check for cancellation *after* the loop finishes or is interrupted
            if self.is_cancellation_requested():
                self.progress_updated.emit("Search cancelled by user.")
                return # Skip saving report

            # Check cancellation before saving report
            if self.is_cancellation_requested():
                self.progress_updated.emit("Search cancelled before saving report.")
                return

            self.progress_updated.emit("Saving final report...")
            output_path = session.save_report(final_answer)
            self.progress_updated.emit(f"Report saved: {output_path}")
            self.search_complete.emit(output_path)

        except ImportError as e:
             # Check for cancellation before reporting error
             if self.is_cancellation_requested():
                 self.progress_updated.emit("Search cancelled during import error.")
                 return
             self.error_occurred.emit(f"Import Error: {e}. Check dependencies.")
        except FileNotFoundError as e:
             # Check for cancellation before reporting error
             if self.is_cancellation_requested():
                 self.progress_updated.emit("Search cancelled during file not found error.")
                 return
             self.error_occurred.emit(f"File Not Found Error: {e}")
        except asyncio.CancelledError:
             # Handle cancellation if raised within the async tasks
             self.progress_updated.emit("Search explicitly cancelled within async task.")
             return
        except Exception as e:
            # Check for cancellation before reporting error
            if self.is_cancellation_requested():
                self.progress_updated.emit("Search cancelled during execution.")
                return
            # Log the full traceback for better debugging
            traceback.print_exc()
            self.error_occurred.emit(f"An error occurred during search: {e}")
        finally:
            # Clean up proxy if it was created
            if self._progress_callback_proxy:
                # Check if disconnect is needed/safe
                try:
                    self._progress_callback_proxy.progress_signal.disconnect()
                except TypeError:
                    pass # Signal already disconnected
                self._progress_callback_proxy = None


class TopicExtractorWorker(QThread):
    """Extracts topics from text using an LLM in a separate thread."""
    topics_extracted = pyqtSignal(str) # Emits the extracted topics string
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str) # For status messages

    def __init__(self, text_to_analyze, llm_config, parent=None):
        super().__init__(parent)
        self.text_to_analyze = text_to_analyze
        self.llm_config = llm_config

    def run(self):
        """Executes the topic extraction task."""
        try:
            self.status_update.emit("Starting topic extraction...")
            # Call the function from llm_providers.tasks
            extracted_topics = extract_topics_from_text(
                text=self.text_to_analyze,
                llm_config=self.llm_config
                # max_topics could be added as a parameter if needed
            )

            if extracted_topics.startswith("Error:"):
                self.error_occurred.emit(f"Topic Extraction Failed: {extracted_topics}")
            else:
                self.status_update.emit("Topic extraction complete.")
                self.topics_extracted.emit(extracted_topics)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"An unexpected error occurred during topic extraction: {e}")


class QueryEnhancerWorker(QThread):
    """Enhances a query using an LLM in a separate thread for preview."""
    enhanced_query_ready = pyqtSignal(str) # Emits the enhanced query string
    enhancement_error = pyqtSignal(str)
    status_update = pyqtSignal(str) # For status messages

    def __init__(self, original_query, llm_config, parent=None):
        super().__init__(parent)
        self.original_query = original_query
        self.llm_config = llm_config

    def run(self):
        """Executes the query enhancement task."""
        try:
            self.status_update.emit("Enhancing query for preview...")
            # Call the function from llm_providers.tasks
            enhanced_query = chain_of_thought_query_enhancement(
                query=self.original_query,
                llm_config=self.llm_config
            )

            # Check if enhancement failed (returns original query on failure)
            if enhanced_query == self.original_query:
                 # Check if the original query was empty or if there was an actual error logged by the function
                 if not self.original_query.strip():
                     self.enhancement_error.emit("Cannot enhance empty query.")
                 else:
                     # Assume enhancement failed silently or returned original due to error
                     self.status_update.emit("[Warning] Query enhancement failed or returned original query. Using original.")
                     # Emit original query so the process can continue, but log it wasn't enhanced
                     self.enhanced_query_ready.emit(self.original_query) # Send original back
            elif enhanced_query.startswith("Error:"):
                 self.enhancement_error.emit(f"Query Enhancement Failed: {enhanced_query}")
            else:
                self.status_update.emit("Query enhancement preview ready.")
                self.enhanced_query_ready.emit(enhanced_query)

        except Exception as e:
            traceback.print_exc()
            self.enhancement_error.emit(f"An unexpected error occurred during query enhancement: {e}")


class GeminiFetcher(QThread):
    """Fetches available Gemini models in a separate thread."""
    models_fetched = pyqtSignal(list)
    fetch_error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def run(self):
        """Executes the *generative* model fetching."""
        try:
            self.status_update.emit("Fetching Gemini generative models...")
            # Load config to get API key
            config = load_config("config.yaml") # Assuming default config path
            api_key = config.get('api_keys', {}).get('gemini_api_key')
            # Pass the key to the listing function
            models = list_gemini_models(gemini_api_key=api_key)
            if models is None:
                # Error message from llm_utils should be more specific now
                self.fetch_error.emit("Could not retrieve Gemini generative models. Check API key/network/console.")
            elif not models:
                self.fetch_error.emit("No suitable Gemini generative models found.")
            else:
                self.models_fetched.emit(models)
        except Exception as e:
            self.fetch_error.emit(f"Error fetching Gemini generative models: {e}")

class OpenRouterFetcher(QThread):
    """Fetches available free OpenRouter *generative* models in a separate thread."""
    models_fetched = pyqtSignal(list)
    fetch_error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def run(self):
        """Executes the *generative* model fetching."""
        try:
            self.status_update.emit("Fetching free OpenRouter generative models...")
            models = list_openrouter_models() # Fetch generative models
            if models is None:
                self.fetch_error.emit("Could not retrieve OpenRouter generative models. Check console/network.")
            elif not models:
                self.fetch_error.emit("No free OpenRouter generative models found (based on pricing).")
            else:
                self.models_fetched.emit(models)
        except Exception as e:
            self.fetch_error.emit(f"Error fetching OpenRouter generative models: {e}")

# Removed GeminiEmbeddingFetcher and OpenRouterEmbeddingFetcher classes
