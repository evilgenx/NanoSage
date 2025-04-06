#!/usr/bin/env python3
# gui/workers.py

import os
import asyncio
import traceback
import re # Added for filename sanitization
import urllib.parse # Added for filename sanitization
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QMutex, QMutexLocker # Added Mutex

# Assuming search_session, llm_utils, and config_utils are accessible from the parent directory
try:
    from search_session import SearchSession
    # Import from the new provider modules
    from llm_providers.gemini import list_gemini_models
    from llm_providers.openrouter import list_openrouter_models
    # Import task functions
    from llm_providers.tasks import (
        extract_topics_from_text, chain_of_thought_query_enhancement,
        refine_text_section # Added refine_text_section
    )
    from config_utils import load_config, save_config, DEFAULT_CONFIG # Added DEFAULT_CONFIG
    # Imports needed for ScrapeWorker
    from web_scraper import scrape_url_to_markdown, can_fetch, DEFAULT_USER_AGENT # Added can_fetch, DEFAULT_USER_AGENT
    from knowledge_base import KnowledgeBase
    from embeddings.factory import create_embedder
    # Additional imports for recursive scraping
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse
    import time
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
    # Existing signals
    progress_updated = pyqtSignal(str) # For general status messages
    # Emits: report_path (str), final_answer_content (str), toc_tree_nodes (list)
    search_complete = pyqtSignal(str, str, list)
    error_occurred = pyqtSignal(str)

    # New signals for TOC visualization
    # Emits node data dictionary (from TOCNode.to_dict())
    tocNodeAdded = pyqtSignal(dict)
    # Emits node_id (str) and dictionary of updated fields (e.g., {'status': 'Done', 'relevance': '0.85'})
    tocNodeUpdated = pyqtSignal(str, dict)


    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.include_visuals = params.get("include_visuals", False) # Store the new parameter
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
                # Keep existing signal for string messages
                progress_signal = pyqtSignal(str)
                # Add signals for structured TOC updates
                toc_add_signal = pyqtSignal(dict)
                toc_update_signal = pyqtSignal(str, dict)

                def __call__(self, message):
                    # Check cancellation first
                    if self.parent().is_cancellation_requested():
                        return # Stop emitting if cancelled

                    if isinstance(message, dict):
                        # Handle structured messages
                        msg_type = message.get("type")
                        if msg_type == "toc_add":
                            node_data = message.get("node_data", {})
                            if node_data:
                                self.toc_add_signal.emit(node_data)
                        elif msg_type == "toc_update":
                            node_id = message.get("node_id")
                            # Create dict of updates, excluding 'type' and 'node_id'
                            updates = {k: v for k, v in message.items() if k not in ["type", "node_id"]}
                            if node_id and updates:
                                self.toc_update_signal.emit(node_id, updates)
                        elif msg_type == "status":
                            status_msg = message.get("message", "")
                            if status_msg:
                                self.progress_signal.emit(status_msg)
                        else:
                            # Fallback for unknown dict types - maybe log?
                            self.progress_signal.emit(f"[Worker] Received unknown structured message: {message}")
                    elif isinstance(message, str):
                        # Handle simple string messages
                        self.progress_signal.emit(message)
                    else:
                        # Handle unexpected types
                        self.progress_signal.emit(f"[Worker] Received unexpected progress type: {type(message)}")


                def set_parent_worker(self, worker):
                    # Store reference to parent worker to access is_cancellation_requested
                    self._parent_worker = worker
                def parent(self):
                    return self._parent_worker


            self._progress_callback_proxy = ProgressCallbackProxy()
            self._progress_callback_proxy.set_parent_worker(self) # Give proxy access to parent
            # Connect the proxy's signals to the worker's signals
            self._progress_callback_proxy.progress_signal.connect(self.progress_updated.emit)
            self._progress_callback_proxy.toc_add_signal.connect(self.tocNodeAdded.emit)
            self._progress_callback_proxy.toc_update_signal.connect(self.tocNodeUpdated.emit)


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
                'include_visuals': self.include_visuals, # Pass the stored value
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
            # Emit all necessary data: path, content, and toc nodes
            self.search_complete.emit(output_path, final_answer, session.toc_tree)

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


class RefinementWorker(QThread):
    """Refines a text section using an LLM in a separate thread."""
    # Emits the original anchor ID and the refined content string
    refinement_complete = pyqtSignal(str, str)
    refinement_error = pyqtSignal(str, str) # Emits anchor ID and error message
    status_update = pyqtSignal(str) # For status messages

    def __init__(self, anchor_id, section_content, instruction, llm_config, parent=None):
        super().__init__(parent)
        self.anchor_id = anchor_id
        self.section_content = section_content
        self.instruction = instruction
        self.llm_config = llm_config

    def run(self):
        """Executes the refinement task."""
        try:
            self.status_update.emit(f"Starting refinement for section '{self.anchor_id}'...")

            # Call the refinement task function
            refined_content = refine_text_section(
                section_content=self.section_content,
                instruction=self.instruction,
                llm_config=self.llm_config
            )

            if refined_content.startswith("Error:"):
                self.status_update.emit(f"[Error] Refinement failed for section '{self.anchor_id}'.")
                self.refinement_error.emit(self.anchor_id, refined_content)
            else:
                self.status_update.emit(f"Refinement complete for section '{self.anchor_id}'.")
                self.refinement_complete.emit(self.anchor_id, refined_content)

        except Exception as e:
            traceback.print_exc()
            error_msg = f"An unexpected error occurred during refinement for section '{self.anchor_id}': {e}"
            self.status_update.emit(f"[Error] {error_msg}")
            self.refinement_error.emit(self.anchor_id, error_msg)


class ScrapeWorker(QThread):
    """Scrapes a URL and adds its content to the knowledge base."""
    status_update = pyqtSignal(str)
    # Emits URL and a snippet of the scraped content on success
    scrape_complete = pyqtSignal(str, str)
    # Emits URL and error message on failure
    scrape_error = pyqtSignal(str, str)

    def __init__(self, url, ignore_robots, depth, embedding_model, device, parent=None): # Added depth
        super().__init__(parent)
        self.url = url
        self.ignore_robots = ignore_robots
        self.depth = depth # Store depth
        self.embedding_model = embedding_model
        self.device = device
        self._mutex = QMutex()
        self._cancellation_requested = False # Basic cancellation flag

    def stop(self):
        """Request cancellation."""
        # Basic cancellation for now, more robust mechanism might be needed
        # depending on where the time is spent (requests vs embedding)
        with QMutexLocker(self._mutex):
            self._cancellation_requested = True
        self.status_update.emit("Scrape cancellation requested.")

    def _is_cancelled(self):
        """Check cancellation flag."""
        with QMutexLocker(self._mutex):
            return self._cancellation_requested

    def _scrape_recursively(self, start_url, max_depth, respect_robots):
        """Helper function to perform recursive scraping."""
        visited = set()
        urls_to_scrape = [(start_url, 0)] # Queue of (url, current_depth)
        all_markdown_content = ""
        scraped_count = 0
        max_pages_to_scrape = 20 # Safety limit to prevent excessive scraping

        while urls_to_scrape and scraped_count < max_pages_to_scrape:
            if self._is_cancelled():
                self.status_update.emit("Scraping cancelled during recursion.")
                return None, "Cancelled"

            current_url, current_depth = urls_to_scrape.pop(0)

            if current_url in visited:
                continue

            if current_depth > max_depth:
                continue

            visited.add(current_url)
            self.status_update.emit(f"Scraping (Depth {current_depth}): {current_url}")

            # Use the existing scrape_url_to_markdown for individual pages
            markdown_content, error_msg = scrape_url_to_markdown(
                url=current_url,
                respect_robots=respect_robots
            )

            if error_msg:
                self.status_update.emit(f"[Warning] Failed to scrape {current_url}: {error_msg}")
                continue # Skip this URL, but continue with others

            if markdown_content:
                all_markdown_content += f"\n\n## Content from: {current_url}\n\n" + markdown_content
                scraped_count += 1

                # Find links if we need to go deeper
                if current_depth < max_depth:
                    try:
                        # Need to fetch again to parse HTML for links (scrape_url_to_markdown only returns markdown)
                        headers = {'User-Agent': DEFAULT_USER_AGENT}
                        response = requests.get(current_url, headers=headers, timeout=15)
                        response.raise_for_status()
                        content_type = response.headers.get('content-type', '').lower()

                        if 'html' in content_type:
                            soup = BeautifulSoup(response.content, 'lxml')
                            base_url_parts = urlparse(current_url)

                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                next_url = urljoin(current_url, href)
                                # Basic validation: same domain, http/https, not an anchor
                                next_url_parts = urlparse(next_url)
                                if (next_url_parts.scheme in ['http', 'https'] and
                                        next_url_parts.netloc == base_url_parts.netloc and
                                        next_url not in visited and
                                        '#' not in next_url): # Avoid fragments

                                    # Check robots.txt for the next URL before adding
                                    if respect_robots:
                                        time.sleep(0.1) # Small delay
                                        if not can_fetch(next_url):
                                            self.status_update.emit(f"[Info] Skipping disallowed link: {next_url}")
                                            continue

                                    if (next_url, current_depth + 1) not in urls_to_scrape:
                                         urls_to_scrape.append((next_url, current_depth + 1))
                        else:
                            self.status_update.emit(f"[Info] Skipping link extraction (non-HTML): {current_url}")

                    except requests.exceptions.RequestException as e:
                        self.status_update.emit(f"[Warning] Failed to fetch {current_url} for link extraction: {e}")
                    except Exception as e:
                         self.status_update.emit(f"[Warning] Error extracting links from {current_url}: {e}")

            time.sleep(0.2) # Be polite between scrapes

        if scraped_count >= max_pages_to_scrape:
             self.status_update.emit(f"[Warning] Reached maximum page limit ({max_pages_to_scrape}). Stopping recursion.")

        return all_markdown_content, None # Return aggregated content


    def run(self):
        """Executes the scraping (potentially recursive) and knowledge base addition."""
        try:
            self.status_update.emit(f"Initializing components for scraping {self.url} (Depth: {self.depth})...")

            # 1. Initialize Embedder
            # Note: This might be slow depending on the model. Consider if KB should be shared.
            # For now, create instances here.
            def embedder_progress(msg): self.status_update.emit(f"[Embedder] {msg}")
            # Corrected indentation for the create_embedder call and subsequent lines
            embedder = create_embedder(
                embedding_model_name=self.embedding_model, # Corrected parameter name
                device=self.device,
                # Note: create_embedder doesn't accept progress_callback directly
                # Progress is handled internally or via cache_manager if implemented
            )
            if not embedder:
                raise ValueError("Embedder creation failed.")
            self.status_update.emit(f"Embedder '{self.embedding_model}' initialized on {self.device}.")

            if self._is_cancelled():
                return

            # 2. Initialize Knowledge Base (In-memory for now)
            # WARNING: This KB instance is temporary and local to this worker.
            # The scraped content will only exist in memory during the app's runtime
            # unless persistence is added to the KnowledgeBase class itself or
            # the main application manages a persistent KB instance passed to workers.
            def kb_progress(msg): self.status_update.emit(f"[KnowledgeBase] {msg}")
            kb = KnowledgeBase(embedder=embedder, progress_callback=kb_progress)
            self.status_update.emit("Temporary in-memory KnowledgeBase initialized.")

            if self._is_cancelled():
                return

            # 3. Perform Scraping (potentially recursive)
            self.status_update.emit(f"Starting scrape for {self.url} (Depth: {self.depth}, Ignore robots: {self.ignore_robots})")

            # Call the recursive helper or the single scrape function based on depth
            if self.depth == 0:
                # Original single-page scrape
                markdown_content, error_msg = scrape_url_to_markdown(
                    url=self.url,
                    respect_robots=not self.ignore_robots
                )
            else:
                # New recursive scrape
                markdown_content, error_msg = self._scrape_recursively(
                    start_url=self.url,
                    max_depth=self.depth,
                    respect_robots=not self.ignore_robots
                )


            if self._is_cancelled():
                return

            # 4. Add to Knowledge Base if successful
            if error_msg:
                # Handle cancellation message specifically
                if error_msg == "Cancelled":
                    self.status_update.emit(f"Scraping cancelled for {self.url}.")
                    # Don't emit scrape_error for user cancellation
                else:
                    self.scrape_error.emit(self.url, error_msg)
            elif markdown_content:
                self.status_update.emit("Scraping successful. Saving aggregated content and adding to knowledge base...")

                # --- Save aggregated scraped content to file ---
                output_dir = "scraped_markdown"
                try:
                    os.makedirs(output_dir, exist_ok=True)

                    # Sanitize URL to create a safe filename
                    parsed_url = urllib.parse.urlparse(self.url)
                    # Use netloc + path, replace slashes, remove scheme
                    filename_base = f"{parsed_url.netloc}{parsed_url.path}".replace('/', '_').replace(':', '_')
                    # Remove potentially problematic characters
                    filename_base = re.sub(r'[^\w\-_\.]', '', filename_base)
                    # Truncate if too long (optional, but good practice)
                    filename_base = filename_base[:100]
                    # Add depth indication to filename if recursive
                    depth_suffix = f"_depth{self.depth}" if self.depth > 0 else ""
                    filename = f"{filename_base}{depth_suffix}.md"
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    self.status_update.emit(f"Aggregated scraped content saved to: {filepath}")

                except Exception as e:
                    self.status_update.emit(f"[Error] Failed to save aggregated scraped content to file: {e}")
                    # Continue to add to KB even if saving fails? Or emit error?
                    # For now, just log and continue.

                # --- Add aggregated content to Knowledge Base ---
                # Use the starting URL as the identifier for the aggregated content
                success = kb.add_scraped_content(self.url, markdown_content)

                if self._is_cancelled():
                    return

                if success:
                    snippet = markdown_content[:150].replace('\n', ' ') + "..." # Slightly longer snippet for aggregated content
                    self.scrape_complete.emit(self.url, snippet)
                    # Removed the inaccurate warning about in-memory KB loss
                else:
                    self.scrape_error.emit(self.url, "Failed to add aggregated scraped content to the knowledge base.")
            else:
                 # Check if error_msg was set (e.g., "Cancelled") before emitting generic error
                 if not error_msg:
                     self.scrape_error.emit(self.url, "Scraping returned no content and no specific error.")

        except Exception as e:
            if self._is_cancelled():
                self.status_update.emit("Scraping cancelled during exception handling.")
                return
            traceback.print_exc()
            self.scrape_error.emit(self.url, f"An unexpected error occurred: {e}")
        # Removed extra lines causing indentation errors at the end of the try block
