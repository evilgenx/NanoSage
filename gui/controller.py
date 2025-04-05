#!/usr/bin/env python3
# gui/controller.py

import logging
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

# Import worker threads using relative import
try:
    from .workers import (
        SearchWorker, GeminiFetcher, OpenRouterFetcher, TopicExtractorWorker,
        QueryEnhancerWorker
    )
except ImportError as e:
    print(f"Error importing workers in controller.py: {e}")
    # Handle appropriately, maybe raise or show error in GUI if possible later
    import sys
    sys.exit(1)

# Import config loading utility
from config_utils import load_config, save_config, DEFAULT_CONFIG
from cache_manager import CacheManager # Added CacheManager import

class GuiController(QObject):
    """
    Handles the application logic, worker management, and interaction
    between the UI (MainWindow) and the backend processes.
    """
    # Define signals if needed for communication back to MainWindow beyond direct calls
    # e.g., signal_update_report_path = pyqtSignal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window # Reference to the MainWindow instance
        self.config_path = main_window.config_path # Get config path from main window
        self.config_data = main_window.config_data # Use initial config data

        # Worker instances
        self.search_worker = None
        self.topic_extractor_worker = None
        self.query_enhancer_worker = None
        self.gemini_fetcher = None
        self.openrouter_fetcher = None
        # Cache manager instance (might be better managed elsewhere, but keep for now)
        self.cache_manager_instance = None

    def log_status(self, message):
        """Helper to call MainWindow's log_status."""
        if self.main_window:
            self.main_window.log_status(message)

    # --- Query Enhancement Slots ---
    def on_enhanced_query_ready(self, enhanced_query_preview):
        """Handles the signal when the enhanced query preview is ready."""
        self.log_status(f"Enhanced Query Preview: {enhanced_query_preview}")
        # IMPORTANT: Proceed with the search using the ORIGINAL query text
        # that was present in the input box when the enhancement started.
        original_query = self.main_window.query_input.toPlainText().strip() # Get original text again

        if not original_query:
             # This case should ideally be caught before starting the enhancer, but double-check
             self.log_status("[Error] Original query is empty after enhancement preview. Aborting search.")
             self.on_enhancement_error("Original query became empty.") # Treat as error
             return

        # Now, prepare parameters and start the actual SearchWorker
        self.log_status("Proceeding with search using the original query...")
        self._start_main_search_worker(original_query) # Pass original query to the search worker starter

    def on_enhancement_error(self, error_message):
        """Handles errors during query enhancement preview."""
        self.log_status(f"Query Enhancement Preview Error: {error_message}")
        QMessageBox.critical(self.main_window, "Query Enhancement Error", error_message)
        # Reset UI state via MainWindow reference
        self.main_window.run_button.setEnabled(True)
        self.main_window.progress_bar.setVisible(False)
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)
        self.query_enhancer_worker = None # Clear worker reference

    def on_enhancement_finished(self):
        """Called when the query enhancement thread finishes (success or error)."""
        # General cleanup if needed, though success/error slots might handle it.
        if self.query_enhancer_worker: # Check if it finished normally
            # UI state should be handled by success/error slots already
            self.query_enhancer_worker = None

    # --- Topic Extraction Slots ---
    def on_topics_extracted(self, topics_string):
        """Handles the signal when topics are successfully extracted."""
        self.log_status("Topics extracted successfully.")
        self.main_window.query_input.setPlainText(topics_string) # Populate input field
        self.main_window.extract_topics_checkbox.setChecked(False) # Uncheck the box
        # Re-enable run button, hide progress/cancel as extraction is done
        self.main_window.run_button.setEnabled(True)
        self.main_window.progress_bar.setVisible(False)
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)
        self.topic_extractor_worker = None # Clear worker reference
        QMessageBox.information(self.main_window, "Topics Extracted", "Extracted topics have been placed in the query box.\nReview or edit them, then click 'Run Search' again to search using these topics.")

    def on_topic_extraction_error(self, error_message):
        """Handles errors during topic extraction."""
        self.log_status(f"Topic Extraction Error: {error_message}")
        QMessageBox.critical(self.main_window, "Topic Extraction Error", error_message)
        # Reset UI state
        self.main_window.run_button.setEnabled(True)
        self.main_window.progress_bar.setVisible(False)
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)
        self.topic_extractor_worker = None # Clear worker reference
        # Keep the checkbox checked so user knows it failed
        # self.main_window.extract_topics_checkbox.setChecked(False)

    def on_topic_extraction_finished(self):
        """Called when the topic extraction thread finishes (success or error)."""
        if self.topic_extractor_worker: # Check if it finished normally or was cleared already
            # Reset UI only if enhancer isn't also running (unlikely scenario)
            if not (self.query_enhancer_worker and self.query_enhancer_worker.isRunning()):
                self.main_window.run_button.setEnabled(True)
                self.main_window.progress_bar.setVisible(False)
                self.main_window.cancel_button.setVisible(False)
                self.main_window.cancel_button.setEnabled(False)
            self.topic_extractor_worker = None

    # --- Generative Model Fetching Slots ---
    def fetch_gemini_models(self):
        """Start the GeminiFetcher thread for *generative* models."""
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
            self.log_status("Already fetching Gemini generative models...")
            return

        self.log_status("Attempting to fetch Gemini generative models (requires GEMINI_API_KEY)...")
        self.main_window.gemini_fetch_button.setEnabled(False)
        self.gemini_fetcher = GeminiFetcher(self.main_window) # Pass main_window as parent if needed by worker
        self.gemini_fetcher.status_update.connect(self.log_status)
        self.gemini_fetcher.models_fetched.connect(self.on_gemini_models_fetched)
        self.gemini_fetcher.fetch_error.connect(self.on_gemini_fetch_error)
        # Re-enable button only if combo is still empty after finish
        self.gemini_fetcher.finished.connect(lambda: self.main_window.gemini_fetch_button.setEnabled(self.main_window.gemini_model_combo.count() == 0))
        self.gemini_fetcher.start()

    def on_gemini_models_fetched(self, models):
        """Populate the Gemini *generative* model combo box."""
        self.log_status(f"Successfully fetched {len(models)} Gemini generative models.")
        self.main_window.gemini_model_combo.clear()
        self.main_window.gemini_model_combo.addItems(models)
        self.main_window.gemini_model_combo.setEnabled(True)
        self.main_window.gemini_fetch_button.setEnabled(False) # Disable after successful fetch

    def on_gemini_fetch_error(self, error_message):
        """Show error message if Gemini *generative* fetch fails."""
        self.log_status(f"Gemini Generative Fetch Error: {error_message}")
        self.main_window.gemini_model_combo.clear()
        self.main_window.gemini_model_combo.setEnabled(False)
        self.main_window.gemini_fetch_button.setEnabled(True) # Re-enable on error

    def fetch_openrouter_models(self):
        """Start the OpenRouterFetcher thread for *generative* models."""
        if self.openrouter_fetcher and self.openrouter_fetcher.isRunning():
            self.log_status("Already fetching OpenRouter generative models...")
            return

        self.log_status("Attempting to fetch free OpenRouter generative models...")
        self.main_window.openrouter_fetch_button.setEnabled(False)
        self.openrouter_fetcher = OpenRouterFetcher(self.main_window) # Pass main_window as parent
        self.openrouter_fetcher.status_update.connect(self.log_status)
        self.openrouter_fetcher.models_fetched.connect(self.on_openrouter_models_fetched)
        self.openrouter_fetcher.fetch_error.connect(self.on_openrouter_fetch_error)
        # Re-enable button only if combo is still empty after finish
        self.openrouter_fetcher.finished.connect(lambda: self.main_window.openrouter_fetch_button.setEnabled(self.main_window.openrouter_model_combo.count() == 0))
        self.openrouter_fetcher.start()

    def on_openrouter_models_fetched(self, models):
        """Populate the OpenRouter *generative* model combo box."""
        self.log_status(f"Successfully fetched {len(models)} free OpenRouter generative models.")
        self.main_window.openrouter_model_combo.clear()
        self.main_window.openrouter_model_combo.addItems(models)
        self.main_window.openrouter_model_combo.setEnabled(True)
        self.main_window.openrouter_fetch_button.setEnabled(False) # Disable after successful fetch

    def on_openrouter_fetch_error(self, error_message):
        """Show error message if OpenRouter *generative* fetch fails."""
        self.log_status(f"OpenRouter Generative Fetch Error: {error_message}")
        self.main_window.openrouter_model_combo.clear()
        self.main_window.openrouter_model_combo.setEnabled(False)
        self.main_window.openrouter_fetch_button.setEnabled(True) # Re-enable on error

    # --- Search Execution ---
    def start_search_process(self):
        """
        Public method called by MainWindow's run_button click.
        Validates inputs and starts the appropriate worker thread:
        1. Topic Extraction (if checkbox checked)
        2. Query Enhancement Preview (if checkbox unchecked)
        3. Main Search (triggered after step 1 or 2 completes)
        """
        # --- Input Validation ---
        # Check if any worker is running
        if (self.search_worker and self.search_worker.isRunning()) or \
           (self.topic_extractor_worker and self.topic_extractor_worker.isRunning()) or \
           (self.query_enhancer_worker and self.query_enhancer_worker.isRunning()):
            self.log_status("An operation (search, topic extraction, or enhancement preview) is already in progress.")
            return

        query_or_text = self.main_window.query_input.toPlainText().strip()
        if not query_or_text:
            QMessageBox.warning(self.main_window, "Input Error", "Please enter text in the query box.")
            return

        # --- Shared Validation for LLM Tasks (Extraction & Enhancement) ---
        rag_model_type = self.main_window.rag_model_combo.currentText()
        selected_generative_gemini = None
        selected_generative_openrouter = None

        # Check if LLM task is requested (extraction or enhancement)
        is_llm_task_needed = self.main_window.extract_topics_checkbox.isChecked() or True # Enhancement is default

        if is_llm_task_needed:
            if rag_model_type == "None":
                QMessageBox.warning(self.main_window, "Input Error", "Topic extraction or query enhancement preview requires a RAG model (Gemini, OpenRouter, or Gemma) to be selected.")
                return
            elif rag_model_type == "gemini":
                if self.main_window.gemini_model_combo.count() == 0 or not self.main_window.gemini_model_combo.currentText():
                    QMessageBox.warning(self.main_window, "Input Error", f"{'Topic extraction' if self.main_window.extract_topics_checkbox.isChecked() else 'Query enhancement'} needs a Gemini generative model. Please fetch and select one.")
                    return
                selected_generative_gemini = self.main_window.gemini_model_combo.currentText()
            elif rag_model_type == "openrouter":
                if self.main_window.openrouter_model_combo.count() == 0 or not self.main_window.openrouter_model_combo.currentText():
                    QMessageBox.warning(self.main_window, "Input Error", f"{'Topic extraction' if self.main_window.extract_topics_checkbox.isChecked() else 'Query enhancement'} needs an OpenRouter generative model. Please fetch and select one.")
                    return
                selected_generative_openrouter = self.main_window.openrouter_model_combo.currentText()

            # Prepare LLM config (used by both topic extractor and query enhancer)
            # Reload config data to get latest API keys etc.
            self.config_data = load_config(self.config_path)
            llm_config = {
                "provider": rag_model_type,
                "model_id": selected_generative_gemini or selected_generative_openrouter,
                "api_key": self.config_data.get('api_keys', {}).get(f'{rag_model_type}_api_key'),
                "personality": self.main_window.personality_input.text() or None
            }

        # --- Action based on Checkbox State ---
        if self.main_window.extract_topics_checkbox.isChecked():
            # --- Start Topic Extractor Worker ---
            self.log_status("Topic extraction requested...")
            self.log_status("Starting topic extraction worker...")
            self.main_window.run_button.setEnabled(False)
            self.main_window.progress_bar.setVisible(True)
            self.main_window.progress_bar.setRange(0, 0) # Indeterminate

            self.topic_extractor_worker = TopicExtractorWorker(query_or_text, llm_config, self.main_window) # Pass main_window as parent
            self.topic_extractor_worker.status_update.connect(self.log_status)
            self.topic_extractor_worker.topics_extracted.connect(self.on_topics_extracted)
            self.topic_extractor_worker.error_occurred.connect(self.on_topic_extraction_error)
            self.topic_extractor_worker.finished.connect(self.on_topic_extraction_finished)
            self.topic_extractor_worker.start()
            # Stop here, wait for extraction

        else:
            # --- Start Query Enhancement Preview Worker ---
            self.log_status("Starting query enhancement preview...")
            self.main_window.run_button.setEnabled(False)
            self.main_window.progress_bar.setVisible(True)
            self.main_window.progress_bar.setRange(0, 0) # Indeterminate

            self.query_enhancer_worker = QueryEnhancerWorker(query_or_text, llm_config, self.main_window) # Pass main_window as parent
            self.query_enhancer_worker.status_update.connect(self.log_status)
            self.query_enhancer_worker.enhanced_query_ready.connect(self.on_enhanced_query_ready)
            self.query_enhancer_worker.enhancement_error.connect(self.on_enhancement_error)
            self.query_enhancer_worker.finished.connect(self.on_enhancement_finished)
            self.query_enhancer_worker.start()
            # Stop here, wait for enhancement preview

    def _start_main_search_worker(self, query_to_use):
        """
        Internal method to prepare parameters and start the main SearchWorker.
        Called AFTER topic extraction or query enhancement preview.
        """
        self.log_status("Preparing to start main search worker...")

        # --- Standard Search Input Validation (Run again before starting search) ---
        embedding_device = self.main_window.device_combo.currentText()
        embedding_model_name = self.main_window.embedding_model_combo.currentText()
        if not embedding_model_name:
             QMessageBox.warning(self.main_window, "Input Error", f"Please select an Embedding Model for the '{embedding_device}' device.")
             self.main_window.run_button.setEnabled(True)
             self.main_window.progress_bar.setVisible(False)
             return

        rag_model_type = self.main_window.rag_model_combo.currentText()
        selected_generative_gemini = None
        selected_generative_openrouter = None
        if rag_model_type == "gemini":
            if self.main_window.gemini_model_combo.count() == 0 or not self.main_window.gemini_model_combo.currentText():
                 QMessageBox.warning(self.main_window, "Input Error", "RAG Model is Gemini, but no generative model selected. Please fetch and select one.")
                 self.main_window.run_button.setEnabled(True)
                 self.main_window.progress_bar.setVisible(False)
                 return
            selected_generative_gemini = self.main_window.gemini_model_combo.currentText()
        elif rag_model_type == "openrouter":
            if self.main_window.openrouter_model_combo.count() == 0 or not self.main_window.openrouter_model_combo.currentText():
                 QMessageBox.warning(self.main_window, "Input Error", "RAG Model is OpenRouter, but no generative model selected. Please fetch and select one.")
                 self.main_window.run_button.setEnabled(True)
                 self.main_window.progress_bar.setVisible(False)
                 return
            selected_generative_openrouter = self.main_window.openrouter_model_combo.currentText()

        # --- Save Current Config & Prepare Search Parameters ---
        self.config_data = load_config(self.config_path) # Reload fresh config

        # Update config_data with current UI settings before passing to worker
        # General
        general_cfg = self.config_data.setdefault('general', {})
        general_cfg['web_search'] = self.main_window.web_search_checkbox.isChecked()
        general_cfg['max_depth'] = self.main_window.max_depth_spinbox.value()
        # Retrieval
        retrieval_cfg = self.config_data.setdefault('retrieval', {})
        retrieval_cfg['top_k'] = self.main_window.top_k_spinbox.value()
        retrieval_cfg['embedding_model'] = embedding_model_name
        # Embeddings (Device) - Assuming 'embeddings' section exists
        embeddings_cfg = self.config_data.setdefault('embeddings', {})
        embeddings_cfg['device'] = embedding_device
        # Corpus
        corpus_cfg = self.config_data.setdefault('corpus', {})
        corpus_cfg['path'] = self.main_window.corpus_dir_label.text() or None
        # RAG / LLM
        llm_cfg = self.config_data.setdefault('llm', {})
        llm_cfg['rag_model'] = rag_model_type if rag_model_type != "None" else None
        llm_cfg['personality'] = self.main_window.personality_input.text() or None
        llm_cfg['gemini_model_id'] = selected_generative_gemini
        llm_cfg['openrouter_model_id'] = selected_generative_openrouter
        # Output Format
        llm_cfg['output_format'] = self.main_window.output_format_combo.currentText() # Save selected format
        # Cache
        cache_cfg = self.config_data.setdefault('cache', {})
        cache_cfg['enabled'] = self.main_window.cache_enabled_checkbox.isChecked()
        # Search Provider specific settings
        selected_provider_text = self.main_window.search_provider_combo.currentText()
        search_provider_key = 'duckduckgo' if selected_provider_text == "DuckDuckGo" else 'searxng'
        search_config = self.config_data.setdefault('search', {})
        search_config['provider'] = search_provider_key
        search_config['enable_iterative_search'] = self.main_window.iterative_search_checkbox.isChecked()

        searxng_config = search_config.setdefault('searxng', {})
        ddg_config = search_config.setdefault('duckduckgo', {})

        search_limit = 5 # Default
        searxng_url = None
        searxng_time_range = None
        searxng_categories = None
        searxng_engines = None

        if search_provider_key == 'searxng':
            searxng_url = self.main_window.searxng_base_url_input.text().strip() or None
            searxng_time_range = self.main_window.searxng_time_range_input.text().strip() or None
            searxng_categories = self.main_window.searxng_categories_input.text().strip() or None
            # Read engines directly from the selector widget's current state
            searxng_engines = self.main_window.searxng_engine_selector.getSelectedEngines()
            if not isinstance(searxng_engines, list): searxng_engines = []

            searxng_config['base_url'] = searxng_url
            searxng_config['time_range'] = searxng_time_range
            searxng_config['categories'] = searxng_categories
            searxng_config['engines'] = searxng_engines # Save current selection

            search_limit = searxng_config.get('max_results', 5)
        else: # duckduckgo
            search_limit = ddg_config.get('max_results', 5)

        # Save the potentially modified config_data before starting the worker
        if save_config(self.config_path, self.config_data):
            self.log_status(f"Current settings saved to {self.config_path}")
        else:
            self.log_status(f"[ERROR] Failed to save configuration before starting search: {self.config_path}")
            QMessageBox.warning(self.main_window, "Config Error", f"Could not save settings to {self.config_path}. Search may use outdated settings.")
            # return # Decide whether to stop

        # --- Prepare All Parameters for Search Worker ---
        search_params = {
            "query": query_to_use,
            "corpus_dir": corpus_cfg['path'],
            "web_search": general_cfg['web_search'],
            "enable_iterative_search": search_config['enable_iterative_search'],
            "max_depth": general_cfg['max_depth'],
            "top_k": retrieval_cfg['top_k'],
            "device": embeddings_cfg['device'],
            "embedding_model_name": retrieval_cfg['embedding_model'],
            "rag_model": llm_cfg['rag_model'],
            "personality": llm_cfg['personality'],
            "selected_gemini_model": llm_cfg['gemini_model_id'],
            "selected_openrouter_model": llm_cfg['openrouter_model_id'],
            "search_provider": search_config['provider'],
            "search_limit": search_limit,
            "searxng_url": searxng_url,
            "searxng_time_range": searxng_time_range,
            "searxng_categories": searxng_categories,
            "searxng_engines": searxng_engines,
            "config_path": self.config_path,
            "output_format": llm_cfg['output_format']
        }

        # --- Start Search Worker ---
        self.log_status("Starting main search worker...")
        # Update UI state via MainWindow
        self.main_window.run_button.setEnabled(False)
        self.main_window.open_report_button.setEnabled(False)
        self.main_window.open_folder_button.setEnabled(False)
        self.main_window.share_email_button.setEnabled(False)
        self.main_window.report_path_label.setText("Running Search...")
        self.main_window.current_report_path = None # Reset state in MainWindow
        self.main_window.current_results_dir = None # Reset state in MainWindow

        self.main_window.progress_bar.setVisible(True)
        self.main_window.progress_bar.setRange(0, 0) # Indeterminate
        self.main_window.cancel_button.setVisible(True)
        self.main_window.cancel_button.setEnabled(True)

        # Ensure search provider visibility is correct before starting
        self.main_window.handle_search_provider_change(self.main_window.search_provider_combo.currentText())

        self.search_worker = SearchWorker(search_params, self.main_window) # Pass main_window as parent
        self.search_worker.progress_updated.connect(self.log_status)
        self.search_worker.search_complete.connect(self.on_search_complete)
        self.search_worker.error_occurred.connect(self.on_search_error)
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.start()

    # --- Search Result Handling Slots ---
    def on_search_complete(self, report_path):
        """Handle successful search completion."""
        self.log_status(f"Search finished successfully!")
        # Update MainWindow state
        self.main_window.current_report_path = report_path
        self.main_window.current_results_dir = os.path.dirname(report_path)
        self.main_window.report_path_label.setText(report_path)
        self.main_window.open_report_button.setEnabled(True)
        self.main_window.open_folder_button.setEnabled(True)
        self.main_window.share_email_button.setEnabled(True)

    def on_search_error(self, error_message):
        """Show error message if search fails."""
        self.log_status(f"Search Error: {error_message}")
        QMessageBox.critical(self.main_window, "Search Error", error_message)
        self.main_window.report_path_label.setText("Search failed.")

    def on_search_finished(self):
        """Called when the SearchWorker thread finishes (success or error)."""
        self.main_window.run_button.setEnabled(True)
        self.search_worker = None # Clear worker reference
        if not self.main_window.current_report_path: # Check if search actually completed successfully
            self.main_window.open_report_button.setEnabled(False)
            self.main_window.open_folder_button.setEnabled(False)
            self.main_window.share_email_button.setEnabled(False)

        self.main_window.progress_bar.setVisible(False)
        self.main_window.progress_bar.setRange(0, 100) # Reset range
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)

    # --- Cancellation ---
    def cancel_current_operation(self):
        """Requests cancellation of the currently running SearchWorker."""
        if self.search_worker and self.search_worker.isRunning():
            self.log_status("Requesting search cancellation...")
            self.search_worker.request_cancellation()
            self.main_window.cancel_button.setEnabled(False)
            self.log_status("Cancellation requested. Waiting for search worker to stop...")
        elif self.topic_extractor_worker and self.topic_extractor_worker.isRunning():
            self.log_status("Topic extraction is running. Cancellation not implemented for this step.")
        elif self.query_enhancer_worker and self.query_enhancer_worker.isRunning():
             self.log_status("Query enhancement preview is running. Cancellation not implemented for this step.")
        else:
            self.log_status("No cancellable operation running.")

    # --- Cache Management ---
    def clear_cache(self):
        """Clears the cache database."""
        # Get cache path from config
        cache_db_path = self.config_data.get('cache', {}).get('db_path', DEFAULT_CONFIG['cache']['db_path'])

        reply = QMessageBox.question(self.main_window, 'Confirm Clear Cache',
                                     f"Are you sure you want to delete the cache file?\n({cache_db_path})",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.log_status(f"Attempting to clear cache: {cache_db_path}")
            try:
                # Use a temporary CacheManager instance just for clearing
                temp_cache_manager = CacheManager(cache_db_path)
                temp_cache_manager.clear_all_cache() # This deletes and recreates
                temp_cache_manager.close() # Close the connection
                self.log_status("Cache cleared successfully.")
                QMessageBox.information(self.main_window, "Cache Cleared", "The cache database has been cleared.")
            except Exception as e:
                error_msg = f"Failed to clear cache: {e}"
                self.log_status(f"[ERROR] {error_msg}")
                logging.exception("Error during cache clearing") # Log full traceback
                QMessageBox.critical(self.main_window, "Cache Error", error_msg)

    # --- Cleanup ---
    def shutdown_workers(self):
        """Stop all running worker threads gracefully."""
        self.log_status("Shutting down controller and workers...")
        # Stop SearchWorker if running
        if self.search_worker and self.search_worker.isRunning():
             self.log_status("Attempting to cancel search on close...")
             self.search_worker.request_cancellation()
             self.search_worker.wait(5000) # Wait up to 5 seconds
             if self.search_worker.isRunning():
                 self.log_status("Search worker did not stop gracefully, terminating...")
                 self.search_worker.terminate()
        # Stop TopicExtractorWorker if running
        if self.topic_extractor_worker and self.topic_extractor_worker.isRunning():
             self.log_status("Terminating topic extraction on close...")
             self.topic_extractor_worker.terminate()
             self.topic_extractor_worker.wait()
        # Stop QueryEnhancerWorker if running
        if self.query_enhancer_worker and self.query_enhancer_worker.isRunning():
             self.log_status("Terminating query enhancement on close...")
             self.query_enhancer_worker.terminate()
             self.query_enhancer_worker.wait()
        # Stop generative fetchers
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
             self.log_status("Terminating Gemini fetcher on close...")
             self.gemini_fetcher.terminate()
             self.gemini_fetcher.wait()
        if self.openrouter_fetcher and self.openrouter_fetcher.isRunning():
             self.log_status("Terminating OpenRouter fetcher on close...")
             self.openrouter_fetcher.terminate()
             self.openrouter_fetcher.wait()
        self.log_status("Worker shutdown complete.")
