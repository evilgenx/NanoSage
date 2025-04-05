#!/usr/bin/env python3
# gui/main_window.py

import sys
import os
import subprocess # To open files/folders
import webbrowser # Added for mailto link
import urllib.parse # Added for URL encoding
import logging # Added logging
from PyQt6.QtWidgets import (
    QMainWindow, QTextEdit, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QFileDialog, QMessageBox # Removed unused layout/widget imports
)
# from PyQt6.QtCore import Qt # Removed unused Qt
# from PyQt6.QtGui import QIcon # Optional: for window icon

# Import the new UI setup function
from .ui_setup import setup_main_window_ui

# Import worker threads using relative import
try:
    from .workers import (
        SearchWorker, GeminiFetcher, OpenRouterFetcher, TopicExtractorWorker,
        QueryEnhancerWorker # Added QueryEnhancerWorker
        # Removed GeminiEmbeddingFetcher, OpenRouterEmbeddingFetcher as they are no longer used
    )
except ImportError as e:
    print(f"Error importing workers in main_window.py: {e}")
    # Handle appropriately, maybe raise or show error in GUI if possible later
    sys.exit(1)

# Import config loading utility
from config_utils import load_config, save_config, DEFAULT_CONFIG # Added DEFAULT_CONFIG
from cache_manager import CacheManager # Added CacheManager import

# Import the selector widget (though it's created in ui_setup, we need its type potentially)
# from .ui_components.searxng_selector import SearxngEngineSelector # Not strictly needed here if only interacting via ui_setup attributes

# --- Main Window ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NanoSage-EG GUI 🧙") # <<< Changed here
        # self.setWindowIcon(QIcon("path/to/icon.png")) # Optional

        # Load initial config
        self.config_path = "config.yaml" # Define config path
        self.config_data = load_config(self.config_path)

        self.search_worker = None
        self.topic_extractor_worker = None
        self.query_enhancer_worker = None # Added worker instance variable
        # Generative model fetchers
        self.gemini_fetcher = None
        self.openrouter_fetcher = None
        # Embedding model fetchers removed as options are gone
        # self.gemini_embedding_fetcher = None
        # self.openrouter_embedding_fetcher = None
        self.current_report_path = None
        self.current_results_dir = None
        self.cache_manager_instance = None # Added instance variable for cache manager

        # Call the UI setup function from the separate module
        setup_main_window_ui(self)

        # --- Set initial UI states based on loaded config ---
        # Search Provider
        default_provider = self.config_data.get('search', {}).get('provider', 'duckduckgo')
        provider_index = 0 if default_provider == 'duckduckgo' else 1
        self.search_provider_combo.setCurrentIndex(provider_index)

        # SearXNG Fields (values loaded from config)
        searxng_config = self.config_data.get('search', {}).get('searxng', {})
        self.searxng_base_url_input.setText(searxng_config.get('base_url', ''))
        self.searxng_time_range_input.setText(searxng_config.get('time_range', '') or '') # Ensure string for setText
        self.searxng_categories_input.setText(searxng_config.get('categories', '') or '') # Ensure string for setText
        # self.searxng_engines_input.setText(searxng_config.get('engines', '')) # Removed old input

        # Set initial state of the SearxngEngineSelector
        initial_engines = searxng_config.get('engines', [])
        # Ensure it's a list, handle potential None or non-list values from config
        if not isinstance(initial_engines, list):
            initial_engines = [] # Default to empty list if config value is invalid
        self.searxng_engine_selector.setSelectedEngines(initial_engines)

        # Initial visibility based on selected provider (call handler)
        self.handle_search_provider_change(self.search_provider_combo.currentText())

        # Initial embedding device/model state (call handler)
        # Assuming default device is 'cpu' if not specified in config (or handle config loading)
        default_embedding_device = self.config_data.get('embeddings', {}).get('device', 'cpu') # Example config structure
        self.device_combo.setCurrentText(default_embedding_device) # Set combo
        self.handle_device_change(default_embedding_device) # Trigger model list update

        # Initial RAG model state (call handler)
        default_rag_model = self.config_data.get('rag', {}).get('model', 'None') # Example config structure
        self.rag_model_combo.setCurrentText(default_rag_model)
        self.handle_rag_model_change(default_rag_model) # Trigger visibility updates

        # Set other config-dependent initial states if any (e.g., checkbox, spinboxes)
        search_config_init = self.config_data.get('search', {}) # Get search config once
        self.web_search_checkbox.setChecked(search_config_init.get('web_search_enabled', True)) # Example
        self.iterative_search_checkbox.setChecked(search_config_init.get('enable_iterative_search', False)) # Initialize new checkbox
        self.max_depth_spinbox.setValue(search_config_init.get('max_depth', 1))
        self.top_k_spinbox.setValue(search_config_init.get('top_k', 3))
        self.corpus_dir_label.setText(self.config_data.get('corpus', {}).get('path', '')) # Example
        self.personality_input.setText(self.config_data.get('rag', {}).get('personality', ''))

        # Cache settings initialization
        cache_config_init = self.config_data.get('cache', {})
        self.cache_enabled_checkbox.setChecked(cache_config_init.get('enabled', False)) # Default to False if not in config

        # Populate Output Format dropdown
        self.output_format_combo.clear()
        output_formats_config = self.config_data.get('llm', {}).get('output_formats', {})
        if output_formats_config:
            self.output_format_combo.addItems(output_formats_config.keys())
            # Optionally set a default, e.g., 'Report' if it exists
            if "Report" in output_formats_config:
                self.output_format_combo.setCurrentText("Report")
        else:
            self.log_status("[Warning] No 'output_formats' found in config.yaml under 'llm'. Dropdown will be empty.")
            self.output_format_combo.setEnabled(False)


        # Connect signals after UI is fully set up and initialized
        self._connect_signals()

    # _init_ui method is now removed

    def _connect_signals(self):
        """Connect UI signals to slots."""
        self.run_button.clicked.connect(self.start_search)
        self.corpus_dir_button.clicked.connect(self.select_corpus_directory)
        # Connect device change signal
        self.device_combo.currentTextChanged.connect(self.handle_device_change)
        # Connect RAG model change signal
        self.rag_model_combo.currentTextChanged.connect(self.handle_rag_model_change)
        # Connect generative model fetch buttons
        self.gemini_fetch_button.clicked.connect(self.fetch_gemini_models)
        self.openrouter_fetch_button.clicked.connect(self.fetch_openrouter_models)
        # Removed connections for embedding model fetch buttons
        # self.gemini_embedding_fetch_button.clicked.connect(self.fetch_gemini_embedding_models)
        # self.openrouter_embedding_fetch_button.clicked.connect(self.fetch_openrouter_embedding_models)
        # Connect result buttons
        self.open_report_button.clicked.connect(self.open_report)
        self.open_folder_button.clicked.connect(self.open_results_folder)
        self.share_email_button.clicked.connect(self.share_report_email) # Connect new button
        # Connect search provider change signal to show/hide SearXNG options
        self.search_provider_combo.currentTextChanged.connect(self.handle_search_provider_change)
        # Connect the engine selector's signal
        self.searxng_engine_selector.selectionChanged.connect(self._handle_searxng_engine_selection_change)
        # Connect cancel button
        self.cancel_button.clicked.connect(self.cancel_search)
        # Connect cache controls
        self.cache_enabled_checkbox.stateChanged.connect(self._handle_cache_enabled_change)
        self.clear_cache_button.clicked.connect(self.clear_cache)


    # --- Slot Methods ---

    # --- Query Enhancement Slots ---
    def on_enhanced_query_ready(self, enhanced_query_preview):
        """Handles the signal when the enhanced query preview is ready."""
        self.log_status(f"Enhanced Query Preview: {enhanced_query_preview}")
        # IMPORTANT: Proceed with the search using the ORIGINAL query text
        # that was present in the input box when the enhancement started.
        original_query = self.query_input.toPlainText().strip() # Get original text again

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
        QMessageBox.critical(self, "Query Enhancement Error", error_message)
        # Reset UI state
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)
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
        self.query_input.setPlainText(topics_string) # Populate input field
        self.extract_topics_checkbox.setChecked(False) # Uncheck the box
        # Re-enable run button, hide progress/cancel as extraction is done
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)
        self.topic_extractor_worker = None # Clear worker reference
        QMessageBox.information(self, "Topics Extracted", "Extracted topics have been placed in the query box.\nReview or edit them, then click 'Run Search' again to search using these topics.")

    def on_topic_extraction_error(self, error_message):
        """Handles errors during topic extraction."""
        self.log_status(f"Topic Extraction Error: {error_message}")
        QMessageBox.critical(self, "Topic Extraction Error", error_message)
        # Reset UI state
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)
        self.topic_extractor_worker = None # Clear worker reference
        # Keep the checkbox checked so user knows it failed
        # self.extract_topics_checkbox.setChecked(False)

    def on_topic_extraction_finished(self):
        """Called when the topic extraction thread finishes (success or error)."""
        # This might not be strictly necessary if handled in success/error slots,
        # but good for ensuring UI state is reset if worker finishes unexpectedly.
        if self.topic_extractor_worker: # Check if it finished normally or was cleared already
            # Reset UI only if enhancer isn't also running (unlikely scenario)
            if not (self.query_enhancer_worker and self.query_enhancer_worker.isRunning()):
                self.run_button.setEnabled(True)
                self.progress_bar.setVisible(False)
                self.cancel_button.setVisible(False)
                self.cancel_button.setEnabled(False)
            self.topic_extractor_worker = None

    def _handle_searxng_engine_selection_change(self, selected_engines):
        """Update the config when SearXNG engine selection changes."""
        if 'search' not in self.config_data: self.config_data['search'] = {}
        if 'searxng' not in self.config_data['search']: self.config_data['search']['searxng'] = {}

        self.config_data['search']['searxng']['engines'] = selected_engines

        # Save the updated config
        if save_config(self.config_path, self.config_data):
            self.log_status(f"SearXNG engine selection saved to {self.config_path}")
        else:
            self.log_status(f"[ERROR] Failed to save configuration to {self.config_path}")
            QMessageBox.warning(self, "Config Error", f"Could not save engine selection to {self.config_path}")

    def handle_search_provider_change(self, provider_text):
        """Show/hide SearXNG specific settings."""
        is_searxng = (provider_text == "SearXNG")
        self.searxng_base_url_label.setVisible(is_searxng)
        self.searxng_base_url_input.setVisible(is_searxng)
        self.searxng_time_range_label.setVisible(is_searxng)
        self.searxng_time_range_input.setVisible(is_searxng)
        self.searxng_categories_label.setVisible(is_searxng)
        self.searxng_categories_input.setVisible(is_searxng)
        # self.searxng_engines_label.setVisible(is_searxng) # Removed old label
        # self.searxng_engines_input.setVisible(is_searxng) # Removed old input
        self.searxng_engine_group.setVisible(is_searxng) # Show/hide the whole group

    def handle_device_change(self, device_name):
        """Handles changes in the Embedding Device selection."""
        self.log_status(f"Embedding device changed to: {device_name}")
        self.embedding_model_combo.clear() # Clear previous model options
        self.embedding_model_combo.setEnabled(True) # Enable by default

        # Removed logic for showing/hiding/resetting Gemini/OpenRouter embedding fetch buttons
        # self.gemini_embedding_fetch_button.setVisible(device_name == "Gemini")
        # self.openrouter_embedding_fetch_button.setVisible(device_name == "OpenRouter")
        # self.gemini_embedding_fetch_button.setEnabled(True)
        # self.openrouter_embedding_fetch_button.setEnabled(True)

        if device_name == "cpu" or device_name == "cuda" or device_name == "rocm": # Added rocm
            self.embedding_model_combo.addItems(["colpali", "all-minilm"])
            self.embedding_model_label.setText("Embedding Model:")
        # Removed elif blocks for "Gemini" and "OpenRouter" as they are no longer options
        # elif device_name == "Gemini":
        #     ...
        # elif device_name == "OpenRouter":
        #     ...
        else:
            # This case should ideally not be reached if only cpu/cuda are options
            self.log_status(f"Warning: Unexpected embedding device selected: {device_name}")
            self.embedding_model_combo.clear()
            self.embedding_model_combo.setEnabled(False)
            self.embedding_model_label.setText("Embedding Model:")


    def select_corpus_directory(self):
        """Open dialog to select local corpus directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Corpus Directory")
        if directory:
            self.corpus_dir_label.setText(directory)

    def handle_rag_model_change(self, model_name):
        """Show/hide model-specific and Personality options based on RAG model selection."""
        is_gemini_rag = (model_name == "gemini")
        is_openrouter_rag = (model_name == "openrouter")
        is_rag_enabled = (model_name != "None")

        # Gemini RAG visibility
        self.gemini_fetch_button.setVisible(is_gemini_rag)
        self.gemini_model_label.setVisible(is_gemini_rag)
        self.gemini_model_combo.setVisible(is_gemini_rag)
        if not is_gemini_rag:
            self.gemini_model_combo.clear()
            # Re-enable fetch button if switching away and no models loaded
            if self.gemini_model_combo.count() == 0:
                 self.gemini_fetch_button.setEnabled(True)

        # OpenRouter RAG visibility
        self.openrouter_fetch_button.setVisible(is_openrouter_rag)
        self.openrouter_model_label.setVisible(is_openrouter_rag)
        self.openrouter_model_combo.setVisible(is_openrouter_rag)
        if not is_openrouter_rag:
            self.openrouter_model_combo.clear()
            # Re-enable fetch button if switching away and no models loaded
            if self.openrouter_model_combo.count() == 0:
                self.openrouter_fetch_button.setEnabled(True)

        # Personality visibility (hide if no RAG model selected)
        self.personality_label.setVisible(is_rag_enabled)
        self.personality_input.setVisible(is_rag_enabled)


    # --- Generative Model Fetching Slots ---

    def fetch_gemini_models(self):
        """Start the GeminiFetcher thread for *generative* models."""
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
            self.log_status("Already fetching Gemini generative models...")
            return

        self.log_status("Attempting to fetch Gemini generative models (requires GEMINI_API_KEY)...")
        self.gemini_fetch_button.setEnabled(False)
        self.gemini_fetcher = GeminiFetcher(self)
        self.gemini_fetcher.status_update.connect(self.log_status)
        self.gemini_fetcher.models_fetched.connect(self.on_gemini_models_fetched)
        self.gemini_fetcher.fetch_error.connect(self.on_gemini_fetch_error)
        # Re-enable button only if combo is still empty after finish
        self.gemini_fetcher.finished.connect(lambda: self.gemini_fetch_button.setEnabled(self.gemini_model_combo.count() == 0))
        self.gemini_fetcher.start()

    def on_gemini_models_fetched(self, models):
        """Populate the Gemini *generative* model combo box."""
        self.log_status(f"Successfully fetched {len(models)} Gemini generative models.")
        self.gemini_model_combo.clear()
        self.gemini_model_combo.addItems(models)
        self.gemini_model_combo.setEnabled(True)
        self.gemini_fetch_button.setEnabled(False) # Disable after successful fetch

    def on_gemini_fetch_error(self, error_message):
        """Show error message if Gemini *generative* fetch fails."""
        self.log_status(f"Gemini Generative Fetch Error: {error_message}")
        self.gemini_model_combo.clear()
        self.gemini_model_combo.setEnabled(False)
        self.gemini_fetch_button.setEnabled(True) # Re-enable on error

    def fetch_openrouter_models(self):
        """Start the OpenRouterFetcher thread for *generative* models."""
        if self.openrouter_fetcher and self.openrouter_fetcher.isRunning():
            self.log_status("Already fetching OpenRouter generative models...")
            return

        self.log_status("Attempting to fetch free OpenRouter generative models...")
        self.openrouter_fetch_button.setEnabled(False)
        self.openrouter_fetcher = OpenRouterFetcher(self)
        self.openrouter_fetcher.status_update.connect(self.log_status)
        self.openrouter_fetcher.models_fetched.connect(self.on_openrouter_models_fetched)
        self.openrouter_fetcher.fetch_error.connect(self.on_openrouter_fetch_error)
        # Re-enable button only if combo is still empty after finish
        self.openrouter_fetcher.finished.connect(lambda: self.openrouter_fetch_button.setEnabled(self.openrouter_model_combo.count() == 0))
        self.openrouter_fetcher.start()

    def on_openrouter_models_fetched(self, models):
        """Populate the OpenRouter *generative* model combo box."""
        self.log_status(f"Successfully fetched {len(models)} free OpenRouter generative models.")
        self.openrouter_model_combo.clear()
        self.openrouter_model_combo.addItems(models)
        self.openrouter_model_combo.setEnabled(True)
        self.openrouter_fetch_button.setEnabled(False) # Disable after successful fetch

    def on_openrouter_fetch_error(self, error_message):
        """Show error message if OpenRouter *generative* fetch fails."""
        self.log_status(f"OpenRouter Generative Fetch Error: {error_message}")
        self.openrouter_model_combo.clear()
        self.openrouter_model_combo.setEnabled(False)
        self.openrouter_fetch_button.setEnabled(True) # Re-enable on error
    # Removed slots related to fetching Gemini/OpenRouter embedding models
    # --- Search Execution ---

    def start_search(self):
        """
        Validate inputs and start the appropriate worker thread:
        1. Topic Extraction (if checkbox checked)
        2. Query Enhancement Preview (if checkbox unchecked)
        3. Main Search (triggered after step 1 or 2 completes)
        """
        # --- Input Validation ---
        # Check if any worker is running
        if (self.search_worker and self.search_worker.isRunning()) or \
           (self.topic_extractor_worker and self.topic_extractor_worker.isRunning()) or \
           (self.query_enhancer_worker and self.query_enhancer_worker.isRunning()): # Added check for enhancer
            self.log_status("An operation (search, topic extraction, or enhancement preview) is already in progress.")
            return

        query_or_text = self.query_input.toPlainText().strip()
        if not query_or_text:
            QMessageBox.warning(self, "Input Error", "Please enter text in the query box.")
            return

        # --- Shared Validation for LLM Tasks (Extraction & Enhancement) ---
        # Both topic extraction and enhancement require a RAG model selection
        rag_model_type = self.rag_model_combo.currentText()
        selected_generative_gemini = None
        selected_generative_openrouter = None

        if rag_model_type == "None":
            QMessageBox.warning(self, "Input Error", "Topic extraction or query enhancement preview requires a RAG model (Gemini, OpenRouter, or Gemma) to be selected.")
            return
        elif rag_model_type == "gemini":
            if self.gemini_model_combo.count() == 0 or not self.gemini_model_combo.currentText():
                QMessageBox.warning(self, "Input Error", f"{'Topic extraction' if self.extract_topics_checkbox.isChecked() else 'Query enhancement'} needs a Gemini generative model. Please fetch and select one.")
                return
            selected_generative_gemini = self.gemini_model_combo.currentText()
        elif rag_model_type == "openrouter":
            if self.openrouter_model_combo.count() == 0 or not self.openrouter_model_combo.currentText():
                QMessageBox.warning(self, "Input Error", f"{'Topic extraction' if self.extract_topics_checkbox.isChecked() else 'Query enhancement'} needs an OpenRouter generative model. Please fetch and select one.")
                return
            selected_generative_openrouter = self.openrouter_model_combo.currentText()

        # Prepare LLM config (used by both topic extractor and query enhancer)
        llm_config = {
            "provider": rag_model_type,
            "model_id": selected_generative_gemini or selected_generative_openrouter, # Pass the specific model if applicable
            "api_key": self.config_data.get('api_keys', {}).get(f'{rag_model_type}_api_key'), # Get relevant API key
            "personality": self.personality_input.text() or None
        }

        # --- Action based on Checkbox State ---
        if self.extract_topics_checkbox.isChecked():
            # --- Start Topic Extractor Worker ---
            self.log_status("Topic extraction requested...")
            self.log_status("Starting topic extraction worker...")
            self.run_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0) # Indeterminate
            # Note: Cancellation for topic extraction isn't implemented here, assumed quick.

            self.topic_extractor_worker = TopicExtractorWorker(query_or_text, llm_config, self)
            self.topic_extractor_worker.status_update.connect(self.log_status)
            self.topic_extractor_worker.topics_extracted.connect(self.on_topics_extracted)
            self.topic_extractor_worker.error_occurred.connect(self.on_topic_extraction_error)
            self.topic_extractor_worker.finished.connect(self.on_topic_extraction_finished) # General cleanup
            self.topic_extractor_worker.start()
            # Stop here, wait for extraction to finish before user clicks Run again

        else:
            # --- Start Query Enhancement Preview Worker ---
            self.log_status("Starting query enhancement preview...")
            self.run_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0) # Indeterminate
            # Note: Cancellation for enhancement isn't implemented here, assumed quick.

            self.query_enhancer_worker = QueryEnhancerWorker(query_or_text, llm_config, self)
            self.query_enhancer_worker.status_update.connect(self.log_status)
            self.query_enhancer_worker.enhanced_query_ready.connect(self.on_enhanced_query_ready) # Connect to new slot
            self.query_enhancer_worker.enhancement_error.connect(self.on_enhancement_error) # Connect to new error slot
            self.query_enhancer_worker.finished.connect(self.on_enhancement_finished) # General cleanup
            self.query_enhancer_worker.start()
            # Stop here, wait for enhancement preview to finish, which then triggers the search worker

    def _handle_cache_enabled_change(self, state):
        """Update config when cache enabled checkbox changes."""
        enabled = (state == 2) # 2 means checked
        if 'cache' not in self.config_data: self.config_data['cache'] = {}
        self.config_data['cache']['enabled'] = enabled
        if save_config(self.config_path, self.config_data):
            self.log_status(f"Cache setting saved to {self.config_path} (Enabled: {enabled})")
        else:
            self.log_status(f"[ERROR] Failed to save cache setting to {self.config_path}")
            QMessageBox.warning(self, "Config Error", f"Could not save cache setting to {self.config_path}")

    def clear_cache(self):
        """Clears the cache database."""
        # Get cache path from config
        cache_db_path = self.config_data.get('cache', {}).get('db_path', DEFAULT_CONFIG['cache']['db_path'])

        reply = QMessageBox.question(self, 'Confirm Clear Cache',
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
                QMessageBox.information(self, "Cache Cleared", "The cache database has been cleared.")
            except Exception as e:
                error_msg = f"Failed to clear cache: {e}"
                self.log_status(f"[ERROR] {error_msg}")
                logging.exception("Error during cache clearing") # Log full traceback
                QMessageBox.critical(self, "Cache Error", error_msg)


    def _start_main_search_worker(self, query_to_use):
        """
        Internal method to prepare parameters and start the main SearchWorker.
        This is called AFTER topic extraction (if used) or query enhancement preview.
        """
        self.log_status("Preparing to start main search worker...")

        # --- Standard Search Input Validation (Run again before starting search) ---
        # Embedding settings validation
        embedding_device = self.device_combo.currentText()
        embedding_model_name = self.embedding_model_combo.currentText()
        if not embedding_model_name:
             QMessageBox.warning(self, "Input Error", f"Please select an Embedding Model for the '{embedding_device}' device.")
             self.run_button.setEnabled(True) # Re-enable button on validation fail
             self.progress_bar.setVisible(False)
             return
        # RAG settings validation (only if RAG model is selected)
        # Re-fetch RAG settings as they might have changed while preview was running
        rag_model_type = self.rag_model_combo.currentText()
        selected_generative_gemini = None
        selected_generative_openrouter = None
        if rag_model_type == "gemini":
            if self.gemini_model_combo.count() == 0 or not self.gemini_model_combo.currentText():
                 QMessageBox.warning(self, "Input Error", "RAG Model is Gemini, but no generative model selected. Please fetch and select one.")
                 self.run_button.setEnabled(True) # Re-enable button on validation fail
                 self.progress_bar.setVisible(False)
                 return
            selected_generative_gemini = self.gemini_model_combo.currentText()
        elif rag_model_type == "openrouter":
            if self.openrouter_model_combo.count() == 0 or not self.openrouter_model_combo.currentText():
                 QMessageBox.warning(self, "Input Error", "RAG Model is OpenRouter, but no generative model selected. Please fetch and select one.")
                 self.run_button.setEnabled(True) # Re-enable button on validation fail
                 self.progress_bar.setVisible(False)
                 return
            selected_generative_openrouter = self.openrouter_model_combo.currentText()

        # --- Save Current Config & Prepare Search Parameters ---
        # Reload config data to ensure we have the latest state (e.g., cache enabled)
        self.config_data = load_config(self.config_path)

        # Update config_data with current UI settings before passing to worker
        # General
        if 'general' not in self.config_data: self.config_data['general'] = {}
        self.config_data['general']['web_search'] = self.web_search_checkbox.isChecked()
        self.config_data['general']['max_depth'] = self.max_depth_spinbox.value()
        # Retrieval
        if 'retrieval' not in self.config_data: self.config_data['retrieval'] = {}
        self.config_data['retrieval']['top_k'] = self.top_k_spinbox.value()
        self.config_data['retrieval']['embedding_model'] = embedding_model_name # Already validated
        # Corpus
        if 'corpus' not in self.config_data: self.config_data['corpus'] = {}
        self.config_data['corpus']['path'] = self.corpus_dir_label.text() or None
        # RAG
        if 'llm' not in self.config_data: self.config_data['llm'] = {}
        self.config_data['llm']['rag_model'] = rag_model_type if rag_model_type != "None" else None
        self.config_data['llm']['personality'] = self.personality_input.text() or None
        self.config_data['llm']['gemini_model_id'] = selected_generative_gemini
        self.config_data['llm']['openrouter_model_id'] = selected_generative_openrouter
        # Cache (already saved by signal, but ensure it's in config_data)
        if 'cache' not in self.config_data: self.config_data['cache'] = {}
        self.config_data['cache']['enabled'] = self.cache_enabled_checkbox.isChecked()

        # Search Provider specific settings
        selected_provider_text = self.search_provider_combo.currentText()
        search_provider_key = 'duckduckgo' if selected_provider_text == "DuckDuckGo" else 'searxng'
        search_config = self.config_data.setdefault('search', {})
        search_config['provider'] = search_provider_key
        search_config['enable_iterative_search'] = self.iterative_search_checkbox.isChecked() # Save iterative search state

        searxng_config = search_config.setdefault('searxng', {})
        ddg_config = search_config.setdefault('duckduckgo', {})

        search_limit = 5 # Default
        searxng_url = None
        searxng_time_range = None
        searxng_categories = None
        searxng_engines = None

        if search_provider_key == 'searxng':
            searxng_url = self.searxng_base_url_input.text().strip() or None
            searxng_time_range = self.searxng_time_range_input.text().strip() or None
            searxng_categories = self.searxng_categories_input.text().strip() or None
            searxng_engines = self.config_data.get('search', {}).get('searxng', {}).get('engines', []) # Read from config (updated by signal)
            if not isinstance(searxng_engines, list): searxng_engines = []

            searxng_config['base_url'] = searxng_url
            searxng_config['time_range'] = searxng_time_range
            searxng_config['categories'] = searxng_categories
            # Engines are already saved by the signal handler

            search_limit = searxng_config.get('max_results', 5)
        else: # duckduckgo
            search_limit = ddg_config.get('max_results', 5)

        # Save the potentially modified config_data before starting the worker
        if save_config(self.config_path, self.config_data):
            self.log_status(f"Current settings saved to {self.config_path}")
        else:
            self.log_status(f"[ERROR] Failed to save configuration before starting search: {self.config_path}")
            QMessageBox.warning(self, "Config Error", f"Could not save settings to {self.config_path}. Search may use outdated settings.")
            # Decide whether to proceed or stop if saving fails
            # return # Uncomment to stop if saving fails

        # --- Prepare All Parameters for Search Worker ---
        search_params = {
            "query": query_to_use, # Use the query passed to this method
            "corpus_dir": self.corpus_dir_label.text() or None,
            "web_search": self.web_search_checkbox.isChecked(),
            "enable_iterative_search": self.iterative_search_checkbox.isChecked(),
            "max_depth": self.max_depth_spinbox.value(),
            "top_k": self.top_k_spinbox.value(),
            "device": embedding_device,
            "embedding_model_name": embedding_model_name,
            "rag_model": rag_model_type if rag_model_type != "None" else None,
            "personality": self.personality_input.text() or None,
            "selected_gemini_model": selected_generative_gemini,
            "selected_openrouter_model": selected_generative_openrouter,
            # Search settings
            "search_provider": search_provider_key,
            "search_limit": search_limit,
            "searxng_url": searxng_url,
            "searxng_time_range": searxng_time_range,
            "searxng_categories": searxng_categories,
            "searxng_engines": searxng_engines,
            "config_path": self.config_path,
            # Add selected output format
            "output_format": self.output_format_combo.currentText()
        }

        # --- Start Search Worker ---
        self.log_status("Starting main search worker...")
        # Ensure Run button is disabled (might have been re-enabled by error handlers)
        self.run_button.setEnabled(False)
        self.open_report_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.share_email_button.setEnabled(False)
        self.report_path_label.setText("Running Search...")
        self.current_report_path = None
        self.current_results_dir = None

        # Ensure progress bar and cancel button are visible/enabled for the main search
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)

        # Ensure search provider visibility is correct before starting
        self.handle_search_provider_change(self.search_provider_combo.currentText())

        self.search_worker = SearchWorker(search_params, self)
        self.search_worker.progress_updated.connect(self.log_status)
        self.search_worker.search_complete.connect(self.on_search_complete)
        self.search_worker.error_occurred.connect(self.on_search_error)
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.start()


    # --- Utility & Result Handling ---

    # Optional: Method to save config changes (e.g., when provider changes)
    # def handle_search_provider_change(self, provider_text):
    #     """Update config data when search provider changes."""
    #     provider_key = 'duckduckgo' if provider_text == "DuckDuckGo" else 'searxng'
    #     if 'search' not in self.config_data:
    #         self.config_data['search'] = {}
    #     self.config_data['search']['provider'] = provider_key
    #     self.save_current_config()

    # def save_current_config(self):
    #     """Save the current state of self.config_data to the file."""
    #     if save_config(self.config_path, self.config_data):
    #         self.log_status(f"Configuration saved to {self.config_path}")
    #     else:
    #         self.log_status(f"[ERROR] Failed to save configuration to {self.config_path}")

    def log_status(self, message):
        """Append a message to the status log."""
        self.status_log.append(message)
        self.status_log.verticalScrollBar().setValue(self.status_log.verticalScrollBar().maximum()) # Auto-scroll

    def on_search_complete(self, report_path):
        """Handle successful search completion."""
        self.log_status(f"Search finished successfully!")
        self.current_report_path = report_path
        self.current_results_dir = os.path.dirname(report_path)
        self.report_path_label.setText(report_path)
        self.open_report_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.share_email_button.setEnabled(True) # Enable email button on completion

    def on_search_error(self, error_message):
        """Show error message if search fails."""
        self.log_status(f"Search Error: {error_message}")
        QMessageBox.critical(self, "Search Error", error_message)
        self.report_path_label.setText("Search failed.")


    def on_search_finished(self):
        """Called when the SearchWorker thread finishes (success or error)."""
        # This is only for the SearchWorker, not the TopicExtractorWorker
        self.run_button.setEnabled(True)
        self.search_worker = None # Clear worker reference
        if not self.current_report_path: # Check if search actually completed successfully
            self.open_report_button.setEnabled(False)
            self.open_folder_button.setEnabled(False)
            self.share_email_button.setEnabled(False)

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100) # Reset range
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)

    def cancel_search(self):
        """Requests cancellation of the currently running SearchWorker."""
        # Note: Cancellation for TopicExtractorWorker is not implemented
        if self.search_worker and self.search_worker.isRunning():
            self.log_status("Requesting search cancellation...")
            self.search_worker.request_cancellation()
            self.cancel_button.setEnabled(False) # Correctly indented now
            self.log_status("Cancellation requested. Waiting for search worker to stop...") # Correctly indented now
        # Add checks for other workers (though they are short-lived)
        elif self.topic_extractor_worker and self.topic_extractor_worker.isRunning():
            self.log_status("Topic extraction is running. Cancellation not implemented for this step.")
            # self.topic_extractor_worker.terminate() # Avoid terminate if possible
        elif self.query_enhancer_worker and self.query_enhancer_worker.isRunning():
             self.log_status("Query enhancement preview is running. Cancellation not implemented for this step.")
             # self.query_enhancer_worker.terminate() # Avoid terminate if possible
        else:
            self.log_status("No cancellable operation running.")

    def open_report(self):
        """Open the generated report file using the default system viewer."""
        if self.current_report_path and os.path.exists(self.current_report_path):
            try:
                if sys.platform == "win32":
                    os.startfile(self.current_report_path)
                elif sys.platform == "darwin": # macOS
                    subprocess.run(["open", self.current_report_path], check=True)
                else: # Linux and other Unix-like
                    subprocess.run(["xdg-open", self.current_report_path], check=True)
                self.log_status(f"Attempting to open report: {self.current_report_path}")
            except Exception as e:
                self.log_status(f"Error opening report: {e}")
                QMessageBox.warning(self, "Open Error", f"Could not open the report file:\n{e}")
        else:
            QMessageBox.warning(self, "File Not Found", "The report file does not exist or path is not set.")

    def open_results_folder(self):
        """Open the folder containing the results."""
        if self.current_results_dir and os.path.exists(self.current_results_dir):
            try:
                if sys.platform == "win32":
                     # Use explorer for Windows, safer for paths with spaces
                     subprocess.run(['explorer', self.current_results_dir])
                elif sys.platform == "darwin": # macOS
                    subprocess.run(["open", self.current_results_dir], check=True)
                else: # Linux and other Unix-like
                    subprocess.run(["xdg-open", self.current_results_dir], check=True)
                self.log_status(f"Attempting to open results folder: {self.current_results_dir}")
            except Exception as e:
                self.log_status(f"Error opening results folder: {e}")
                QMessageBox.warning(self, "Open Error", f"Could not open the results folder:\n{e}")
        else:
             QMessageBox.warning(self, "Folder Not Found", "The results directory does not exist or path is not set.")

    def share_report_email(self):
        """Open the default email client with a pre-filled message."""
        if self.current_report_path and os.path.exists(self.current_report_path):
            try:
                subject = "NanoSage Research Report"
                body = (
                    f"Please find the research report attached.\n\n"
                    f"You can find the file at:\n{self.current_report_path}\n\n"
                    f"(Please attach the file manually before sending)"
                )
                # URL encode subject and body
                encoded_subject = urllib.parse.quote(subject)
                encoded_body = urllib.parse.quote(body)

                mailto_url = f"mailto:?subject={encoded_subject}&body={encoded_body}"

                webbrowser.open(mailto_url)
                self.log_status(f"Attempting to open email client for report: {self.current_report_path}")
            except Exception as e:
                self.log_status(f"Error opening email client: {e}")
                QMessageBox.warning(self, "Email Error", f"Could not open the email client:\n{e}")
        else:
            QMessageBox.warning(self, "File Not Found", "The report file does not exist or path is not set. Cannot share.")


    def closeEvent(self, event):
        """Ensure threads are stopped on close."""
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
        # Removed termination for embedding fetchers
        event.accept()
