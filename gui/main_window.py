#!/usr/bin/env python3
# gui/main_window.py

import sys
import os
import subprocess # To open files/folders
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
        SearchWorker, GeminiFetcher, OpenRouterFetcher
        # Removed GeminiEmbeddingFetcher, OpenRouterEmbeddingFetcher as they are no longer used
    )
except ImportError as e:
    print(f"Error importing workers in main_window.py: {e}")
    # Handle appropriately, maybe raise or show error in GUI if possible later
    sys.exit(1)

# Import config loading utility
from config_utils import load_config, save_config # Added save_config

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
        # Generative model fetchers
        self.gemini_fetcher = None
        self.openrouter_fetcher = None
        # Embedding model fetchers removed as options are gone
        # self.gemini_embedding_fetcher = None
        # self.openrouter_embedding_fetcher = None
        self.current_report_path = None
        self.current_results_dir = None

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
        self.searxng_time_range_input.setText(searxng_config.get('time_range', ''))
        self.searxng_categories_input.setText(searxng_config.get('categories', ''))
        self.searxng_engines_input.setText(searxng_config.get('engines', ''))

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
        self.web_search_checkbox.setChecked(self.config_data.get('search', {}).get('web_search_enabled', True)) # Example
        self.max_depth_spinbox.setValue(self.config_data.get('search', {}).get('max_depth', 1))
        self.top_k_spinbox.setValue(self.config_data.get('search', {}).get('top_k', 3))
        self.corpus_dir_label.setText(self.config_data.get('corpus', {}).get('path', '')) # Example
        self.personality_input.setText(self.config_data.get('rag', {}).get('personality', ''))

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
        # Connect search provider change signal to show/hide SearXNG options
        self.search_provider_combo.currentTextChanged.connect(self.handle_search_provider_change)
        # Connect cancel button
        self.cancel_button.clicked.connect(self.cancel_search)

    # --- Slot Methods ---

    def handle_search_provider_change(self, provider_text):
        """Show/hide SearXNG specific settings."""
        is_searxng = (provider_text == "SearXNG")
        self.searxng_base_url_label.setVisible(is_searxng)
        self.searxng_base_url_input.setVisible(is_searxng)
        self.searxng_time_range_label.setVisible(is_searxng)
        self.searxng_time_range_input.setVisible(is_searxng)
        self.searxng_categories_label.setVisible(is_searxng)
        self.searxng_categories_input.setVisible(is_searxng)
        self.searxng_engines_label.setVisible(is_searxng)
        self.searxng_engines_input.setVisible(is_searxng)

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

        if device_name == "cpu" or device_name == "cuda":
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
        """Validate inputs and start the SearchWorker thread."""
        # --- Input Validation ---
        if self.search_worker and self.search_worker.isRunning():
            self.log_status("Search is already in progress.")
            return

        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Input Error", "Please enter a search query.")
            return

        # Embedding settings validation
        embedding_device = self.device_combo.currentText()
        embedding_model_name = self.embedding_model_combo.currentText()
        if not embedding_model_name:
             QMessageBox.warning(self, "Input Error", f"Please select an Embedding Model for the '{embedding_device}' device.")
             return
        if embedding_device in ["Gemini", "OpenRouter"] and self.embedding_model_combo.count() == 0:
             QMessageBox.warning(self, "Input Error", f"Please fetch and select an Embedding Model for {embedding_device}.")
             return


        # RAG settings validation
        rag_model_type = self.rag_model_combo.currentText()
        selected_generative_gemini = None
        selected_generative_openrouter = None

        if rag_model_type == "gemini":
            if self.gemini_model_combo.count() == 0 or not self.gemini_model_combo.currentText():
                 QMessageBox.warning(self, "Input Error", "RAG Model is Gemini, but no generative model selected. Please fetch and select one.")
                 return
            selected_generative_gemini = self.gemini_model_combo.currentText()
        elif rag_model_type == "openrouter":
            if self.openrouter_model_combo.count() == 0 or not self.openrouter_model_combo.currentText():
                 QMessageBox.warning(self, "Input Error", "RAG Model is OpenRouter, but no generative model selected. Please fetch and select one.")
                 return
            selected_generative_openrouter = self.openrouter_model_combo.currentText()

        # --- Prepare Search Provider Parameters ---
        selected_provider_text = self.search_provider_combo.currentText()
        search_provider_key = 'duckduckgo' if selected_provider_text == "DuckDuckGo" else 'searxng'

        # Get provider-specific settings from loaded config
        # --- Save Current Config & Prepare Search Provider Parameters ---
        search_config = self.config_data.setdefault('search', {}) # Ensure 'search' key exists
        search_config['provider'] = search_provider_key # Save selected provider

        searxng_config = search_config.setdefault('searxng', {}) # Ensure 'searxng' key exists
        ddg_config = search_config.setdefault('duckduckgo', {}) # Ensure 'duckduckgo' key exists

        search_limit = 5 # Default limit
        searxng_url = None
        searxng_time_range = None
        searxng_categories = None
        searxng_engines = None

        if search_provider_key == 'searxng':
            # Read from GUI and update config_data
            searxng_url = self.searxng_base_url_input.text().strip() or None
            searxng_time_range = self.searxng_time_range_input.text().strip() or None
            searxng_categories = self.searxng_categories_input.text().strip() or None
            searxng_engines = self.searxng_engines_input.text().strip() or None

            searxng_config['base_url'] = searxng_url
            searxng_config['time_range'] = searxng_time_range
            searxng_config['categories'] = searxng_categories
            searxng_config['engines'] = searxng_engines
            # Note: max_results for searxng is read inside download_webpages_searxng

            # Try saving the updated config
            if save_config(self.config_path, self.config_data):
                self.log_status(f"SearXNG settings saved to {self.config_path}")
            else:
                self.log_status(f"[ERROR] Failed to save configuration to {self.config_path}")
                QMessageBox.warning(self, "Config Error", f"Could not save settings to {self.config_path}")
                # Decide if you want to proceed without saving or stop
                # return # Uncomment to stop if saving fails

            # Use the limit from the config for searxng (read internally by worker)
            search_limit = searxng_config.get('max_results', 5)

        elif search_provider_key == 'duckduckgo':
            # Use the limit from the config for duckduckgo
            search_limit = ddg_config.get('max_results', 5)
            # Try saving the updated config (only provider changed)
            if save_config(self.config_path, self.config_data):
                 self.log_status(f"Search provider saved to {self.config_path}")
            else:
                 self.log_status(f"[ERROR] Failed to save configuration to {self.config_path}")


        # --- Prepare All Parameters for Worker ---
        params = {
            "query": query,
            "corpus_dir": self.corpus_dir_label.text() or None,
            "web_search": self.web_search_checkbox.isChecked(),
            "max_depth": self.max_depth_spinbox.value(),
            "top_k": self.top_k_spinbox.value(),
            "device": embedding_device, # Pass the selected embedding device
            "embedding_model_name": embedding_model_name, # Pass the specific embedding model name
            "rag_model": rag_model_type if rag_model_type != "None" else None,
            "personality": self.personality_input.text() or None,
            "selected_gemini_model": selected_generative_gemini, # Specific generative model
            "selected_openrouter_model": selected_generative_openrouter, # Specific generative model
            # Add resolved search settings
            "search_provider": search_provider_key,
            "search_limit": search_limit, # Pass the resolved limit for DDG
            # Pass SearXNG specific params read from GUI (worker will use config, but pass for clarity/potential override)
            "searxng_url": searxng_url,
            "searxng_time_range": searxng_time_range,
            "searxng_categories": searxng_categories,
            "searxng_engines": searxng_engines,
            "config_path": self.config_path # Pass config path so worker can reload fresh config
        }

        # --- Start Worker ---

        self.log_status("Starting search...")
        self.run_button.setEnabled(False)
        self.open_report_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.report_path_label.setText("Running...")
        self.current_report_path = None
        self.current_results_dir = None

        # Show and start indeterminate progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate mode

        # Show and enable cancel button
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)

        # --- Call initial visibility handler for search provider ---
        self.handle_search_provider_change(self.search_provider_combo.currentText())


        self.search_worker = SearchWorker(params, self)
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

    def on_search_error(self, error_message):
        """Show error message if search fails."""
        self.log_status(f"Search Error: {error_message}")
        QMessageBox.critical(self, "Search Error", error_message)
        self.report_path_label.setText("Search failed.")


    def on_search_finished(self):
        """Called when the search thread finishes (success or error)."""
        self.run_button.setEnabled(True)
        self.search_worker = None # Allow starting a new search
        # Hide and reset progress bar
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100) # Reset range
        # Hide and disable cancel button
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)

    def cancel_search(self):
        """Requests cancellation of the currently running search worker."""
        if self.search_worker and self.search_worker.isRunning():
            self.log_status("Requesting search cancellation...")
            self.search_worker.request_cancellation() # Need to implement this method in SearchWorker
            self.cancel_button.setEnabled(False) # Prevent multiple clicks
            self.log_status("Cancellation requested. Waiting for worker to stop...")
        else:
            self.log_status("No search running to cancel.")


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

    def closeEvent(self, event):
        """Ensure threads are stopped on close."""
        # Add logic here if threads need graceful shutdown
        if self.search_worker and self.search_worker.isRunning():
             # Optionally wait or terminate
             pass
        # Stop generative fetchers
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
             self.gemini_fetcher.terminate() # Or use a more graceful stop if implemented
             self.gemini_fetcher.wait()
        if self.openrouter_fetcher and self.openrouter_fetcher.isRunning():
             self.openrouter_fetcher.terminate()
             self.openrouter_fetcher.wait()
        # Removed termination logic for embedding fetchers
        # if self.gemini_embedding_fetcher and self.gemini_embedding_fetcher.isRunning():
        #      self.gemini_embedding_fetcher.terminate()
        #      self.gemini_embedding_fetcher.wait()
        # if self.openrouter_embedding_fetcher and self.openrouter_embedding_fetcher.isRunning():
        #      self.openrouter_embedding_fetcher.terminate()
        #      self.openrouter_embedding_fetcher.wait()
        event.accept()
