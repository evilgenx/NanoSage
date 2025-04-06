#!/usr/bin/env python3
# gui/main_window.py

import sys
import os
# import subprocess # Moved to result_actions
# import webbrowser # Moved to result_actions
# import urllib.parse # Moved to result_actions
import logging # Added logging
from PyQt6.QtWidgets import (
    QMainWindow, QTextEdit, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QFileDialog, QMessageBox # Removed unused layout/widget imports
)
# from PyQt6.QtCore import Qt # Removed unused Qt
# from PyQt6.QtGui import QIcon # Optional: for window icon

# Import the new UI setup function
from .ui_setup import setup_main_window_ui
# Import the new controller and result actions
from .controller import GuiController
from . import result_actions # Import the module

# Import config loading utility
from config_utils import load_config, save_config, DEFAULT_CONFIG # Added DEFAULT_CONFIG
from cache_manager import CacheManager # Added CacheManager import

from PyQt6.QtGui import QAction # Import QAction for menu items

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

        # Worker instances are now managed by the controller
        # self.search_worker = None
        # self.topic_extractor_worker = None
        # self.query_enhancer_worker = None
        # self.gemini_fetcher = None
        # self.openrouter_fetcher = None

        self.current_report_path = None
        self.current_results_dir = None
        # self.cache_manager_instance = None # Cache manager interaction might move to controller too

        # Call the UI setup function from the separate module
        setup_main_window_ui(self)

        # Instantiate the controller
        self.controller = GuiController(self)

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

        # Initial RAG model state (load from new config keys)
        llm_config_init = self.config_data.get('llm', {}) # Get LLM config once
        default_rag_model_type = llm_config_init.get('rag_model_type', 'gemma')
        self.rag_model_combo.setCurrentText(default_rag_model_type)
        self.handle_rag_model_change(default_rag_model_type) # Trigger visibility updates

        # Load RAG personality
        self.personality_input.setText(llm_config_init.get('rag_personality', ''))
        # Note: Selected Gemini/OpenRouter models are stored in config but applied after fetching.

        # Set other config-dependent initial states if any (e.g., checkbox, spinboxes)
        general_config_init = self.config_data.get('general', {}) # Get general config
        retrieval_config_init = self.config_data.get('retrieval', {}) # Get retrieval config
        search_config_init = self.config_data.get('search', {}) # Get search config

        self.web_search_checkbox.setChecked(general_config_init.get('web_search', True))
        self.iterative_search_checkbox.setChecked(search_config_init.get('enable_iterative_search', False)) # Assuming this key exists or add default
        self.max_depth_spinbox.setValue(general_config_init.get('max_depth', 1))
        self.top_k_spinbox.setValue(retrieval_config_init.get('top_k', 3))
        self.corpus_dir_label.setText(general_config_init.get('corpus_dir', '') or '') # Use general config

        # Cache settings initialization
        cache_config_init = self.config_data.get('cache', {})
        self.cache_enabled_checkbox.setChecked(cache_config_init.get('enabled', False)) # Default to False if not in config

        # Populate Output Format dropdown
        self.output_format_combo.clear()
        output_formats_config = self.config_data.get('llm', {}).get('output_formats', {})
        if output_formats_config:
            self.output_format_combo.addItems(output_formats_config.keys())
            # Set selected output format from config
            selected_format = llm_config_init.get('selected_output_format', 'report') # Default to 'report'
            if selected_format in output_formats_config:
                self.output_format_combo.setCurrentText(selected_format)
            elif self.output_format_combo.count() > 0:
                 # If saved format not found, select the first available one
                 self.output_format_combo.setCurrentIndex(0)
        else:
            self.log_status("[Warning] No 'output_formats' found in config.yaml under 'llm'. Dropdown will be empty.")
            self.output_format_combo.setEnabled(False)


        # Connect signals after UI is fully set up and initialized
        self._connect_signals()

        # --- Create Menu Bar ---
        self._create_menu_bar()

    # _init_ui method is now removed

    def _create_menu_bar(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        # File Menu (Example - if needed later)
        # file_menu = menu_bar.addMenu("&File")
        # exit_action = QAction("E&xit", self)
        # exit_action.triggered.connect(self.close)
        # file_menu.addAction(exit_action)

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")
        scrape_action = QAction("&Scrape URL...", self)
        # Connect to a controller method (to be created)
        scrape_action.triggered.connect(self.controller.show_scrape_dialog)
        tools_menu.addAction(scrape_action)

    def _connect_signals(self):
        """Connect UI signals to slots in the controller or result actions."""
        # Connect run button to controller's start method (Handled in GuiController.__init__)
        # self.run_button.clicked.connect(self.controller.start_search_process) # Redundant
        self.corpus_dir_button.clicked.connect(self.select_corpus_directory)
        # Connect device change signal (still handled locally for UI updates)
        self.device_combo.currentTextChanged.connect(self.handle_device_change)
        # Connect RAG model change signal
        self.rag_model_combo.currentTextChanged.connect(self.handle_rag_model_change)
        # Connect generative model fetch buttons to controller's ModelFetcherManager
        self.gemini_fetch_button.clicked.connect(self.controller.model_fetcher_manager.fetch_gemini_models)
        self.openrouter_fetch_button.clicked.connect(self.controller.model_fetcher_manager.fetch_openrouter_models)
        # Connect result buttons to result_actions functions (using lambda to pass state)
        self.open_report_button.clicked.connect(lambda: result_actions.open_report(self.current_report_path, self.log_status))
        self.open_folder_button.clicked.connect(lambda: result_actions.open_results_folder(self.current_results_dir, self.log_status))
        self.share_email_button.clicked.connect(lambda: result_actions.share_report_email(self.current_report_path, self.log_status))
        # Connect search provider change signal (still handled locally)
        self.search_provider_combo.currentTextChanged.connect(self.handle_search_provider_change)
        # Connect the engine selector's signal (still handled locally for config saving)
        self.searxng_engine_selector.selectionChanged.connect(self._handle_searxng_engine_selection_change)
        # Connect cancel button to controller
        self.cancel_button.clicked.connect(self.controller.cancel_current_operation)
        # Connect cache controls (handled by _save_current_settings_to_config on exit)
        # self.cache_enabled_checkbox.stateChanged.connect(self._handle_cache_enabled_change) # No longer needed for immediate save
        self.clear_cache_button.clicked.connect(self.controller.clear_cache) # Connect clear button to controller
        # Connect corpus clear button (handled by _save_current_settings_to_config on exit)
        self.corpus_clear_button.clicked.connect(self.clear_corpus_directory) # Still need the action


    # --- Slot Methods (Keep UI-specific handlers) ---

    # Worker slots (on_*, fetch_*) are moved to GuiController

    def _handle_searxng_engine_selection_change(self, selected_engines):
        """Update the config when SearXNG engine selection changes."""
        # This might also move to the controller if config handling is centralized there
        if 'search' not in self.config_data: self.config_data['search'] = {}
        if 'searxng' not in self.config_data['search']: self.config_data['search']['searxng'] = {}

        self.config_data['search']['searxng']['engines'] = selected_engines

        # Config is now saved on exit, no immediate save needed here
        # if save_config(self.config_path, self.config_data):
        #     self.log_status(f"SearXNG engine selection updated in config (will save on exit)")
        # else:
        #     self.log_status(f"[ERROR] Failed to update config data in memory.")
        #     QMessageBox.warning(self, "Config Error", f"Could not update engine selection in memory.")
        pass # No immediate save action needed

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
                self.embedding_model_label.setText("Embedding Model:")


    def select_corpus_directory(self):
        """Open dialog to select local corpus directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Corpus Directory")
        if directory:
            self.corpus_dir_label.setText(directory)
            # Optionally update config immediately or let controller handle it before search
            # if 'corpus' not in self.config_data: self.config_data['corpus'] = {}
            # self.config_data['corpus']['path'] = directory
            # Update config in memory, save on exit
            if 'general' not in self.config_data: self.config_data['general'] = {}
            self.config_data['general']['corpus_dir'] = directory
            self.log_status(f"Corpus directory set to: {directory} (will save on exit)")
            # No immediate save needed here
            # if save_config(self.config_path, self.config_data): ...

    def clear_corpus_directory(self):
        """Clear the selected corpus directory path."""
        self.corpus_dir_label.setText("")
        # Update config in memory, save on exit
        if 'general' not in self.config_data: self.config_data['general'] = {}
        self.config_data['general']['corpus_dir'] = "" # Set path to empty string
        self.log_status("Corpus directory cleared (will save on exit)")
        # No immediate save needed here
        # if save_config(self.config_path, self.config_data): ...


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


    # --- Generative Model Fetching Slots (Moved to Controller) ---
    # def fetch_gemini_models(self): ...
    # def on_gemini_models_fetched(self, models): ...
    # def on_gemini_fetch_error(self, error_message): ...
    # def fetch_openrouter_models(self): ...
    # def on_openrouter_models_fetched(self, models): ...
    # def on_openrouter_fetch_error(self, error_message): ...

    # --- Search Execution (Moved to Controller) ---
    # def start_search(self): ...
    # def _start_main_search_worker(self, query_to_use): ...

    # --- Cache Handling ---
    # Cache enabled state is now saved on exit via _save_current_settings_to_config
    # def _handle_cache_enabled_change(self, state): ... # No longer needed

    # clear_cache is handled by the controller via button signal

    # _start_main_search_worker is moved to GuiController

    # --- Utility & Result Handling ---

    # log_status remains here as it directly interacts with the UI element
    def log_status(self, message):
        """Append a message to the status log."""
        self.status_log.append(message)
        self.status_log.verticalScrollBar().setValue(self.status_log.verticalScrollBar().maximum()) # Auto-scroll

    # Search result slots (on_search_complete/error/finished) are moved to GuiController

    # cancel_search is moved to GuiController

    # Result action methods (open_report, open_results_folder, share_report_email) are moved to result_actions.py

    # --- Configuration Saving ---
    def _save_current_settings_to_config(self):
        """Gather current UI settings and update self.config_data."""
        try:
            # General Tab
            if 'general' not in self.config_data: self.config_data['general'] = {}
            self.config_data['general']['web_search'] = self.web_search_checkbox.isChecked()
            self.config_data['general']['corpus_dir'] = self.corpus_dir_label.text()
            self.config_data['general']['max_depth'] = self.max_depth_spinbox.value()
            # Assuming device is under general now based on DEFAULT_CONFIG structure? Check config_utils.py if needed.
            # If device is still under 'embeddings', adjust accordingly. Let's assume 'general' for now.
            self.config_data['general']['device'] = self.device_combo.currentText()

            # Retrieval Tab (assuming top_k is here)
            if 'retrieval' not in self.config_data: self.config_data['retrieval'] = {}
            self.config_data['retrieval']['top_k'] = self.top_k_spinbox.value()
            # Assuming embedding_model is under retrieval
            self.config_data['retrieval']['embedding_model'] = self.embedding_model_combo.currentText()

            # Search Tab
            if 'search' not in self.config_data: self.config_data['search'] = {}
            self.config_data['search']['provider'] = self.search_provider_combo.currentText().lower() # Save as lowercase
            self.config_data['search']['enable_iterative_search'] = self.iterative_search_checkbox.isChecked() # Save iterative search state
            if 'searxng' not in self.config_data['search']: self.config_data['search']['searxng'] = {}
            self.config_data['search']['searxng']['base_url'] = self.searxng_base_url_input.text()
            self.config_data['search']['searxng']['time_range'] = self.searxng_time_range_input.text() or None # Save None if empty
            self.config_data['search']['searxng']['categories'] = self.searxng_categories_input.text() or None # Save None if empty
            self.config_data['search']['searxng']['engines'] = self.searxng_engine_selector.getSelectedEngines()

            # RAG Tab (LLM section)
            if 'llm' not in self.config_data: self.config_data['llm'] = {}
            self.config_data['llm']['rag_model_type'] = self.rag_model_combo.currentText()
            self.config_data['llm']['rag_personality'] = self.personality_input.text()
            self.config_data['llm']['selected_output_format'] = self.output_format_combo.currentText()
            # Save the currently selected model IDs if the combo boxes have items
            if self.gemini_model_combo.count() > 0:
                 self.config_data['llm']['selected_gemini_model'] = self.gemini_model_combo.currentText()
            if self.openrouter_model_combo.count() > 0:
                 self.config_data['llm']['selected_openrouter_model'] = self.openrouter_model_combo.currentText()

            # Cache Tab
            if 'cache' not in self.config_data: self.config_data['cache'] = {}
            self.config_data['cache']['enabled'] = self.cache_enabled_checkbox.isChecked()

            # Save the updated config data to the file
            if save_config(self.config_path, self.config_data):
                self.log_status(f"Configuration saved successfully to {self.config_path}")
            else:
                # Log error, but don't prevent closing
                self.log_status(f"[ERROR] Failed to save configuration to {self.config_path} on exit.")
                # Optionally show a non-blocking message? Maybe too intrusive on close.

        except Exception as e:
            # Log any unexpected error during saving
            self.log_status(f"[ERROR] Unexpected error saving configuration: {e}")
            logging.exception("Error during configuration save on exit:") # Log traceback


    def closeEvent(self, event):
        """Save settings and ensure threads are stopped on close."""
        self.log_status("Saving configuration before exiting...")
        self._save_current_settings_to_config() # Save settings first

        if self.controller:
            self.log_status("Shutting down background workers...")
            self.controller.shutdown_workers()

        self.log_status("Exiting application.")
        event.accept()
