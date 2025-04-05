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
        """Connect UI signals to slots in the controller or result actions."""
        # Connect run button to controller's start method
        self.run_button.clicked.connect(self.controller.start_search_process)
        self.corpus_dir_button.clicked.connect(self.select_corpus_directory)
        # Connect device change signal (still handled locally for UI updates)
        self.device_combo.currentTextChanged.connect(self.handle_device_change)
        # Connect RAG model change signal
        self.rag_model_combo.currentTextChanged.connect(self.handle_rag_model_change)
        # Connect generative model fetch buttons to controller
        self.gemini_fetch_button.clicked.connect(self.controller.fetch_gemini_models)
        self.openrouter_fetch_button.clicked.connect(self.controller.fetch_openrouter_models)
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
        # Connect cache controls (still handled locally/via controller)
        self.cache_enabled_checkbox.stateChanged.connect(self._handle_cache_enabled_change)
        self.clear_cache_button.clicked.connect(self.controller.clear_cache) # Connect clear button to controller
        # Connect corpus clear button
        self.corpus_clear_button.clicked.connect(self.clear_corpus_directory)


    # --- Slot Methods (Keep UI-specific handlers) ---

    # Worker slots (on_*, fetch_*) are moved to GuiController

    def _handle_searxng_engine_selection_change(self, selected_engines):
        """Update the config when SearXNG engine selection changes."""
        # This might also move to the controller if config handling is centralized there
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
                self.embedding_model_label.setText("Embedding Model:")


    def select_corpus_directory(self):
        """Open dialog to select local corpus directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Corpus Directory")
        if directory:
            self.corpus_dir_label.setText(directory)
            # Optionally update config immediately or let controller handle it before search
            # if 'corpus' not in self.config_data: self.config_data['corpus'] = {}
            # self.config_data['corpus']['path'] = directory
            # Optionally update config immediately or let controller handle it before search
            # We will save it here for immediate effect
            if 'corpus' not in self.config_data: self.config_data['corpus'] = {}
            self.config_data['corpus']['path'] = directory
            if save_config(self.config_path, self.config_data):
                self.log_status(f"Corpus directory saved to {self.config_path}")
            else:
                self.log_status(f"[ERROR] Failed to save corpus directory to {self.config_path}")
                QMessageBox.warning(self, "Config Error", f"Could not save corpus directory to {self.config_path}")

    def clear_corpus_directory(self):
        """Clear the selected corpus directory path."""
        self.corpus_dir_label.setText("")
        if 'corpus' not in self.config_data: self.config_data['corpus'] = {}
        self.config_data['corpus']['path'] = "" # Set path to empty string
        if save_config(self.config_path, self.config_data):
            self.log_status("Corpus directory cleared and saved.")
        else:
            self.log_status(f"[ERROR] Failed to save cleared corpus directory to {self.config_path}")
            QMessageBox.warning(self, "Config Error", f"Could not save cleared corpus directory to {self.config_path}")


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
    def _handle_cache_enabled_change(self, state):
        """Update config when cache enabled checkbox changes."""
        # This logic remains here as it directly affects the config this window manages
        enabled = (state == 2) # 2 means checked
        if 'cache' not in self.config_data: self.config_data['cache'] = {}
        self.config_data['cache']['enabled'] = enabled
        if save_config(self.config_path, self.config_data):
            self.log_status(f"Cache setting saved to {self.config_path} (Enabled: {enabled})")
        else:
            self.log_status(f"[ERROR] Failed to save cache setting to {self.config_path}")
            QMessageBox.warning(self, "Config Error", f"Could not save cache setting to {self.config_path}")

    # clear_cache is now handled by the controller via button signal

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

    def closeEvent(self, event):
        """Ensure threads are stopped on close by notifying the controller."""
        if self.controller:
            self.controller.shutdown_workers()
        event.accept()
