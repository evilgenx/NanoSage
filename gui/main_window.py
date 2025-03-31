#!/usr/bin/env python3
# gui/main_window.py

import sys
import os
import subprocess # To open files/folders
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon # Optional: for window icon

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
        self.setWindowTitle("NanoSage GUI 🧙")
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

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initialize UI elements."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Configuration ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Query Input
        query_group = QGroupBox("Search Query")
        query_layout = QVBoxLayout()
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter your research query here...")
        query_layout.addWidget(self.query_input)
        query_group.setLayout(query_layout)
        left_layout.addWidget(query_group)

        # Configuration Options
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout()

        self.web_search_checkbox = QCheckBox("Enable Web Search")
        config_layout.addRow(self.web_search_checkbox)

        corpus_layout = QHBoxLayout()
        self.corpus_dir_label = QLineEdit()
        self.corpus_dir_label.setPlaceholderText("Optional: Path to local documents")
        self.corpus_dir_label.setReadOnly(True)
        self.corpus_dir_button = QPushButton("Browse...")
        corpus_layout.addWidget(self.corpus_dir_label)
        corpus_layout.addWidget(self.corpus_dir_button)
        config_layout.addRow("Local Corpus:", corpus_layout)

        self.max_depth_spinbox = QSpinBox()
        self.max_depth_spinbox.setRange(0, 10)
        self.max_depth_spinbox.setValue(1)
        config_layout.addRow("Max Recursion Depth:", self.max_depth_spinbox)

        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(1, 50)
        self.top_k_spinbox.setValue(3)
        config_layout.addRow("Top K Results:", self.top_k_spinbox) # Renamed label slightly

        # --- Search Provider Configuration ---
        search_group = QGroupBox("Search Settings")
        search_layout = QFormLayout()
        self.search_provider_combo = QComboBox()
        self.search_provider_combo.addItems(["DuckDuckGo", "SearXNG"])
        # Set initial value from loaded config
        default_provider = self.config_data.get('search', {}).get('provider', 'duckduckgo')
        provider_index = 0 if default_provider == 'duckduckgo' else 1
        self.search_provider_combo.setCurrentIndex(provider_index)
        search_layout.addRow("Search Provider:", self.search_provider_combo)
        # Add more search settings here if needed (e.g., max results, SearXNG URL input)
        search_group.setLayout(search_layout)
        config_layout.addRow(search_group) # Add search group to main config layout


        # --- Embedding Configuration ---
        embedding_group = QGroupBox("Embedding Settings")
        embedding_layout = QFormLayout()

        self.device_combo = QComboBox()
        # Add new API device options
        self.device_combo.addItems(["cpu", "cuda"])
        embedding_layout.addRow("Embedding Device:", self.device_combo)

        # Renamed combo box for clarity
        self.embedding_model_combo = QComboBox()
        # Initially populate based on default device ('cpu')
        self.embedding_model_combo.addItems(["colpali", "all-minilm"])
        self.embedding_model_label = QLabel("Embedding Model:") # Label for the combo
        embedding_layout.addRow(self.embedding_model_label, self.embedding_model_combo)

        # Removed fetch buttons for Gemini/OpenRouter embedding models
        # self.gemini_embedding_fetch_button = QPushButton("Fetch Gemini Embed Models")
        # self.openrouter_embedding_fetch_button = QPushButton("Fetch OpenRouter Embed Models")
        # self.gemini_embedding_fetch_button.setVisible(False)
        # self.openrouter_embedding_fetch_button.setVisible(False)
        # embedding_layout.addRow(self.gemini_embedding_fetch_button)
        # embedding_layout.addRow(self.openrouter_embedding_fetch_button)

        embedding_group.setLayout(embedding_layout)
        config_layout.addRow(embedding_group) # Add embedding group to main config layout

        # --- RAG Configuration ---
        rag_group = QGroupBox("RAG Settings")
        rag_layout = QFormLayout()

        self.rag_model_combo = QComboBox()
        self.rag_model_combo.addItems(["gemma", "pali", "gemini", "openrouter", "None"])
        rag_layout.addRow("RAG Model Type:", self.rag_model_combo)

        # Gemini RAG Specific - initially hidden
        self.gemini_fetch_button = QPushButton("Fetch Gemini Gen Models") # Renamed button
        self.gemini_model_combo = QComboBox()
        self.gemini_fetch_button.setVisible(False)
        self.gemini_model_combo.setVisible(False)
        self.gemini_model_label = QLabel("Select Gemini Gen Model:") # Renamed label
        self.gemini_model_label.setVisible(False)
        rag_layout.addRow(self.gemini_fetch_button)
        rag_layout.addRow(self.gemini_model_label, self.gemini_model_combo)

        # OpenRouter RAG Specific - initially hidden
        self.openrouter_fetch_button = QPushButton("Fetch Free Gen Models") # Renamed button
        self.openrouter_model_combo = QComboBox()
        self.openrouter_fetch_button.setVisible(False)
        self.openrouter_model_combo.setVisible(False)
        self.openrouter_model_label = QLabel("Select OpenRouter Gen Model:") # Renamed label
        self.openrouter_model_label.setVisible(False)
        rag_layout.addRow(self.openrouter_fetch_button)
        rag_layout.addRow(self.openrouter_model_label, self.openrouter_model_combo)

        self.personality_input = QLineEdit()
        self.personality_input.setPlaceholderText("Optional: e.g., 'scientific', 'cheerful'")
        self.personality_label = QLabel("RAG Personality:")
        rag_layout.addRow(self.personality_label, self.personality_input)
        # Initially hide personality if RAG is None
        self.personality_label.setVisible(self.rag_model_combo.currentText() != "None")
        self.personality_input.setVisible(self.rag_model_combo.currentText() != "None")

        rag_group.setLayout(rag_layout)
        config_layout.addRow(rag_group) # Add RAG group to main config layout

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)


        # Execution Button
        self.run_button = QPushButton("Run Search")
        self.run_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #cccccc; }")
        left_layout.addWidget(self.run_button)

        left_layout.addStretch() # Push elements to the top

        # --- Right Panel: Status & Results ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        # Status Log
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout()
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setPlaceholderText("Search progress will appear here...")
        status_layout.addWidget(self.status_log)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)


        # Results
        results_group = QGroupBox("Results")
        results_layout = QFormLayout() # Use FormLayout for label-button pairs
        self.report_path_label = QLabel("Report will appear here.")
        self.report_path_label.setWordWrap(True)
        self.open_report_button = QPushButton("Open Report")
        self.open_folder_button = QPushButton("Open Results Folder")
        self.open_report_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)

        results_layout.addRow("Report Path:", self.report_path_label)
        button_layout = QHBoxLayout() # Layout for buttons side-by-side
        button_layout.addWidget(self.open_report_button)
        button_layout.addWidget(self.open_folder_button)
        results_layout.addRow(button_layout) # Add the button layout as a row

        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 1) # Give left panel less stretch factor
        main_layout.addWidget(right_panel, 2) # Give right panel more stretch factor


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
        # Connect search provider change signal (optional: for saving config immediately)
        # self.search_provider_combo.currentTextChanged.connect(self.handle_search_provider_change)

    # --- Slot Methods ---

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
        search_config = self.config_data.get('search', {})
        provider_config = search_config.get(search_provider_key, {})
        search_limit = provider_config.get('max_results', 5) # Default to 5 if not found
        searxng_url = search_config.get('searxng', {}).get('base_url') if search_provider_key == 'searxng' else None

        # Optional: Save the selected provider back to config immediately
        # self.save_current_config() # Uncomment if you want immediate saving

        # --- Prepare All Parameters ---
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
            "search_limit": search_limit,
            "searxng_url": searxng_url
            # config_path could be added if needed
        }

        # --- Start Worker ---

        self.log_status("Starting search...")
        self.run_button.setEnabled(False)
        self.open_report_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.report_path_label.setText("Running...")
        self.current_report_path = None
        self.current_results_dir = None


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
