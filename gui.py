#!/usr/bin/env python3
# gui.py

import sys
import os
import asyncio
import subprocess # To open files/folders
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QSizePolicy
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject
from PyQt6.QtGui import QIcon # Optional: for window icon

# Assuming search_session and main are in the same directory or Python path
try:
    from search_session import SearchSession, list_gemini_models
    from main import load_config # Import load_config from main.py
except ImportError as e:
    print(f"Error importing from search_session or main: {e}")
    print("Ensure search_session.py is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

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

    def run(self):
        """Executes the search session."""
        try:
            # Create a proxy object to safely emit signals from the asyncio loop
            class ProgressCallbackProxy(QObject):
                progress_signal = pyqtSignal(str)
                def __call__(self, message):
                    self.progress_signal.emit(message)

            self._progress_callback_proxy = ProgressCallbackProxy()
            self._progress_callback_proxy.progress_signal.connect(self.progress_updated.emit)

            self.progress_updated.emit("Initializing search session...")

            # Load config (assuming config.yaml exists or is handled)
            config = load_config(self.params.get("config_path", "config.yaml"))

            session = SearchSession(
                query=self.params["query"],
                config=config,
                corpus_dir=self.params.get("corpus_dir"),
                device=self.params.get("device", "cpu"),
                retrieval_model=self.params.get("retrieval_model", "colpali"),
                top_k=self.params.get("top_k", 3),
                web_search_enabled=self.params.get("web_search", False),
                personality=self.params.get("personality"),
                rag_model=self.params.get("rag_model", "gemma"),
                selected_gemini_model=self.params.get("selected_gemini_model"),
                max_depth=self.params.get("max_depth", 1),
                progress_callback=self._progress_callback_proxy # Pass the callable proxy
            )

            self.progress_updated.emit("Starting search process...")
            # Run the asyncio event loop within the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_answer = loop.run_until_complete(session.run_session())
            loop.close()

            self.progress_updated.emit("Saving final report...")
            output_path = session.save_report(final_answer)
            self.progress_updated.emit(f"Report saved: {output_path}")
            self.search_complete.emit(output_path)

        except ImportError as e:
             self.error_occurred.emit(f"Import Error: {e}. Check dependencies.")
        except FileNotFoundError as e:
             self.error_occurred.emit(f"File Not Found Error: {e}")
        except Exception as e:
            self.error_occurred.emit(f"An error occurred during search: {e}")
        finally:
            # Clean up proxy if it was created
            if self._progress_callback_proxy:
                self._progress_callback_proxy.progress_signal.disconnect()
                self._progress_callback_proxy = None


class GeminiFetcher(QThread):
    """Fetches available Gemini models in a separate thread."""
    models_fetched = pyqtSignal(list)
    fetch_error = pyqtSignal(str)
    status_update = pyqtSignal(str) # New signal for status updates

    def run(self):
        """Executes the model fetching."""
        try:
            self.status_update.emit("Fetching Gemini models...") # Use status_update signal
            models = list_gemini_models()
            if models is None:
                self.fetch_error.emit("Could not retrieve Gemini models. Check API key/network.")
            elif not models:
                self.fetch_error.emit("No suitable Gemini models found.")
            else:
                self.models_fetched.emit(models)
        except Exception as e:
            self.fetch_error.emit(f"Error fetching Gemini models: {e}")


# --- Main Window ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NanoSage GUI 🧙")
        # self.setWindowIcon(QIcon("path/to/icon.png")) # Optional

        self.search_worker = None
        self.gemini_fetcher = None
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
        config_layout.addRow("Top K Local Results:", self.top_k_spinbox)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"]) # Add more if needed
        config_layout.addRow("Device:", self.device_combo)

        self.retrieval_model_combo = QComboBox()
        self.retrieval_model_combo.addItems(["colpali", "all-minilm"])
        config_layout.addRow("Retrieval Model:", self.retrieval_model_combo)

        self.rag_model_combo = QComboBox()
        self.rag_model_combo.addItems(["gemma", "pali", "gemini", "None"]) # Added None
        config_layout.addRow("RAG Model:", self.rag_model_combo)

        # Gemini Specific - initially hidden
        self.gemini_fetch_button = QPushButton("Fetch Gemini Models")
        self.gemini_model_combo = QComboBox()
        self.gemini_fetch_button.setVisible(False)
        self.gemini_model_combo.setVisible(False)
        config_layout.addRow(self.gemini_fetch_button)
        config_layout.addRow("Select Gemini Model:", self.gemini_model_combo)


        self.personality_input = QLineEdit()
        self.personality_input.setPlaceholderText("Optional: e.g., 'scientific', 'cheerful'")
        self.personality_label = QLabel("RAG Personality:") # Label for the input
        config_layout.addRow(self.personality_label, self.personality_input)
        # Initially hide personality if RAG is None
        self.personality_label.setVisible(self.rag_model_combo.currentText() != "None")
        self.personality_input.setVisible(self.rag_model_combo.currentText() != "None")


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
        self.rag_model_combo.currentTextChanged.connect(self.handle_rag_model_change)
        self.gemini_fetch_button.clicked.connect(self.fetch_gemini_models)
        self.open_report_button.clicked.connect(self.open_report)
        self.open_folder_button.clicked.connect(self.open_results_folder)

    # --- Slot Methods ---

    def select_corpus_directory(self):
        """Open dialog to select local corpus directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Corpus Directory")
        if directory:
            self.corpus_dir_label.setText(directory)

    def handle_rag_model_change(self, model_name):
        """Show/hide Gemini and Personality options based on RAG model selection."""
        is_gemini = (model_name == "gemini")
        is_rag_enabled = (model_name != "None")

        self.gemini_fetch_button.setVisible(is_gemini)
        self.gemini_model_combo.setVisible(is_gemini)
        # Clear Gemini combo if switching away
        if not is_gemini:
            self.gemini_model_combo.clear()
            self.gemini_fetch_button.setEnabled(True) # Re-enable fetch button

        self.personality_label.setVisible(is_rag_enabled)
        self.personality_input.setVisible(is_rag_enabled)


    def fetch_gemini_models(self):
        """Start the GeminiFetcher thread."""
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
            self.log_status("Already fetching Gemini models...")
            return

        self.log_status("Attempting to fetch Gemini models...")
        self.gemini_fetch_button.setEnabled(False)
        self.gemini_fetcher = GeminiFetcher(self)
        self.gemini_fetcher.status_update.connect(self.log_status) # Connect new status signal
        self.gemini_fetcher.models_fetched.connect(self.on_gemini_models_fetched)
        self.gemini_fetcher.fetch_error.connect(self.on_gemini_fetch_error)
        # Also connect finished signal to re-enable button regardless of outcome
        self.gemini_fetcher.finished.connect(lambda: self.gemini_fetch_button.setEnabled(True))
        self.gemini_fetcher.start()

    def on_gemini_models_fetched(self, models):
        """Populate the Gemini model combo box."""
        self.log_status(f"Successfully fetched {len(models)} Gemini models.")
        self.gemini_model_combo.clear()
        self.gemini_model_combo.addItems(models)
        self.gemini_model_combo.setEnabled(True)
        self.gemini_fetch_button.setEnabled(False) # Disable after successful fetch

    def on_gemini_fetch_error(self, error_message):
        """Show error message if Gemini fetch fails."""
        self.log_status(f"Gemini Fetch Error: {error_message}")
        # QMessageBox.warning(self, "Gemini Model Fetch Error", error_message) # Removed the pop-up dialog
        self.gemini_model_combo.clear()
        self.gemini_model_combo.setEnabled(False)
        self.gemini_fetch_button.setEnabled(True) # Re-enable on error


    def start_search(self):
        """Validate inputs and start the SearchWorker thread."""
        if self.search_worker and self.search_worker.isRunning():
            self.log_status("Search is already in progress.")
            return

        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Input Error", "Please enter a search query.")
            return

        rag_model = self.rag_model_combo.currentText()
        selected_gemini = None
        if rag_model == "gemini":
            if self.gemini_model_combo.count() == 0:
                 QMessageBox.warning(self, "Input Error", "Please fetch and select a Gemini model first.")
                 return
            selected_gemini = self.gemini_model_combo.currentText()


        params = {
            "query": query,
            "corpus_dir": self.corpus_dir_label.text() or None,
            "web_search": self.web_search_checkbox.isChecked(),
            "max_depth": self.max_depth_spinbox.value(),
            "top_k": self.top_k_spinbox.value(),
            "device": self.device_combo.currentText(),
            "retrieval_model": self.retrieval_model_combo.currentText(),
            "rag_model": rag_model if rag_model != "None" else None, # Pass None if "None" selected
            "personality": self.personality_input.text() or None,
            "selected_gemini_model": selected_gemini
            # config_path could be added if needed
        }

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
        self.search_worker.finished.connect(self.on_search_finished) # Re-enable button
        self.search_worker.start()

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
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
             # Optionally wait or terminate
             pass
        event.accept()


# --- Application Entry Point ---

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
