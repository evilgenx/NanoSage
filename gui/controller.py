#!/usr/bin/env python3
# gui/controller.py

import logging
import logging
import os
import re # Keep re for refinement content extraction if needed here, or move if fully delegated
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QModelIndex
from PyQt6.QtWidgets import QMessageBox, QMenu, QInputDialog

# Import config loading utility
from config_utils import load_config, save_config, DEFAULT_CONFIG
from cache_manager import CacheManager

# Import the new controller/manager classes
try:
    from .search_orchestrator import SearchOrchestrator
    from .result_display_manager import ResultDisplayManager
    from .refinement_controller import RefinementController
    from .model_fetcher_manager import ModelFetcherManager
    from .scrape_dialog import ScrapeDialog # Import the new dialog
    from .workers import ScrapeWorker # Import the worker (will create later)
except ImportError as e:
    print(f"Error importing sub-controllers in controller.py: {e}")
    import sys
    sys.exit(1)

class GuiController(QObject):
    """
    Handles the application logic, worker management, and interaction
    between the UI (MainWindow) and the backend processes.
    """
    # No signals defined directly in GuiController anymore

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.config_path = main_window.config_path
        self.config_data = main_window.config_data # Use initial config

        # Instantiate the sub-controllers/managers
        self.result_display_manager = ResultDisplayManager(main_window, parent=self)
        self.search_orchestrator = SearchOrchestrator(main_window, self.result_display_manager, self.config_path, parent=self)
        self.refinement_controller = RefinementController(main_window, self.config_path, parent=self)
        self.model_fetcher_manager = ModelFetcherManager(main_window, parent=self)

        # Connect signals from sub-controllers to GuiController slots or MainWindow slots
        self.search_orchestrator.status_update.connect(self.log_status)
        self.search_orchestrator.search_process_complete.connect(self._handle_search_complete)
        self.search_orchestrator.search_process_error.connect(self._handle_search_error)
        # Connect new orchestrator signals for TOC
        self.search_orchestrator.search_started.connect(self.main_window._clear_toc_tree)
        self.search_orchestrator.toc_node_added.connect(self.main_window._on_toc_node_added)
        self.search_orchestrator.toc_node_updated.connect(self.main_window._on_toc_node_updated)


        self.result_display_manager.status_update.connect(self.log_status)
        self.result_display_manager.refinement_requested.connect(self._handle_refinement_request) # Connect to new slot

        self.refinement_controller.status_update.connect(self.log_status)
        self.refinement_controller.refinement_process_complete.connect(self._handle_refinement_complete) # Connect to new slot
        self.refinement_controller.refinement_process_error.connect(self._handle_refinement_error) # Connect to new slot

        self.model_fetcher_manager.status_update.connect(self.log_status)

        # Connect MainWindow UI signals to the appropriate sub-controller slots
        # (These connections might be better placed in MainWindow's setup, but keep here for now)
        self.main_window.run_button.clicked.connect(self.search_orchestrator.start_search_process)
        self.main_window.cancel_button.clicked.connect(self.cancel_current_operation) # Keep cancel here
        self.main_window.gemini_fetch_button.clicked.connect(self.model_fetcher_manager.fetch_gemini_models)
        self.main_window.openrouter_fetch_button.clicked.connect(self.model_fetcher_manager.fetch_openrouter_models)
        self.main_window.clear_cache_button.clicked.connect(self.clear_cache) # Keep cache clear here

        # Worker instance for scraping (managed directly by controller for now)
        self.scrape_worker = None

    def log_status(self, message):
        """Helper to call MainWindow's log_status."""
        # Keep this simple helper
        if self.main_window:
            self.main_window.log_status(message)

    # --- New Slots to Handle Signals from Sub-Controllers ---

    def _handle_search_complete(self, report_path, final_answer_content, toc_tree_nodes):
        """Handles successful search completion signal from SearchOrchestrator."""
        self.log_status("GuiController received search complete signal.")
        # Delegate display to ResultDisplayManager
        self.result_display_manager.display_results_and_toc(report_path, final_answer_content, toc_tree_nodes)

    def _handle_search_error(self, error_message):
        """Handles search error signal from SearchOrchestrator."""
        self.log_status(f"GuiController received search error signal: {error_message}")
        QMessageBox.critical(self.main_window, "Search Error", error_message)
        # Optionally clear results or update status label via ResultDisplayManager
        self.result_display_manager.clear_results() # Example: clear results on error
        self.main_window.report_path_label.setText("Search failed.")


    def _handle_refinement_request(self, anchor_id, instruction):
        """Handles refinement request signal from ResultDisplayManager."""
        self.log_status(f"GuiController received refinement request for anchor: {anchor_id}")
        # Get current HTML from display manager before starting refinement
        current_html = self.result_display_manager.current_report_html
        if not current_html:
            self.log_status("[Error] Cannot start refinement: No current report HTML available.")
            QMessageBox.warning(self.main_window, "Refinement Error", "Cannot refine section, no report content loaded.")
            return
        # Delegate to RefinementController
        self.refinement_controller.start_refinement_process(anchor_id, instruction, current_html)

    def _handle_refinement_complete(self, anchor_id, refined_content):
        """Handles refinement complete signal from RefinementController."""
        self.log_status(f"GuiController received refinement complete for anchor: {anchor_id}")
        # Delegate updating the display to ResultDisplayManager
        self.result_display_manager.update_refined_section(anchor_id, refined_content)

    def _handle_refinement_error(self, anchor_id, error_message):
        """Handles refinement error signal from RefinementController."""
        # Error message box is already shown by RefinementController, just log here.
        self.log_status(f"GuiController received refinement error for anchor '{anchor_id}': {error_message}")
        # No further action needed here usually, UI reset handled by RefinementController

    # --- Keep Cache Management Here (for now) ---
    def clear_cache(self):
        """Clears the cache database."""
        # Get cache path from config
        # Reload config data in case path changed
        self.config_data = load_config(self.config_path)
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


    # --- Updated Cancellation ---
    def cancel_current_operation(self):
        """Requests cancellation of the currently running operation (delegates)."""
        self.log_status("Cancel requested by user.")
        # Primarily delegate to SearchOrchestrator as it handles the main cancellable task
        cancelled = self.search_orchestrator.cancel_current_operation()
        if not cancelled:
            # Optionally check other controllers if they have cancellable operations
            # e.g., if refinement could be cancelled:
            # cancelled = self.refinement_controller.cancel_refinement()
            self.log_status("No active cancellable operation found to cancel.")


    # --- Updated Cleanup ---
    def shutdown_workers(self):
        """Stop all running worker threads by delegating to sub-controllers."""
        self.log_status("Shutting down controller and all sub-component workers...")
        self.search_orchestrator.shutdown_workers()
        self.refinement_controller.shutdown_worker()
        self.model_fetcher_manager.shutdown_workers()
        # Add shutdown for ResultDisplayManager if it ever gets workers
        if self.scrape_worker and self.scrape_worker.isRunning():
            self.log_status("Stopping active scrape worker...")
            self.scrape_worker.stop() # Assuming worker has a stop method
            self.scrape_worker.wait() # Wait for it to finish cleanly
            self.log_status("Scrape worker stopped.")
        self.log_status("All worker shutdown routines called.")

    # --- Scraping Dialog and Logic ---

    def show_scrape_dialog(self):
        """Shows the dialog to get URL and options for scraping."""
        if self.scrape_worker and self.scrape_worker.isRunning():
             QMessageBox.warning(self.main_window, "Busy", "A scraping process is already running.")
             return

        dialog = ScrapeDialog(self.main_window)
        if dialog.exec():
            values = dialog.get_values()
            if values:
                self.log_status(f"Starting scrape for URL: {values['url']} with depth {values['depth']}")
                # Pass the depth value to the worker starter
                self._start_scrape_worker(values['url'], values['ignore_robots'], values['depth'])

    def _start_scrape_worker(self, url, ignore_robots, depth): # Added depth parameter
        """Initializes and starts the ScrapeWorker."""
        # Disable scrape menu item? Or handle busy state check in show_scrape_dialog
        self.main_window.progress_bar.setRange(0, 0) # Indeterminate progress
        self.main_window.progress_bar.setVisible(True)
        self.main_window.run_button.setEnabled(False) # Disable run during scrape
        # self.main_window.cancel_button.setEnabled(True) # Enable cancel? Need cancel logic in worker

        # Reload config to get current embedding settings
        self.config_data = load_config(self.config_path)
        embedding_model = self.config_data.get('retrieval', {}).get('embedding_model', DEFAULT_CONFIG['retrieval']['embedding_model'])
        device = self.config_data.get('general', {}).get('device', DEFAULT_CONFIG['general']['device'])

        # Pass depth to the ScrapeWorker constructor
        self.scrape_worker = ScrapeWorker(url, ignore_robots, depth, embedding_model, device)
        self.scrape_worker.status_update.connect(self.log_status)
        self.scrape_worker.scrape_complete.connect(self._handle_scrape_complete)
        self.scrape_worker.scrape_error.connect(self._handle_scrape_error)
        self.scrape_worker.finished.connect(self._scrape_worker_finished) # Generic finished signal
        self.scrape_worker.start()

    def _handle_scrape_complete(self, url, content_snippet):
        """Handles successful completion of the scraping process."""
        self.log_status(f"Successfully scraped and added content from {url} to knowledge base.")
        QMessageBox.information(self.main_window, "Scrape Complete",
                                f"Successfully scraped and added content from:\n{url}\n\nSnippet:\n{content_snippet}...")
        # UI reset is handled in _scrape_worker_finished

    def _handle_scrape_error(self, url, error_message):
        """Handles errors during the scraping process."""
        self.log_status(f"[ERROR] Scraping failed for {url}: {error_message}")
        QMessageBox.critical(self.main_window, "Scrape Error",
                             f"Failed to scrape content from:\n{url}\n\nError: {error_message}")
        # UI reset is handled in _scrape_worker_finished

    def _scrape_worker_finished(self):
        """Resets UI elements after the scrape worker finishes (success or error)."""
        self.log_status("Scrape worker finished.")
        self.main_window.progress_bar.setVisible(False)
        self.main_window.run_button.setEnabled(True)
        # self.main_window.cancel_button.setEnabled(False)
        # self.main_window.cancel_button.setVisible(False)
        self.scrape_worker = None # Clear worker instance


# --- REMOVED ALL THE MOVED METHODS ---
# (Query Enhancement, Topic Extraction, Model Fetching, Search Execution, Result Handling, Refinement Logic)
# The SEARCH block should cover the removal of these methods implicitly by replacing the large block.
