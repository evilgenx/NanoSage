#!/usr/bin/env python3
# gui/controller.py

import logging # <<< Keep only one import
import os
import re # Keep re for refinement content extraction if needed here, or move if fully delegated
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QModelIndex
from PyQt6.QtWidgets import QMessageBox, QMenu, QInputDialog, QDialog # Added QDialog

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
    from .config_editor_dialog import ConfigEditorDialog # <<< Import the new config dialog
except ImportError as e:
    # Use logger here if possible, but it might fail early
    logging.error(f"Error importing sub-controllers/dialogs in controller.py: {e}", exc_info=True) # <<< Use logger
    import sys
    sys.exit(1)

logger = logging.getLogger(__name__) # <<< Get logger

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
        # Connect orchestrator signals for progress UI management
        self.search_orchestrator.search_started.connect(self.main_window.reset_progress_ui) # Reset UI on start
        # Assuming SearchOrchestrator relays the structured signal from the worker
        self.search_orchestrator.structured_progress_update.connect(self.main_window.handle_structured_progress)
        # Connect completion/error signals to finalize the progress UI
        self.search_orchestrator.search_process_complete.connect(lambda: self.main_window.finalize_progress_ui(success=True))
        self.search_orchestrator.search_process_error.connect(lambda: self.main_window.finalize_progress_ui(success=False))
        # Remove direct TOC signal connections, handled by handle_structured_progress now
        # self.search_orchestrator.toc_node_added.connect(self.main_window._on_toc_node_added)
        # self.search_orchestrator.toc_node_updated.connect(self.main_window._on_toc_node_updated)


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
        # Config editor dialog instance (create when needed)
        self.config_editor_dialog = None # <<< Added instance variable

    def log_status(self, message):
        """Helper to call MainWindow's log_status."""
        # Keep this simple helper
        if self.main_window:
            # Log to main window status area AND logger
            logger.info(f"Status Update: {message}") # <<< Use logger
            self.main_window.log_status(message)

    # --- New Slots to Handle Signals from Sub-Controllers ---

    def _handle_search_complete(self, report_path, final_answer_content, toc_tree_nodes):
        """Handles successful search completion signal from SearchOrchestrator."""
        logger.info("GuiController received search complete signal.") # <<< Use logger
        # Delegate display to ResultDisplayManager
        self.result_display_manager.display_results_and_toc(report_path, final_answer_content, toc_tree_nodes)
        # Finalize UI is now handled by direct signal connection

    def _handle_search_error(self, error_message):
        """Handles search error signal from SearchOrchestrator."""
        logger.error(f"GuiController received search error signal: {error_message}") # <<< Use logger
        # QMessageBox is now shown by handle_structured_progress for 'error' type
        # QMessageBox.critical(self.main_window, "Search Error", error_message)
        # Optionally clear results or update status label via ResultDisplayManager
        self.result_display_manager.clear_results() # Example: clear results on error
        self.main_window.report_path_label.setText("Search failed.")
        # Finalize UI is now handled by direct signal connection


    def _handle_refinement_request(self, anchor_id, instruction):
        """Handles refinement request signal from ResultDisplayManager."""
        logger.info(f"GuiController received refinement request for anchor: {anchor_id}") # <<< Use logger
        # Get current HTML from display manager before starting refinement
        current_html = self.result_display_manager.current_report_html
        if not current_html:
            logger.error("Cannot start refinement: No current report HTML available.") # <<< Use logger
            QMessageBox.warning(self.main_window, "Refinement Error", "Cannot refine section, no report content loaded.")
            return
        # Delegate to RefinementController
        self.refinement_controller.start_refinement_process(anchor_id, instruction, current_html)

    def _handle_refinement_complete(self, anchor_id, refined_content):
        """Handles refinement complete signal from RefinementController."""
        logger.info(f"GuiController received refinement complete for anchor: {anchor_id}") # <<< Use logger
        # Delegate updating the display to ResultDisplayManager
        self.result_display_manager.update_refined_section(anchor_id, refined_content)

    def _handle_refinement_error(self, anchor_id, error_message):
        """Handles refinement error signal from RefinementController."""
        # Error message box is already shown by RefinementController, just log here.
        logger.error(f"GuiController received refinement error for anchor '{anchor_id}': {error_message}") # <<< Use logger
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
            logger.info(f"Attempting to clear cache: {cache_db_path}") # <<< Use logger
            try:
                # Use a temporary CacheManager instance just for clearing
                temp_cache_manager = CacheManager(cache_db_path)
                temp_cache_manager.clear_all_cache() # This deletes and recreates
                temp_cache_manager.close() # Close the connection
                self.log_status("Cache cleared successfully.")
                logger.info("Cache cleared successfully.") # <<< Use logger
                QMessageBox.information(self.main_window, "Cache Cleared", "The cache database has been cleared.")
            except Exception as e:
                error_msg = f"Failed to clear cache: {e}"
                self.log_status(f"[ERROR] {error_msg}")
                logger.exception("Error during cache clearing") # <<< Use logger.exception
                QMessageBox.critical(self.main_window, "Cache Error", error_msg)


    # --- Updated Cancellation ---
    def cancel_current_operation(self):
        """Requests cancellation of the currently running operation (delegates)."""
        self.log_status("Cancel requested by user.")
        logger.info("Cancel requested by user.") # <<< Use logger
        # Primarily delegate to SearchOrchestrator as it handles the main cancellable task
        cancelled = self.search_orchestrator.cancel_current_operation()
        if not cancelled:
            # Optionally check other controllers if they have cancellable operations
            # e.g., if refinement could be cancelled:
            # cancelled = self.refinement_controller.cancel_refinement()
            self.log_status("No active cancellable operation found to cancel.")
            logger.info("No active cancellable operation found to cancel.") # <<< Use logger


    # --- Updated Cleanup ---
    def shutdown_workers(self):
        """Stop all running worker threads by delegating to sub-controllers."""
        self.log_status("Shutting down controller and all sub-component workers...")
        logger.info("Shutting down controller and all sub-component workers...") # <<< Use logger
        self.search_orchestrator.shutdown_workers()
        self.refinement_controller.shutdown_worker()
        self.model_fetcher_manager.shutdown_workers()
        # Add shutdown for ResultDisplayManager if it ever gets workers
        if self.scrape_worker and self.scrape_worker.isRunning():
            self.log_status("Stopping active scrape worker...")
            logger.info("Stopping active scrape worker...") # <<< Use logger
            self.scrape_worker.stop() # Assuming worker has a stop method
            self.scrape_worker.wait() # Wait for it to finish cleanly
            self.log_status("Scrape worker stopped.")
            logger.info("Scrape worker stopped.") # <<< Use logger
        self.log_status("All worker shutdown routines called.")
        logger.info("All worker shutdown routines called.") # <<< Use logger

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
                logger.info(f"Starting scrape for URL: {values['url']} with depth {values['depth']}") # <<< Use logger
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
        logger.info(f"Successfully scraped and added content from {url} to knowledge base.") # <<< Use logger
        QMessageBox.information(self.main_window, "Scrape Complete",
                                f"Successfully scraped and added content from:\n{url}\n\nSnippet:\n{content_snippet}...")
        # UI reset is handled in _scrape_worker_finished

    def _handle_scrape_error(self, url, error_message):
        """Handles errors during the scraping process."""
        self.log_status(f"[ERROR] Scraping failed for {url}: {error_message}")
        logger.error(f"Scraping failed for {url}: {error_message}") # <<< Use logger
        QMessageBox.critical(self.main_window, "Scrape Error",
                             f"Failed to scrape content from:\n{url}\n\nError: {error_message}")
        # UI reset is handled in _scrape_worker_finished

    def _scrape_worker_finished(self):
        """Resets UI elements after the scrape worker finishes (success or error)."""
        self.log_status("Scrape worker finished.")
        logger.info("Scrape worker finished.") # <<< Use logger
        self.main_window.progress_bar.setVisible(False)
        self.main_window.run_button.setEnabled(True)
        # self.main_window.cancel_button.setEnabled(False)
        # self.main_window.cancel_button.setVisible(False)
        self.scrape_worker = None # Clear worker instance

    # --- Config Editor Logic --- # <<< Added section
    def show_config_editor(self):
        """Creates and shows the configuration editor dialog."""
        # Avoid opening multiple dialogs
        if self.config_editor_dialog and self.config_editor_dialog.isVisible():
            self.config_editor_dialog.raise_()
            self.config_editor_dialog.activateWindow()
            return

        self.log_status("Opening configuration editor...")
        logger.info("Opening configuration editor...") # <<< Use logger
        # Pass the main window as parent
        self.config_editor_dialog = ConfigEditorDialog(config_path=self.config_path, parent=self.main_window)
        # Execute the dialog modally
        result = self.config_editor_dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.log_status("Configuration changes saved.")
            logger.info("Configuration changes saved via editor.") # <<< Use logger
            # Reload config data in controller and main window
            self.config_data = load_config(self.config_path)
            self.main_window.config_data = self.config_data
            # Apply changes to the main window UI
            self._apply_saved_config()
        else:
            self.log_status("Configuration editing cancelled.")
            logger.info("Configuration editing cancelled.") # <<< Use logger

        # Clean up dialog reference
        self.config_editor_dialog = None

    def _apply_saved_config(self):
        """Updates the main window UI elements based on the currently loaded self.config_data."""
        self.log_status("Applying saved configuration to main window UI...")
        logger.info("Applying saved configuration to main window UI...") # <<< Use logger
        try:
            # Reload values into relevant main window widgets
            # This mirrors the logic in MainWindow.__init__ for setting initial states

            # General Tab Widgets
            general_cfg = self.config_data.get('general', {})
            self.main_window.web_search_checkbox.setChecked(general_cfg.get('web_search', True))
            self.main_window.max_depth_spinbox.setValue(general_cfg.get('max_depth', 1))
            self.main_window.device_combo.setCurrentText(general_cfg.get('device', 'cpu'))
            self.main_window.corpus_dir_label.setText(general_cfg.get('corpus_dir', '') or '')

            # Retrieval Tab Widgets
            retrieval_cfg = self.config_data.get('retrieval', {})
            self.main_window.top_k_spinbox.setValue(retrieval_cfg.get('top_k', 3))
            # Handle embedding model change carefully, might need to trigger device change handler
            current_embedding_model = retrieval_cfg.get('embedding_model', 'colpali')
            if self.main_window.embedding_model_combo.currentText() != current_embedding_model:
                 # Check if the model exists in the current list before setting
                 if self.main_window.embedding_model_combo.findText(current_embedding_model) != -1:
                     self.main_window.embedding_model_combo.setCurrentText(current_embedding_model)
                 else:
                     # If the saved model isn't valid for the current device, log warning
                     warn_msg = f"Saved embedding model '{current_embedding_model}' not available for device '{general_cfg.get('device', 'cpu')}'. UI might not reflect saved value."
                     self.log_status(f"[Warning] {warn_msg}")
                     logger.warning(warn_msg) # <<< Use logger

            # Search Tab Widgets
            search_cfg = self.config_data.get('search', {})
            provider = search_cfg.get('provider', 'duckduckgo')
            self.main_window.search_provider_combo.setCurrentText("DuckDuckGo" if provider == 'duckduckgo' else "SearXNG")
            self.main_window.iterative_search_checkbox.setChecked(search_cfg.get('enable_iterative_search', False))
            self.main_window.include_visuals_checkbox.setChecked(search_cfg.get('include_visuals', False))
            # SearXNG specific fields
            searxng_cfg = search_cfg.get('searxng', {})
            self.main_window.searxng_base_url_input.setText(searxng_cfg.get('base_url', ''))
            self.main_window.searxng_time_range_input.setText(searxng_cfg.get('time_range', '') or '')
            self.main_window.searxng_categories_input.setText(searxng_cfg.get('categories', '') or '')
            # Update SearXNG engine selector (if needed, though it reads config on init)
            engines = searxng_cfg.get('engines', [])
            if isinstance(engines, list):
                self.main_window.searxng_engine_selector.setSelectedEngines(engines)
            # Trigger visibility update
            self.main_window.handle_search_provider_change(self.main_window.search_provider_combo.currentText())

            # RAG Tab Widgets
            llm_cfg = self.config_data.get('llm', {})
            rag_type = llm_cfg.get('rag_model_type', 'gemma')
            self.main_window.rag_model_combo.setCurrentText(rag_type)
            self.main_window.personality_input.setText(llm_cfg.get('rag_personality', ''))
            # Trigger RAG model change handler to update visibility and potentially model lists
            self.main_window.handle_rag_model_change(rag_type)
            # Note: We don't automatically re-fetch/re-select Gemini/OpenRouter models here,
            # as the user might need to fetch them again if the API key changed.
            # The UI state will reflect the saved *type*, but model selection might be cleared.

            # Cache Tab Widget
            cache_cfg = self.config_data.get('cache', {})
            self.main_window.cache_enabled_checkbox.setChecked(cache_cfg.get('enabled', False))

            self.log_status("Main window UI updated with saved configuration.")
            logger.info("Main window UI updated with saved configuration.") # <<< Use logger

        except Exception as e:
            err_msg = f"Failed to apply saved configuration to UI: {e}"
            self.log_status(f"[ERROR] {err_msg}")
            logger.exception("Error applying saved config") # <<< Use logger.exception
            QMessageBox.warning(self.main_window, "UI Update Error", f"Could not fully update the main window UI with the saved settings: {e}")
