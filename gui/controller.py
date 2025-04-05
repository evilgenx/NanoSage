#!/usr/bin/env python3
# gui/controller.py

import logging
import os # Added for path manipulation in on_search_complete
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QModelIndex # Added Qt, QModelIndex
from PyQt6.QtWidgets import QMessageBox, QMenu, QInputDialog # Added QMenu, QInputDialog
from PyQt6.QtGui import QStandardItemModel, QStandardItem # Added QStandardItemModel, QStandardItem

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
        self.refinement_worker = None # Add worker instance for refinement
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
    def on_search_complete(self, report_path, final_answer_content, toc_tree_nodes):
        """
        Handle successful search completion.
        Receives report path, the final report content (with anchors), and TOC nodes.
        """
        self.log_status(f"Search finished successfully!")
        # Update MainWindow state
        self.main_window.current_report_path = report_path
        self.main_window.current_results_dir = os.path.dirname(report_path)
        self.main_window.report_path_label.setText(report_path)
        self.main_window.current_results_dir = os.path.dirname(report_path)
        self.main_window.report_path_label.setText(report_path)
        self.main_window.open_report_button.setEnabled(True)
        self.main_window.open_folder_button.setEnabled(True)
        self.main_window.share_email_button.setEnabled(True)

        # --- Display Results and Populate TOC ---
        self.log_status("Displaying report content and populating Table of Contents...")
        # 1. Display report content (supports Markdown)
        self.main_window.results_text_edit.setMarkdown(final_answer_content)

        # 2. Populate TOC Tree
        self._populate_toc_tree(toc_tree_nodes)

        # 3. Connect TOC click signal (disconnect first if already connected)
        try:
            self.main_window.toc_tree_widget.clicked.disconnect()
        except TypeError:
            pass # Signal not connected
        self.main_window.toc_tree_widget.clicked.connect(self._handle_toc_click)

        # 4. Setup Context Menu for TOC (disconnect first if already connected)
        self.main_window.toc_tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        try:
             self.main_window.toc_tree_widget.customContextMenuRequested.disconnect()
        except TypeError:
             pass # Signal not connected
        self.main_window.toc_tree_widget.customContextMenuRequested.connect(self._handle_toc_context_menu)

        self.log_status("Results displayed.")

    def _populate_toc_tree(self, toc_nodes):
        """Recursively populates the QTreeView model from TOCNode data."""
        model = self.main_window.toc_tree_widget.model()
        if not isinstance(model, QStandardItemModel):
            # Should have been set in ui_setup, but check just in case
            model = QStandardItemModel()
            self.main_window.toc_tree_widget.setModel(model)
        model.clear() # Clear previous TOC

        def add_items(parent_item, nodes):
            for node in nodes:
                # Create item with the query text
                item = QStandardItem(node.query_text)
                # Store the anchor ID in the item's data (using UserRole)
                item.setData(node.anchor_id, Qt.ItemDataRole.UserRole)
                item.setEditable(False) # Make items non-editable
                # Add tooltip if needed (e.g., show summary)
                # item.setToolTip(f"Anchor: {node.anchor_id}\nSummary: {node.summary[:100]}...")
                parent_item.appendRow(item)
                if node.children:
                    add_items(item, node.children) # Recursively add children

        add_items(model.invisibleRootItem(), toc_nodes)
        # Optionally expand top-level items
        # self.main_window.toc_tree_widget.expandToDepth(0)

    def _handle_toc_click(self, index: QModelIndex):
        """Scrolls the results view to the anchor associated with the clicked TOC item."""
        if not index.isValid():
            return
        item = self.main_window.toc_tree_widget.model().itemFromIndex(index)
        if item:
            anchor_id = item.data(Qt.ItemDataRole.UserRole)
            if anchor_id:
                self.log_status(f"Navigating to anchor: {anchor_id}")
                self.main_window.results_text_edit.scrollToAnchor(anchor_id)
            else:
                self.log_status(f"No anchor ID found for TOC item: {item.text()}")

    def _handle_toc_context_menu(self, position):
        """Shows a context menu for the TOC tree."""
        index = self.main_window.toc_tree_widget.indexAt(position)
        if not index.isValid():
            return # Clicked outside an item

        item = self.main_window.toc_tree_widget.model().itemFromIndex(index)
        anchor_id = item.data(Qt.ItemDataRole.UserRole) if item else None

        if not anchor_id:
            return # No anchor associated, nothing to refine

        menu = QMenu()
        refine_action = menu.addAction("Refine Section...")
        action = menu.exec(self.main_window.toc_tree_widget.viewport().mapToGlobal(position))

        if action == refine_action:
            self._handle_refine_request(anchor_id)

    def _handle_refine_request(self, anchor_id):
        """Handles the 'Refine Section' action from the TOC context menu."""
        self.log_status(f"Refine requested for section with anchor: {anchor_id}")

        # --- TODO: Implement Refinement Logic ---
        # 1. Extract Section Content:
        #    - Find the text in self.main_window.results_text_edit between
        #      the anchor `anchor_id` and the *next* anchor in the document,
        #      or the end of the document. This requires careful parsing.
        #    - This might be complex with Markdown/HTML. A simpler approach
        #      might be needed initially, or pass the raw markdown segment.
        section_content = self._extract_section_content(anchor_id) # Placeholder for extraction logic
        if not section_content:
             self.log_status(f"[Error] Could not extract content for anchor {anchor_id}.")
             QMessageBox.warning(self.main_window, "Refinement Error", "Could not extract the content for the selected section.")
             return

        # 2. Get User Instructions:
        instruction, ok = QInputDialog.getText(self.main_window, "Refine Section",
                                               "How should this section be refined?\n(e.g., 'Summarize this', 'Make it simpler', 'Add bullet points')")
        if not ok or not instruction.strip():
            self.log_status("Refinement cancelled by user.")
            return

        # 3. Call LLM for Refinement using RefinementWorker:
        if self.refinement_worker and self.refinement_worker.isRunning():
            self.log_status("[Warning] Refinement is already in progress.")
            QMessageBox.warning(self.main_window, "Busy", "Another refinement operation is already running.")
            return

        # Prepare LLM config (use current RAG settings)
        rag_model_type = self.main_window.rag_model_combo.currentText()
        selected_generative_model = None
        if rag_model_type == "gemini":
            selected_generative_model = self.main_window.gemini_model_combo.currentText()
        elif rag_model_type == "openrouter":
            selected_generative_model = self.main_window.openrouter_model_combo.currentText()
        # Add gemma/pali if needed

        if rag_model_type == "None":
             QMessageBox.warning(self.main_window, "Refinement Error", "Please select a RAG model (Gemini, OpenRouter, Gemma) to perform refinement.")
             return
        if (rag_model_type in ["gemini", "openrouter"]) and not selected_generative_model:
             QMessageBox.warning(self.main_window, "Refinement Error", f"Please select a specific model for the '{rag_model_type}' provider.")
             return

        # Reload config for potential API key updates
        self.config_data = load_config(self.config_path)
        llm_config = {
            "provider": rag_model_type,
            "model_id": selected_generative_model, # Will be None for gemma/pali, handled in task
            "api_key": self.config_data.get('api_keys', {}).get(f'{rag_model_type}_api_key'),
            "personality": self.main_window.personality_input.text() or None
        }

        self.log_status(f"Starting refinement worker for anchor '{anchor_id}' with instruction: '{instruction}'")
        # Show progress indication (optional, could use progress bar)
        self.main_window.progress_bar.setVisible(True)
        self.main_window.progress_bar.setRange(0, 0) # Indeterminate

        self.refinement_worker = RefinementWorker(
            anchor_id=anchor_id,
            section_content=section_content, # Pass the extracted HTML segment
            instruction=instruction,
            llm_config=llm_config,
            parent=self.main_window # Set parent
        )
        self.refinement_worker.status_update.connect(self.log_status)
        self.refinement_worker.refinement_complete.connect(self._on_refinement_complete)
        self.refinement_worker.refinement_error.connect(self._on_refinement_error)
        self.refinement_worker.finished.connect(self._on_refinement_finished)
        self.refinement_worker.start()

        # 4. Update Results View (Handled in _on_refinement_complete slot):
        #    - On success, receive the `anchor_id` and `refined_content`.
        #    - Replace the original `section_content` in `results_text_edit`
        #      with `refined_content`, keeping the anchor intact. This requires
        #      careful text manipulation using the anchor_id.

    def _on_refinement_complete(self, anchor_id, refined_content):
        """Handles successful refinement completion."""
        self.log_status(f"Refinement successful for anchor '{anchor_id}'. Updating display...")

        # --- Replace Content Logic ---
        # This is complex and potentially fragile. We need to replace the *original*
        # HTML segment associated with anchor_id with the new refined_content.
        # The refined_content likely won't have the anchor tag, so we need to prepend it.

        # 1. Get the full current HTML
        current_full_html = self.main_window.results_text_edit.toHtml()

        # 2. Find the original segment again (using the same logic as extraction)
        #    We need the start and end points of the original segment to replace it.
        start_pattern = re.compile(r'(<a\s+name="' + re.escape(anchor_id) + r'"\s*>\s*</a>)', re.IGNORECASE)
        start_match = start_pattern.search(current_full_html)

        if not start_match:
            self.log_status(f"[Error] Cannot update display: Original start anchor tag not found for {anchor_id} during replacement.")
            QMessageBox.critical(self.main_window, "Update Error", f"Failed to find the original section for '{anchor_id}' to update the display.")
            return

        original_anchor_tag = start_match.group(1)
        start_index_after_anchor = start_match.end()

        # Find the next anchor tag
        next_anchor_pattern = re.compile(r'<a\s+name=".*?"\s*>\s*</a>', re.IGNORECASE)
        next_match = next_anchor_pattern.search(current_full_html, pos=start_index_after_anchor)

        start_replace_index = start_match.start() # Start replacing from the original anchor tag
        if next_match:
            end_replace_index = next_match.start() # Replace up to the next anchor tag
        else:
            # If no next anchor, replace until the end of the document
            # This might be problematic if the original content didn't go to the end.
            # A safer approach might be needed, perhaps storing the original extracted segment.
            # For now, assume replacement goes to the end if no next anchor.
            end_replace_index = len(current_full_html)

        # 3. Construct the replacement HTML (Original Anchor + Refined Content)
        #    Assume refined_content is HTML/Markdown compatible with QTextEdit
        replacement_html = original_anchor_tag + "\n" + refined_content # Add newline for spacing

        # 4. Create the new full HTML
        new_full_html = current_full_html[:start_replace_index] + replacement_html + current_full_html[end_replace_index:]

        # 5. Set the new HTML content
        self.main_window.results_text_edit.setHtml(new_full_html)
        self.log_status(f"Display updated for refined section '{anchor_id}'.")

    def _on_refinement_error(self, anchor_id, error_message):
        """Handles refinement errors."""
        self.log_status(f"[Error] Refinement failed for anchor '{anchor_id}': {error_message}")
        QMessageBox.critical(self.main_window, "Refinement Error", f"Failed to refine section '{anchor_id}':\n{error_message}")

    def _on_refinement_finished(self):
        """Called when the RefinementWorker finishes (success or error)."""
        self.log_status("Refinement worker finished.")
        self.refinement_worker = None # Clear worker reference
        # Hide progress bar only if no other major worker is running
        if not (self.search_worker and self.search_worker.isRunning()):
             self.main_window.progress_bar.setVisible(False)
             self.main_window.progress_bar.setRange(0, 100) # Reset


    def _extract_section_content(self, anchor_id):
        """
        Placeholder: Extracts HTML content associated with an anchor.
        Attempts to find the content between the specified anchor and the next anchor tag.
        Returns the HTML fragment (including the starting anchor) or None if extraction fails.
        NOTE: This relies on the LLM report consistently using <a name="..."> tags.
        """
        self.log_status(f"Attempting to extract content for anchor: {anchor_id}")
        full_html = self.main_window.results_text_edit.toHtml()

        # Pattern to find the starting anchor tag specifically
        # Using non-greedy match .*? to avoid issues if anchor_id appears elsewhere
        start_pattern = re.compile(r'(<a\s+name="' + re.escape(anchor_id) + r'"\s*>\s*</a>)', re.IGNORECASE)
        start_match = start_pattern.search(full_html)

        if not start_match:
            self.log_status(f"[Error] Start anchor tag not found for {anchor_id}")
            return None

        start_index = start_match.end() # Position *after* the starting anchor tag

        # Pattern to find the *next* <a name="..."> tag after the starting position
        # Using non-greedy match .*?
        next_anchor_pattern = re.compile(r'<a\s+name=".*?"\s*>\s*</a>', re.IGNORECASE)
        next_match = next_anchor_pattern.search(full_html, pos=start_index)

        if next_match:
            end_index = next_match.start() # Position *before* the next anchor tag
            self.log_status(f"Found next anchor at index {end_index}")
            # Return the content including the starting anchor tag itself up to the next anchor
            section_html = start_match.group(1) + full_html[start_index:end_index]
        else:
            # If no next anchor, take content from start anchor to the end of the document
            # (Need to be careful about closing HTML tags - this might grab too much/invalid HTML)
            self.log_status("No subsequent anchor found, taking content to the end.")
            # Let's try taking from the start match's start index to the end
            section_html = full_html[start_match.start():]
            # This is still risky. A better approach might involve parsing the HTML structure,
            # but for now, we return the segment including the start anchor.

        self.log_status(f"Extracted HTML segment length for {anchor_id}: {len(section_html)}")
        # print(f"[DEBUG] Extracted HTML for {anchor_id}:\n{section_html[:500]}...") # Optional debug print
        return section_html # Return the HTML fragment


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
