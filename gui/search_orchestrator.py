#!/usr/bin/env python3
# gui/search_orchestrator.py

import logging
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

# Import worker threads using relative import
try:
    from .workers import (
        SearchWorker, TopicExtractorWorker, QueryEnhancerWorker
    )
except ImportError as e:
    print(f"Error importing workers in search_orchestrator.py: {e}")
    import sys
    sys.exit(1)

# Import config loading utility
from config_utils import load_config, save_config

class SearchOrchestrator(QObject):
    """
    Manages the search workflow, including pre-processing (topic extraction,
    query enhancement) and the main SearchWorker.
    """
    # Signal to pass completion data (report path, content, toc) upwards
    search_process_complete = pyqtSignal(str, str, list)
    # Signal to pass errors upwards
    search_process_error = pyqtSignal(str)
    # Signal for status updates
    status_update = pyqtSignal(str)

    def __init__(self, main_window, result_display_manager, config_path, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.result_display_manager = result_display_manager # To call display methods
        self.config_path = config_path
        self.config_data = load_config(config_path) # Load initial config

        # Worker instances managed by this orchestrator
        self.search_worker = None
        self.topic_extractor_worker = None
        self.query_enhancer_worker = None

    def log_status(self, message):
        """Emit status update signal."""
        self.status_update.emit(message)

    # --- Query Enhancement Slots ---
    def on_enhanced_query_ready(self, enhanced_query_preview):
        """Handles the signal when the enhanced query preview is ready."""
        self.log_status(f"Enhanced Query Preview: {enhanced_query_preview}")
        # IMPORTANT: Proceed with the search using the ORIGINAL query text
        original_query = self.main_window.query_input.toPlainText().strip()

        if not original_query:
             self.log_status("[Error] Original query is empty after enhancement preview. Aborting search.")
             self.on_enhancement_error("Original query became empty.")
             return

        self.log_status("Proceeding with search using the original query...")
        self._start_main_search_worker(original_query)

    def on_enhancement_error(self, error_message):
        """Handles errors during query enhancement preview."""
        self.log_status(f"Query Enhancement Preview Error: {error_message}")
        QMessageBox.critical(self.main_window, "Query Enhancement Error", error_message)
        self._reset_ui_after_preprocessing_failure()
        self.query_enhancer_worker = None

    def on_enhancement_finished(self):
        """Called when the query enhancement thread finishes."""
        if self.query_enhancer_worker:
            self.query_enhancer_worker = None
            # UI reset should happen in success/error slots

    # --- Topic Extraction Slots ---
    def on_topics_extracted(self, topics_string):
        """Handles the signal when topics are successfully extracted."""
        self.log_status("Topics extracted successfully.")
        self.main_window.query_input.setPlainText(topics_string)
        self.main_window.extract_topics_checkbox.setChecked(False)
        self._reset_ui_after_preprocessing_success()
        self.topic_extractor_worker = None
        QMessageBox.information(self.main_window, "Topics Extracted", "Extracted topics placed in query box. Review/edit, then click 'Run Search' again.")

    def on_topic_extraction_error(self, error_message):
        """Handles errors during topic extraction."""
        self.log_status(f"Topic Extraction Error: {error_message}")
        QMessageBox.critical(self.main_window, "Topic Extraction Error", error_message)
        self._reset_ui_after_preprocessing_failure()
        self.topic_extractor_worker = None

    def on_topic_extraction_finished(self):
        """Called when the topic extraction thread finishes."""
        if self.topic_extractor_worker:
            self.topic_extractor_worker = None
            # UI reset should happen in success/error slots

    # --- Search Execution ---
    def start_search_process(self):
        """
        Public method called by MainWindow/GuiController.
        Validates inputs and starts the appropriate pre-processing or main search worker.
        """
        # Check if any worker managed by this orchestrator is running
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

            self.config_data = load_config(self.config_path) # Reload config
            llm_config = {
                "provider": rag_model_type,
                "model_id": selected_generative_gemini or selected_generative_openrouter,
                "api_key": self.config_data.get('api_keys', {}).get(f'{rag_model_type}_api_key'),
                "personality": self.main_window.personality_input.text() or None
            }

        # --- Action based on Checkbox State ---
        if self.main_window.extract_topics_checkbox.isChecked():
            self.log_status("Topic extraction requested...")
            self._start_topic_extractor(query_or_text, llm_config)
        else:
            self.log_status("Starting query enhancement preview...")
            self._start_query_enhancer(query_or_text, llm_config)

    def _start_topic_extractor(self, text_to_extract, llm_config):
        """Starts the TopicExtractorWorker."""
        self.log_status("Starting topic extraction worker...")
        self._set_ui_for_preprocessing()

        self.topic_extractor_worker = TopicExtractorWorker(text_to_extract, llm_config, self.main_window)
        self.topic_extractor_worker.status_update.connect(self.log_status)
        self.topic_extractor_worker.topics_extracted.connect(self.on_topics_extracted)
        self.topic_extractor_worker.error_occurred.connect(self.on_topic_extraction_error)
        self.topic_extractor_worker.finished.connect(self.on_topic_extraction_finished)
        self.topic_extractor_worker.start()

    def _start_query_enhancer(self, query_to_enhance, llm_config):
        """Starts the QueryEnhancerWorker."""
        self.log_status("Starting query enhancement preview worker...")
        self._set_ui_for_preprocessing()

        self.query_enhancer_worker = QueryEnhancerWorker(query_to_enhance, llm_config, self.main_window)
        self.query_enhancer_worker.status_update.connect(self.log_status)
        self.query_enhancer_worker.enhanced_query_ready.connect(self.on_enhanced_query_ready)
        self.query_enhancer_worker.enhancement_error.connect(self.on_enhancement_error)
        self.query_enhancer_worker.finished.connect(self.on_enhancement_finished)
        self.query_enhancer_worker.start()

    def _start_main_search_worker(self, query_to_use):
        """
        Internal method to prepare parameters and start the main SearchWorker.
        """
        self.log_status("Preparing to start main search worker...")

        # --- Standard Search Input Validation ---
        embedding_device = self.main_window.device_combo.currentText()
        embedding_model_name = self.main_window.embedding_model_combo.currentText()
        if not embedding_model_name:
             QMessageBox.warning(self.main_window, "Input Error", f"Please select an Embedding Model for the '{embedding_device}' device.")
             self._reset_ui_after_preprocessing_failure() # Reset UI as search won't start
             return

        rag_model_type = self.main_window.rag_model_combo.currentText()
        selected_generative_gemini = None
        selected_generative_openrouter = None
        if rag_model_type == "gemini":
            if self.main_window.gemini_model_combo.count() == 0 or not self.main_window.gemini_model_combo.currentText():
                 QMessageBox.warning(self.main_window, "Input Error", "RAG Model is Gemini, but no generative model selected. Please fetch and select one.")
                 self._reset_ui_after_preprocessing_failure()
                 return
            selected_generative_gemini = self.main_window.gemini_model_combo.currentText()
        elif rag_model_type == "openrouter":
            if self.main_window.openrouter_model_combo.count() == 0 or not self.main_window.openrouter_model_combo.currentText():
                 QMessageBox.warning(self.main_window, "Input Error", "RAG Model is OpenRouter, but no generative model selected. Please fetch and select one.")
                 self._reset_ui_after_preprocessing_failure()
                 return
            selected_generative_openrouter = self.main_window.openrouter_model_combo.currentText()

        # --- Save Current Config & Prepare Search Parameters ---
        self.config_data = load_config(self.config_path) # Reload fresh config
        # (Code to update config_data with UI settings - same as original controller)
        general_cfg = self.config_data.setdefault('general', {})
        general_cfg['web_search'] = self.main_window.web_search_checkbox.isChecked()
        general_cfg['max_depth'] = self.main_window.max_depth_spinbox.value()
        retrieval_cfg = self.config_data.setdefault('retrieval', {})
        retrieval_cfg['top_k'] = self.main_window.top_k_spinbox.value()
        retrieval_cfg['embedding_model'] = embedding_model_name
        embeddings_cfg = self.config_data.setdefault('embeddings', {})
        embeddings_cfg['device'] = embedding_device
        corpus_cfg = self.config_data.setdefault('corpus', {})
        corpus_cfg['path'] = self.main_window.corpus_dir_label.text() or None
        llm_cfg = self.config_data.setdefault('llm', {})
        llm_cfg['rag_model'] = rag_model_type if rag_model_type != "None" else None
        llm_cfg['personality'] = self.main_window.personality_input.text() or None
        llm_cfg['gemini_model_id'] = selected_generative_gemini
        llm_cfg['openrouter_model_id'] = selected_generative_openrouter
        llm_cfg['output_format'] = self.main_window.output_format_combo.currentText()
        cache_cfg = self.config_data.setdefault('cache', {})
        cache_cfg['enabled'] = self.main_window.cache_enabled_checkbox.isChecked()
        selected_provider_text = self.main_window.search_provider_combo.currentText()
        search_provider_key = 'duckduckgo' if selected_provider_text == "DuckDuckGo" else 'searxng'
        search_config = self.config_data.setdefault('search', {})
        search_config['provider'] = search_provider_key
        search_config['enable_iterative_search'] = self.main_window.iterative_search_checkbox.isChecked()
        searxng_config = search_config.setdefault('searxng', {})
        ddg_config = search_config.setdefault('duckduckgo', {})
        search_limit = 5
        searxng_url = None
        searxng_time_range = None
        searxng_categories = None
        searxng_engines = None
        if search_provider_key == 'searxng':
            searxng_url = self.main_window.searxng_base_url_input.text().strip() or None
            searxng_time_range = self.main_window.searxng_time_range_input.text().strip() or None
            searxng_categories = self.main_window.searxng_categories_input.text().strip() or None
            searxng_engines = self.main_window.searxng_engine_selector.getSelectedEngines()
            if not isinstance(searxng_engines, list): searxng_engines = []
            searxng_config['base_url'] = searxng_url
            searxng_config['time_range'] = searxng_time_range
            searxng_config['categories'] = searxng_categories
            searxng_config['engines'] = searxng_engines
            search_limit = searxng_config.get('max_results', 5)
        else:
            search_limit = ddg_config.get('max_results', 5)

        if save_config(self.config_path, self.config_data):
            self.log_status(f"Current settings saved to {self.config_path}")
        else:
            self.log_status(f"[ERROR] Failed to save configuration before starting search: {self.config_path}")
            QMessageBox.warning(self.main_window, "Config Error", f"Could not save settings to {self.config_path}. Search may use outdated settings.")

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
            "output_format": llm_cfg['output_format'],
            "include_visuals": self.main_window.include_visuals_checkbox.isChecked() # Add the new checkbox state
        }

        # --- Start Search Worker ---
        self.log_status("Starting main search worker...")
        self._set_ui_for_main_search()

        self.search_worker = SearchWorker(search_params, self.main_window)
        self.search_worker.progress_updated.connect(self.log_status)
        self.search_worker.search_complete.connect(self.on_search_complete)
        self.search_worker.error_occurred.connect(self.on_search_error)
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.start()

    # --- Search Result Handling Slots ---
    def on_search_complete(self, report_path, final_answer_content, toc_tree_nodes):
        """Handle successful search completion."""
        self.log_status(f"Search finished successfully!")
        # Emit signal upwards instead of directly manipulating UI results area
        self.search_process_complete.emit(report_path, final_answer_content, toc_tree_nodes)
        # Reset UI state related to running the search
        self._reset_ui_after_main_search_finish(success=True)


    def on_search_error(self, error_message):
        """Handle search failure."""
        self.log_status(f"Search Error: {error_message}")
        # Emit signal upwards
        self.search_process_error.emit(error_message)
        # Reset UI state related to running the search
        self._reset_ui_after_main_search_finish(success=False)

    def on_search_finished(self):
        """Called when the SearchWorker thread finishes (success or error)."""
        self.search_worker = None # Clear worker reference
        # UI reset is handled in success/error slots

    # --- UI State Management Helpers ---
    def _set_ui_for_preprocessing(self):
        """Sets UI elements for topic extraction or enhancement preview."""
        self.main_window.run_button.setEnabled(False)
        self.main_window.progress_bar.setVisible(True)
        self.main_window.progress_bar.setRange(0, 0) # Indeterminate
        # Cancel button might not be applicable here, keep hidden/disabled
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)

    def _set_ui_for_main_search(self):
        """Sets UI elements when the main search starts."""
        self.main_window.run_button.setEnabled(False)
        self.main_window.open_report_button.setEnabled(False)
        self.main_window.open_folder_button.setEnabled(False)
        self.main_window.share_email_button.setEnabled(False)
        self.main_window.report_path_label.setText("Running Search...")
        self.main_window.current_report_path = None
        self.main_window.current_results_dir = None
        self.main_window.progress_bar.setVisible(True)
        self.main_window.progress_bar.setRange(0, 0) # Indeterminate
        self.main_window.cancel_button.setVisible(True)
        self.main_window.cancel_button.setEnabled(True)
        # Ensure search provider visibility is correct
        self.main_window.handle_search_provider_change(self.main_window.search_provider_combo.currentText())


    def _reset_ui_after_preprocessing_failure(self):
        """Resets UI after topic extraction or enhancement preview fails."""
        self.main_window.run_button.setEnabled(True)
        self.main_window.progress_bar.setVisible(False)
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)

    def _reset_ui_after_preprocessing_success(self):
        """Resets UI after topic extraction or enhancement preview succeeds (but before main search)."""
        self.main_window.run_button.setEnabled(True)
        self.main_window.progress_bar.setVisible(False)
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)

    def _reset_ui_after_main_search_finish(self, success: bool):
        """Resets UI elements after the main search worker finishes."""
        self.main_window.run_button.setEnabled(True)
        self.main_window.progress_bar.setVisible(False)
        self.main_window.progress_bar.setRange(0, 100) # Reset range
        self.main_window.cancel_button.setVisible(False)
        self.main_window.cancel_button.setEnabled(False)
        if not success:
            self.main_window.report_path_label.setText("Search failed.")
            self.main_window.open_report_button.setEnabled(False)
            self.main_window.open_folder_button.setEnabled(False)
            self.main_window.share_email_button.setEnabled(False)
        # If success, the main controller/result manager handles enabling buttons

    # --- Cancellation ---
    def cancel_current_operation(self):
        """Requests cancellation of the currently running worker."""
        if self.search_worker and self.search_worker.isRunning():
            self.log_status("Requesting search cancellation...")
            self.search_worker.request_cancellation()
            self.main_window.cancel_button.setEnabled(False) # Disable immediately
            self.log_status("Cancellation requested. Waiting for search worker to stop...")
            return True # Indicate cancellation was attempted
        elif self.topic_extractor_worker and self.topic_extractor_worker.isRunning():
            self.log_status("Topic extraction is running. Cancellation not implemented.")
            # Optionally terminate: self.topic_extractor_worker.terminate()
            return False
        elif self.query_enhancer_worker and self.query_enhancer_worker.isRunning():
             self.log_status("Query enhancement preview is running. Cancellation not implemented.")
             # Optionally terminate: self.query_enhancer_worker.terminate()
             return False
        else:
            self.log_status("No cancellable operation running in orchestrator.")
            return False

    # --- Cleanup ---
    def shutdown_workers(self):
        """Stop all running worker threads managed by this orchestrator."""
        self.log_status("Shutting down search orchestrator workers...")
        if self.search_worker and self.search_worker.isRunning():
             self.log_status("Attempting to cancel search on close...")
             self.search_worker.request_cancellation()
             self.search_worker.wait(3000) # Wait 3 seconds
             if self.search_worker.isRunning():
                 self.log_status("Search worker did not stop gracefully, terminating...")
                 self.search_worker.terminate()
                 self.search_worker.wait()
        if self.topic_extractor_worker and self.topic_extractor_worker.isRunning():
             self.log_status("Terminating topic extraction on close...")
             self.topic_extractor_worker.terminate()
             self.topic_extractor_worker.wait()
        if self.query_enhancer_worker and self.query_enhancer_worker.isRunning():
             self.log_status("Terminating query enhancement on close...")
             self.query_enhancer_worker.terminate()
             self.query_enhancer_worker.wait()
        self.log_status("Search orchestrator worker shutdown complete.")
