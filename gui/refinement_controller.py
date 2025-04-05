#!/usr/bin/env python3
# gui/refinement_controller.py

import logging
import re
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QInputDialog

# Import worker thread using relative import
try:
    from .workers import RefinementWorker
except ImportError as e:
    print(f"Error importing workers in refinement_controller.py: {e}")
    import sys
    sys.exit(1)

# Import config loading utility
from config_utils import load_config

class RefinementController(QObject):
    """
    Manages the process of refining sections of the report, including
    handling the RefinementWorker and interacting with the ResultDisplayManager.
    """
    # Signal to pass completion data upwards (anchor_id, refined_content)
    refinement_process_complete = pyqtSignal(str, str)
    # Signal to pass errors upwards
    refinement_process_error = pyqtSignal(str, str) # anchor_id, error_message
    # Signal for status updates
    status_update = pyqtSignal(str)

    def __init__(self, main_window, config_path, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.config_path = config_path
        self.config_data = load_config(config_path) # Load initial config

        # Worker instance managed by this controller
        self.refinement_worker = None

    def log_status(self, message):
        """Emit status update signal."""
        self.status_update.emit(message)

    def start_refinement_process(self, anchor_id, instruction, current_report_html):
        """
        Public method called by GuiController when refinement is requested.
        Validates inputs, extracts content, and starts the RefinementWorker.
        """
        self.log_status(f"Refinement process started for anchor: {anchor_id}")

        # 1. Extract Section Content:
        section_content = self._extract_section_content(anchor_id, current_report_html)
        if not section_content:
             self.log_status(f"[Error] Could not extract content for anchor {anchor_id}.")
             QMessageBox.warning(self.main_window, "Refinement Error", "Could not extract the content for the selected section.")
             # Emit error signal? Or just log? Let's emit.
             self.refinement_process_error.emit(anchor_id, "Could not extract section content.")
             return

        # 2. Check if Refinement Worker is already running
        if self.refinement_worker and self.refinement_worker.isRunning():
            self.log_status("[Warning] Refinement is already in progress.")
            QMessageBox.warning(self.main_window, "Busy", "Another refinement operation is already running.")
            return

        # 3. Prepare LLM config (use current RAG settings from UI)
        rag_model_type = self.main_window.rag_model_combo.currentText()
        selected_generative_model = None
        if rag_model_type == "gemini":
            selected_generative_model = self.main_window.gemini_model_combo.currentText()
        elif rag_model_type == "openrouter":
            selected_generative_model = self.main_window.openrouter_model_combo.currentText()
        # Add gemma/pali if needed in the future

        if rag_model_type == "None":
             QMessageBox.warning(self.main_window, "Refinement Error", "Please select a RAG model (Gemini, OpenRouter, Gemma) to perform refinement.")
             self.refinement_process_error.emit(anchor_id, "No RAG model selected.")
             return
        if (rag_model_type in ["gemini", "openrouter"]) and not selected_generative_model:
             QMessageBox.warning(self.main_window, "Refinement Error", f"Please select a specific model for the '{rag_model_type}' provider.")
             self.refinement_process_error.emit(anchor_id, f"No specific model selected for {rag_model_type}.")
             return

        # Reload config for potential API key updates
        self.config_data = load_config(self.config_path)
        llm_config = {
            "provider": rag_model_type,
            "model_id": selected_generative_model, # Will be None for gemma/pali, handled in task
            "api_key": self.config_data.get('api_keys', {}).get(f'{rag_model_type}_api_key'),
            "personality": self.main_window.personality_input.text() or None
        }

        # 4. Start Refinement Worker
        self.log_status(f"Starting refinement worker for anchor '{anchor_id}' with instruction: '{instruction}'")
        self._set_ui_for_refinement()

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

    def _extract_section_content(self, anchor_id, full_html):
        """
        Extracts HTML content associated with an anchor from the provided full HTML.
        Attempts to find the content between the specified anchor and the next anchor tag.
        Returns the HTML fragment (including the starting anchor) or None if extraction fails.
        """
        self.log_status(f"Attempting to extract content for anchor: {anchor_id}")

        # Pattern to find the starting anchor tag specifically
        start_pattern = re.compile(r'(<a\s+name="' + re.escape(anchor_id) + r'"\s*>\s*</a>)', re.IGNORECASE)
        start_match = start_pattern.search(full_html)

        if not start_match:
            self.log_status(f"[Error] Start anchor tag not found for {anchor_id}")
            return None

        start_index = start_match.end() # Position *after* the starting anchor tag

        # Pattern to find the *next* <a name="..."> tag after the starting position
        next_anchor_pattern = re.compile(r'<a\s+name=".*?"\s*>\s*</a>', re.IGNORECASE)
        next_match = next_anchor_pattern.search(full_html, pos=start_index)

        if next_match:
            end_index = next_match.start() # Position *before* the next anchor tag
            self.log_status(f"Found next anchor at index {end_index}")
            # Return the content including the starting anchor tag itself up to the next anchor
            section_html = start_match.group(1) + full_html[start_index:end_index]
        else:
            # If no next anchor, take content from start anchor to the end of the document
            self.log_status("No subsequent anchor found, taking content to the end.")
            section_html = full_html[start_match.start():] # Includes start anchor

        self.log_status(f"Extracted HTML segment length for {anchor_id}: {len(section_html)}")
        return section_html

    # --- Refinement Worker Slots ---
    def _on_refinement_complete(self, anchor_id, refined_content):
        """Handles successful refinement completion by emitting a signal."""
        self.log_status(f"Refinement successful for anchor '{anchor_id}'.")
        self.refinement_process_complete.emit(anchor_id, refined_content)
        self._reset_ui_after_refinement_finish()

    def _on_refinement_error(self, anchor_id, error_message):
        """Handles refinement errors by emitting a signal."""
        self.log_status(f"[Error] Refinement failed for anchor '{anchor_id}': {error_message}")
        self.refinement_process_error.emit(anchor_id, error_message)
        QMessageBox.critical(self.main_window, "Refinement Error", f"Failed to refine section '{anchor_id}':\n{error_message}")
        self._reset_ui_after_refinement_finish()

    def _on_refinement_finished(self):
        """Called when the RefinementWorker finishes (success or error)."""
        self.log_status("Refinement worker finished.")
        self.refinement_worker = None # Clear worker reference
        # UI reset is handled in success/error slots

    # --- UI State Management Helpers ---
    def _set_ui_for_refinement(self):
        """Sets UI elements when refinement starts."""
        # Indicate progress (e.g., using the main progress bar)
        self.main_window.progress_bar.setVisible(True)
        self.main_window.progress_bar.setRange(0, 0) # Indeterminate
        # Disable run button? Maybe not necessary for refinement.

    def _reset_ui_after_refinement_finish(self):
        """Resets UI elements after refinement finishes."""
        # Hide progress bar only if no other major worker (like search) is running
        # This logic might need to be coordinated by the main GuiController
        # For now, assume we can hide it if the refinement worker is done.
        if not (self.main_window.controller.search_orchestrator.search_worker and \
                self.main_window.controller.search_orchestrator.search_worker.isRunning()):
             self.main_window.progress_bar.setVisible(False)
             self.main_window.progress_bar.setRange(0, 100) # Reset

    # --- Cleanup ---
    def shutdown_worker(self):
        """Stop the refinement worker if running."""
        self.log_status("Shutting down refinement controller worker...")
        if self.refinement_worker and self.refinement_worker.isRunning():
            self.log_status("Terminating refinement worker on close...")
            # Refinement might be quick, terminate might be okay
            self.refinement_worker.terminate()
            self.refinement_worker.wait()
        self.log_status("Refinement controller worker shutdown complete.")
