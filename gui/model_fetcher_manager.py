#!/usr/bin/env python3
# gui/model_fetcher_manager.py

import logging
from PyQt6.QtCore import QObject, pyqtSignal

# Import worker threads using relative import
try:
    from .workers import GeminiFetcher, OpenRouterFetcher
except ImportError as e:
    print(f"Error importing workers in model_fetcher_manager.py: {e}")
    import sys
    sys.exit(1)

class ModelFetcherManager(QObject):
    """
    Manages fetching available generative models from Gemini and OpenRouter,
    handling their respective worker threads and UI updates.
    """
    # Signal for status updates
    status_update = pyqtSignal(str)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # Worker instances managed by this manager
        self.gemini_fetcher = None
        self.openrouter_fetcher = None

    def log_status(self, message):
        """Emit status update signal."""
        self.status_update.emit(message)

    # --- Generative Model Fetching Slots ---
    def fetch_gemini_models(self):
        """Start the GeminiFetcher thread for *generative* models."""
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
            self.log_status("Already fetching Gemini generative models...")
            return

        self.log_status("Attempting to fetch Gemini generative models (requires GEMINI_API_KEY)...")
        self.main_window.gemini_fetch_button.setEnabled(False)
        self.gemini_fetcher = GeminiFetcher(self.main_window) # Pass main_window as parent
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

    # --- Cleanup ---
    def shutdown_workers(self):
        """Stop all running fetcher threads."""
        self.log_status("Shutting down model fetcher workers...")
        if self.gemini_fetcher and self.gemini_fetcher.isRunning():
             self.log_status("Terminating Gemini fetcher on close...")
             self.gemini_fetcher.terminate()
             self.gemini_fetcher.wait()
        if self.openrouter_fetcher and self.openrouter_fetcher.isRunning():
             self.log_status("Terminating OpenRouter fetcher on close...")
             self.openrouter_fetcher.terminate()
             self.openrouter_fetcher.wait()
        self.log_status("Model fetcher worker shutdown complete.")
