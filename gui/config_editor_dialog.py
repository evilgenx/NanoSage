#!/usr/bin/env python3
# gui/config_editor_dialog.py

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout,
    QLineEdit, QSpinBox, QCheckBox, QComboBox, QPushButton,
    QDialogButtonBox, QMessageBox, QLabel, QHBoxLayout, QApplication
)
from PyQt6.QtGui import QIcon

# Import config utilities
from config_utils import load_config, save_config, DEFAULT_CONFIG

class ConfigEditorDialog(QDialog):
    """
    A dialog window for editing the application's configuration (config.yaml).
    """
    def __init__(self, config_path="config.yaml", parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.config_data = load_config(self.config_path) # Load current config

        self.setWindowTitle("Edit Configuration")
        self.setMinimumWidth(600)

        # Main layout
        layout = QVBoxLayout(self)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self._create_general_tab()
        self._create_retrieval_tab()
        self._create_llm_tab()
        self._create_search_tab()
        self._create_knowledge_base_tab() # <<< Add new tab creation
        self._create_cache_tab()
        self._create_api_keys_tab()
        # Add more tabs as needed (e.g., advanced?)

        # Dialog buttons (Save, Cancel)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept) # Connect Save to accept
        self.button_box.rejected.connect(self.reject) # Connect Cancel to reject
        layout.addWidget(self.button_box)

        # Load initial values into widgets
        self._load_values()

    def _create_general_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)

        # Widgets for general settings
        self.general_web_search = QCheckBox("Enable Web Search by Default")
        self.general_max_depth = QSpinBox()
        self.general_max_depth.setRange(0, 10)
        self.general_device = QComboBox()
        self.general_device.addItems(["cpu", "cuda", "rocm"])
        self.general_corpus_dir = QLineEdit() # Display only, set in main window
        self.general_corpus_dir.setReadOnly(True)
        self.general_corpus_dir.setPlaceholderText("Set via 'Browse...' button in main window")

        layout.addRow(self.general_web_search)
        layout.addRow("Default Max Recursion Depth:", self.general_max_depth)
        layout.addRow("Default Embedding Device:", self.general_device)
        layout.addRow("Current Local Corpus:", self.general_corpus_dir)

        self.tabs.addTab(tab, "General")

    def _create_retrieval_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)

        # Widgets for retrieval settings
        self.retrieval_embedding_model = QComboBox()
        # Populate common local models, others might be added dynamically if needed
        self.retrieval_embedding_model.addItems([
            "colpali", "all-minilm", "multi-qa-mpnet", "all-mpnet", "multi-qa-minilm"
        ])
        self.retrieval_top_k = QSpinBox()
        self.retrieval_top_k.setRange(1, 50)

        layout.addRow("Default Embedding Model:", self.retrieval_embedding_model)
        layout.addRow("Default Top K Results:", self.retrieval_top_k)

        self.tabs.addTab(tab, "Retrieval")

    def _create_llm_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)

        # Widgets for LLM settings
        self.llm_rag_model_type = QComboBox()
        self.llm_rag_model_type.addItems(["gemma", "pali", "gemini", "openrouter", "None"])
        self.llm_rag_personality = QLineEdit()
        self.llm_gemma_model_id = QLineEdit()
        self.llm_gemini_model_id = QLineEdit()
        self.llm_openrouter_model_id = QLineEdit()
        # Prompt template path - maybe make this selectable later? For now, just display.
        self.llm_rag_report_prompt_template = QLineEdit()
        self.llm_rag_report_prompt_template.setReadOnly(True) # Display only for now

        layout.addRow("Default RAG Model Type:", self.llm_rag_model_type)
        layout.addRow("Default RAG Personality:", self.llm_rag_personality)
        layout.addRow("Default Gemma Model ID (Ollama):", self.llm_gemma_model_id)
        layout.addRow("Default Gemini Model ID:", self.llm_gemini_model_id)
        layout.addRow("Default OpenRouter Model ID:", self.llm_openrouter_model_id)
        layout.addRow("Report Prompt Template Path:", self.llm_rag_report_prompt_template)
        # Note: Output formats are managed via prompt files, not directly edited here yet.

        self.tabs.addTab(tab, "LLM / RAG")

    def _create_search_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)

        # Widgets for search settings
        self.search_provider = QComboBox()
        self.search_provider.addItems(["duckduckgo", "searxng"])
        self.search_enable_iterative = QCheckBox("Enable Iterative Search by Default")
        self.search_include_visuals = QCheckBox("Include Visuals in Report by Default")

        # DuckDuckGo
        self.search_ddg_max_results = QSpinBox()
        self.search_ddg_max_results.setRange(1, 20)

        # SearXNG
        self.search_searxng_base_url = QLineEdit()
        self.search_searxng_max_results = QSpinBox()
        self.search_searxng_max_results.setRange(1, 20)
        self.search_searxng_language = QLineEdit()
        self.search_searxng_safesearch = QComboBox()
        self.search_searxng_safesearch.addItems(["0 (Off)", "1 (Moderate)", "2 (Strict)"])
        self.search_searxng_time_range = QLineEdit()
        self.search_searxng_categories = QLineEdit()
        # Engines are complex, maybe just display for now or link to main window selector?
        self.search_searxng_engines_display = QLabel("Managed in main window")

        layout.addRow("Default Search Provider:", self.search_provider)
        layout.addRow(self.search_enable_iterative)
        layout.addRow(self.search_include_visuals)
        layout.addRow(QLabel("--- DuckDuckGo ---"))
        layout.addRow("Default Max Results:", self.search_ddg_max_results)
        layout.addRow(QLabel("--- SearXNG ---"))
        layout.addRow("Base URL:", self.search_searxng_base_url)
        layout.addRow("Default Max Results:", self.search_searxng_max_results)
        layout.addRow("Language Code (e.g., en):", self.search_searxng_language)
        layout.addRow("Safesearch Level:", self.search_searxng_safesearch)
        layout.addRow("Time Range Filter:", self.search_searxng_time_range)
        layout.addRow("Categories (comma-sep):", self.search_searxng_categories)
        layout.addRow("Engines:", self.search_searxng_engines_display)

        self.tabs.addTab(tab, "Search")

    def _create_knowledge_base_tab(self): # <<< Add method for new tab
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)

        # Widgets for knowledge base settings
        self.kb_provider = QComboBox()
        self.kb_provider.addItems(["chromadb"]) # Only chromadb supported for now
        self.kb_provider.setEnabled(False) # Disable changing provider for now

        self.kb_chromadb_path = QLineEdit()
        self.kb_chromadb_collection = QLineEdit()

        layout.addRow("Vector Store Provider:", self.kb_provider)
        layout.addRow(QLabel("--- ChromaDB Settings ---"))
        layout.addRow("Database Path:", self.kb_chromadb_path)
        layout.addRow("Collection Name:", self.kb_chromadb_collection)

        self.tabs.addTab(tab, "Knowledge Base")


    def _create_cache_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)

        # Widgets for cache settings
        self.cache_enabled = QCheckBox("Enable Caching")
        self.cache_db_path = QLineEdit()

        layout.addRow(self.cache_enabled)
        layout.addRow("Cache Database Path:", self.cache_db_path)

        self.tabs.addTab(tab, "Cache")

    def _create_api_keys_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)

        # Widgets for API keys
        self.api_gemini_key = QLineEdit()
        self.api_gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_openrouter_key = QLineEdit()
        self.api_openrouter_key.setEchoMode(QLineEdit.EchoMode.Password)

        layout.addRow("Gemini API Key:", self.api_gemini_key)
        layout.addRow("OpenRouter API Key:", self.api_openrouter_key)
        layout.addRow(QLabel("Note: Leave blank to use environment variables."))

        self.tabs.addTab(tab, "API Keys")

    def _load_values(self):
        """Load values from self.config_data into the UI widgets."""
        try:
            # General
            general_cfg = self.config_data.get('general', {})
            self.general_web_search.setChecked(general_cfg.get('web_search', DEFAULT_CONFIG['general']['web_search']))
            self.general_max_depth.setValue(general_cfg.get('max_depth', DEFAULT_CONFIG['general']['max_depth']))
            self.general_device.setCurrentText(general_cfg.get('device', DEFAULT_CONFIG['general']['device']))
            self.general_corpus_dir.setText(general_cfg.get('corpus_dir', '') or '') # Display only

            # Retrieval
            retrieval_cfg = self.config_data.get('retrieval', {})
            self.retrieval_embedding_model.setCurrentText(retrieval_cfg.get('embedding_model', DEFAULT_CONFIG['retrieval']['embedding_model']))
            self.retrieval_top_k.setValue(retrieval_cfg.get('top_k', DEFAULT_CONFIG['retrieval']['top_k']))

            # LLM
            llm_cfg = self.config_data.get('llm', {})
            self.llm_rag_model_type.setCurrentText(llm_cfg.get('rag_model_type', DEFAULT_CONFIG['llm']['rag_model_type']))
            self.llm_rag_personality.setText(llm_cfg.get('rag_personality', DEFAULT_CONFIG['llm']['rag_personality']))
            self.llm_gemma_model_id.setText(llm_cfg.get('gemma_model_id', DEFAULT_CONFIG['llm']['gemma_model_id']))
            self.llm_gemini_model_id.setText(llm_cfg.get('gemini_model_id', DEFAULT_CONFIG['llm']['gemini_model_id']))
            self.llm_openrouter_model_id.setText(llm_cfg.get('openrouter_model_id', DEFAULT_CONFIG['llm']['openrouter_model_id']))
            self.llm_rag_report_prompt_template.setText(llm_cfg.get('rag_report_prompt_template', DEFAULT_CONFIG['llm']['rag_report_prompt_template']))

            # Search
            search_cfg = self.config_data.get('search', {})
            self.search_provider.setCurrentText(search_cfg.get('provider', DEFAULT_CONFIG['search']['provider']))
            self.search_enable_iterative.setChecked(search_cfg.get('enable_iterative_search', DEFAULT_CONFIG['search']['enable_iterative_search']))
            self.search_include_visuals.setChecked(search_cfg.get('include_visuals', DEFAULT_CONFIG['search']['include_visuals']))
            ddg_cfg = search_cfg.get('duckduckgo', {})
            self.search_ddg_max_results.setValue(ddg_cfg.get('max_results', DEFAULT_CONFIG['search']['duckduckgo']['max_results']))
            searxng_cfg = search_cfg.get('searxng', {})
            self.search_searxng_base_url.setText(searxng_cfg.get('base_url', DEFAULT_CONFIG['search']['searxng']['base_url']))
            self.search_searxng_max_results.setValue(searxng_cfg.get('max_results', DEFAULT_CONFIG['search']['searxng']['max_results']))
            self.search_searxng_language.setText(searxng_cfg.get('language', DEFAULT_CONFIG['search']['searxng']['language']))
            safesearch_map = {"0 (Off)": 0, "1 (Moderate)": 1, "2 (Strict)": 2}
            safesearch_val = str(searxng_cfg.get('safesearch', DEFAULT_CONFIG['search']['searxng']['safesearch']))
            safesearch_text = next((k for k, v in safesearch_map.items() if str(v) == safesearch_val), "1 (Moderate)")
            self.search_searxng_safesearch.setCurrentText(safesearch_text)
            self.search_searxng_time_range.setText(searxng_cfg.get('time_range', '') or '')
            self.search_searxng_categories.setText(searxng_cfg.get('categories', '') or '')

            # Cache
            cache_cfg = self.config_data.get('cache', {})
            self.cache_enabled.setChecked(cache_cfg.get('enabled', DEFAULT_CONFIG['cache']['enabled']))
            self.cache_db_path.setText(cache_cfg.get('db_path', DEFAULT_CONFIG['cache']['db_path']))

            # API Keys
            api_keys_cfg = self.config_data.get('api_keys', {})
            self.api_gemini_key.setText(api_keys_cfg.get('gemini_api_key', ''))
            self.api_openrouter_key.setText(api_keys_cfg.get('openrouter_api_key', ''))

            # Knowledge Base
            kb_cfg = self.config_data.get('knowledge_base', {})
            self.kb_provider.setCurrentText(kb_cfg.get('provider', DEFAULT_CONFIG['knowledge_base']['provider']))
            chromadb_cfg = kb_cfg.get('chromadb', {})
            self.kb_chromadb_path.setText(chromadb_cfg.get('path', DEFAULT_CONFIG['knowledge_base']['chromadb']['path']))
            self.kb_chromadb_collection.setText(chromadb_cfg.get('collection_name', DEFAULT_CONFIG['knowledge_base']['chromadb']['collection_name']))


        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Error loading configuration values: {e}")
            print(f"[ERROR] ConfigEditorDialog._load_values: {e}")

    def _save_values(self):
        """Read values from UI widgets and update self.config_data."""
        try:
            # Ensure sections exist
            general_cfg = self.config_data.setdefault('general', {})
            retrieval_cfg = self.config_data.setdefault('retrieval', {})
            llm_cfg = self.config_data.setdefault('llm', {})
            search_cfg = self.config_data.setdefault('search', {})
            ddg_cfg = search_cfg.setdefault('duckduckgo', {})
            searxng_cfg = search_cfg.setdefault('searxng', {})
            cache_cfg = self.config_data.setdefault('cache', {})
            api_keys_cfg = self.config_data.setdefault('api_keys', {})
            kb_cfg = self.config_data.setdefault('knowledge_base', {}) # <<< Ensure KB section exists
            chromadb_cfg = kb_cfg.setdefault('chromadb', {}) # <<< Ensure ChromaDB section exists

            # General
            general_cfg['web_search'] = self.general_web_search.isChecked()
            general_cfg['max_depth'] = self.general_max_depth.value()
            general_cfg['device'] = self.general_device.currentText()
            # corpus_dir is not edited here

            # Retrieval
            retrieval_cfg['embedding_model'] = self.retrieval_embedding_model.currentText()
            retrieval_cfg['top_k'] = self.retrieval_top_k.value()

            # LLM
            llm_cfg['rag_model_type'] = self.llm_rag_model_type.currentText()
            llm_cfg['rag_personality'] = self.llm_rag_personality.text()
            llm_cfg['gemma_model_id'] = self.llm_gemma_model_id.text()
            llm_cfg['gemini_model_id'] = self.llm_gemini_model_id.text()
            llm_cfg['openrouter_model_id'] = self.llm_openrouter_model_id.text()
            # prompt template path is not edited here

            # Search
            search_cfg['provider'] = self.search_provider.currentText()
            search_cfg['enable_iterative_search'] = self.search_enable_iterative.isChecked()
            search_cfg['include_visuals'] = self.search_include_visuals.isChecked()
            ddg_cfg['max_results'] = self.search_ddg_max_results.value()
            searxng_cfg['base_url'] = self.search_searxng_base_url.text()
            searxng_cfg['max_results'] = self.search_searxng_max_results.value()
            searxng_cfg['language'] = self.search_searxng_language.text()
            safesearch_map = {"0 (Off)": 0, "1 (Moderate)": 1, "2 (Strict)": 2}
            searxng_cfg['safesearch'] = safesearch_map.get(self.search_searxng_safesearch.currentText(), 1)
            searxng_cfg['time_range'] = self.search_searxng_time_range.text() or None
            searxng_cfg['categories'] = self.search_searxng_categories.text() or None
            # engines are not edited here

            # Cache
            cache_cfg['enabled'] = self.cache_enabled.isChecked()
            cache_cfg['db_path'] = self.cache_db_path.text()

            # API Keys (save empty string if blank, don't save None)
            api_keys_cfg['gemini_api_key'] = self.api_gemini_key.text()
            api_keys_cfg['openrouter_api_key'] = self.api_openrouter_key.text()

            # Knowledge Base
            kb_cfg['provider'] = self.kb_provider.currentText() # Although disabled, read it
            chromadb_cfg['path'] = self.kb_chromadb_path.text()
            chromadb_cfg['collection_name'] = self.kb_chromadb_collection.text()

            return True # Indicate success

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error preparing configuration values for saving: {e}")
            print(f"[ERROR] ConfigEditorDialog._save_values: {e}")
            return False # Indicate failure

    def accept(self):
        """Override accept to save values before closing."""
        if self._save_values():
            if save_config(self.config_path, self.config_data):
                QMessageBox.information(self, "Config Saved", f"Configuration saved successfully to\n{self.config_path}")
                super().accept() # Close dialog if save successful
            else:
                QMessageBox.critical(self, "Save Error", f"Failed to write configuration file:\n{self.config_path}")
                # Don't close dialog on save failure
        # else: _save_values already showed an error message

# Example usage (for testing)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    # Ensure a default config exists for testing
    if not os.path.exists("config.yaml"):
         save_config("config.yaml", DEFAULT_CONFIG)
    dialog = ConfigEditorDialog()
    if dialog.exec():
        print("Config saved.")
    else:
        print("Config cancelled.")
    sys.exit()
