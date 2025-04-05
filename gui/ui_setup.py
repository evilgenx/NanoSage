#!/usr/bin/env python3
# gui/ui_setup.py

"""
Module responsible for setting up the UI elements of the MainWindow.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QScrollArea, QTreeWidgetItemIterator,
    QSizePolicy, QProgressBar, QTabWidget, QApplication # Added QTabWidget and QApplication
)
from PyQt6.QtGui import QIcon # Import QIcon
from PyQt6.QtCore import QSize # Import QSize for icon sizing

from gui.ui_components.searxng_selector import SearxngEngineSelector # Import the new widget
# Note: MainWindow itself is passed in, so we don't import it directly
# Note: QFileDialog, QMessageBox, QMainWindow are used in MainWindow logic, not setup

def setup_main_window_ui(main_window):
    """
    Initializes and arranges UI elements for the given MainWindow instance.
    Widgets are added as attributes to the main_window object.

    Args:
        main_window (MainWindow): The main window instance to populate with UI elements.
    """
    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    main_layout = QHBoxLayout(central_widget)
    main_layout.setContentsMargins(10, 10, 10, 10) # Add margins around the main layout
    main_layout.setSpacing(10) # Add spacing between left and right panels

    # --- Left Panel: Configuration ---
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 0, 0) # Let group boxes handle inner margins
    left_layout.setSpacing(10) # Space between query group, tabs, button
    left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # Query Input
    query_group = QGroupBox("Search Query")
    query_layout = QVBoxLayout()
    query_layout.setContentsMargins(10, 10, 10, 10) # Margins inside query group
    query_layout.setSpacing(8) # Spacing inside query group
    main_window.query_input = QTextEdit()
    main_window.query_input.setObjectName("query_input") # Set object name
    main_window.query_input.setPlaceholderText("Enter your research query here...")
    query_layout.addWidget(main_window.query_input)

    # Add the new checkbox for topic extraction
    main_window.extract_topics_checkbox = QCheckBox("Extract Topics from Input (for large text)")
    main_window.extract_topics_checkbox.setToolTip("Check this to use the LLM to extract key topics/phrases from the input text above, instead of treating it as a direct query. Useful for pasting articles or long text.")
    query_layout.addWidget(main_window.extract_topics_checkbox)

    query_group.setLayout(query_layout)
    left_layout.addWidget(query_group)

    # --- Configuration Tabs ---
    config_tabs = QTabWidget()

    # -- General Tab --
    general_tab = QWidget()
    general_layout = QFormLayout(general_tab)
    general_layout.setContentsMargins(10, 10, 10, 10) # Margins for tab content
    general_layout.setVerticalSpacing(8) # Space between rows
    general_layout.setHorizontalSpacing(10) # Space between label/widget

    main_window.web_search_checkbox = QCheckBox("Enable Web Search")
    general_layout.addRow(main_window.web_search_checkbox)

    corpus_layout = QHBoxLayout()
    main_window.corpus_dir_label = QLineEdit()
    main_window.corpus_dir_label.setPlaceholderText("Optional: Path to local documents")
    main_window.corpus_dir_label.setReadOnly(True)
    main_window.corpus_dir_button = QPushButton("Browse...")
    # Add Icon for Browse
    browse_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_DirOpenIcon)
    main_window.corpus_dir_button.setIcon(browse_icon)
    corpus_layout.addWidget(main_window.corpus_dir_label)
    corpus_layout.addWidget(main_window.corpus_dir_button)
    general_layout.addRow("Local Corpus:", corpus_layout)

    main_window.max_depth_spinbox = QSpinBox()
    main_window.max_depth_spinbox.setRange(0, 10)
    main_window.max_depth_spinbox.setValue(1) # Default value
    general_layout.addRow("Max Recursion Depth:", main_window.max_depth_spinbox)

    main_window.top_k_spinbox = QSpinBox()
    main_window.top_k_spinbox.setRange(1, 50)
    main_window.top_k_spinbox.setValue(3) # Default value
    general_layout.addRow("Top K Results:", main_window.top_k_spinbox)

    config_tabs.addTab(general_tab, "General")

    # -- Search Tab --
    search_tab = QWidget()
    search_layout = QFormLayout(search_tab) # Use QFormLayout for consistency
    search_layout.setContentsMargins(10, 10, 10, 10) # Margins for tab content
    search_layout.setVerticalSpacing(8) # Space between rows
    search_layout.setHorizontalSpacing(10) # Space between label/widget

    main_window.search_provider_combo = QComboBox()
    main_window.search_provider_combo.addItems(["DuckDuckGo", "SearXNG"])
    search_layout.addRow("Search Provider:", main_window.search_provider_combo)

    # Iterative Search Checkbox
    main_window.iterative_search_checkbox = QCheckBox("Enable Iterative Search (Experimental)")
    search_layout.addRow(main_window.iterative_search_checkbox)

    # SearXNG Specific Settings (visibility handled in MainWindow)
    main_window.searxng_base_url_label = QLabel("SearXNG URL:")
    main_window.searxng_base_url_input = QLineEdit()
    main_window.searxng_time_range_label = QLabel("Time Range:")
    main_window.searxng_time_range_input = QLineEdit()
    main_window.searxng_time_range_input.setPlaceholderText("Optional: day, week, month, year")
    main_window.searxng_categories_label = QLabel("Categories:")
    main_window.searxng_categories_input = QLineEdit()
    main_window.searxng_categories_input.setPlaceholderText("Optional: general,images,...")

    search_layout.addRow(main_window.searxng_base_url_label, main_window.searxng_base_url_input)
    search_layout.addRow(main_window.searxng_time_range_label, main_window.searxng_time_range_input)
    search_layout.addRow(main_window.searxng_categories_label, main_window.searxng_categories_input)

    # SearXNG Engine Selection Widget
    main_window.searxng_engine_group = QGroupBox("SearXNG Engine Selection")
    searxng_engine_layout = QVBoxLayout() # Use QVBoxLayout for the group
    searxng_engine_layout.setContentsMargins(5, 5, 5, 5) # Slightly tighter margins for this specific group
    main_window.searxng_engine_selector = SearxngEngineSelector()
    searxng_engine_layout.addWidget(main_window.searxng_engine_selector)
    main_window.searxng_engine_group.setLayout(searxng_engine_layout)
    search_layout.addRow(main_window.searxng_engine_group) # Add the group box to the search tab layout

    config_tabs.addTab(search_tab, "Search")

    # -- Embeddings Tab --
    embeddings_tab = QWidget()
    embedding_layout = QFormLayout(embeddings_tab)
    embedding_layout.setContentsMargins(10, 10, 10, 10) # Margins for tab content
    embedding_layout.setVerticalSpacing(8) # Space between rows
    embedding_layout.setHorizontalSpacing(10) # Space between label/widget

    main_window.device_combo = QComboBox()
    main_window.device_combo.addItems(["cpu", "cuda", "rocm"]) # Added rocm
    embedding_layout.addRow("Embedding Device:", main_window.device_combo)

    main_window.embedding_model_combo = QComboBox()
    main_window.embedding_model_label = QLabel("Embedding Model:")
    embedding_layout.addRow(main_window.embedding_model_label, main_window.embedding_model_combo)

    config_tabs.addTab(embeddings_tab, "Embeddings")

    # -- RAG Tab --
    rag_tab = QWidget()
    rag_layout = QFormLayout(rag_tab)
    rag_layout.setContentsMargins(10, 10, 10, 10) # Margins for tab content
    rag_layout.setVerticalSpacing(8) # Space between rows
    rag_layout.setHorizontalSpacing(10) # Space between label/widget

    main_window.rag_model_combo = QComboBox()
    main_window.rag_model_combo.addItems(["gemma", "pali", "gemini", "openrouter", "None"])
    rag_layout.addRow("RAG Model Type:", main_window.rag_model_combo)

    # Gemini RAG Specific (visibility handled in MainWindow)
    main_window.gemini_fetch_button = QPushButton("Fetch Gemini Gen Models")
    main_window.gemini_model_combo = QComboBox()
    main_window.gemini_model_label = QLabel("Select Gemini Gen Model:")
    rag_layout.addRow(main_window.gemini_fetch_button)
    rag_layout.addRow(main_window.gemini_model_label, main_window.gemini_model_combo)

    # OpenRouter RAG Specific (visibility handled in MainWindow)
    main_window.openrouter_fetch_button = QPushButton("Fetch Free Gen Models")
    main_window.openrouter_model_combo = QComboBox()
    main_window.openrouter_model_label = QLabel("Select OpenRouter Gen Model:")
    rag_layout.addRow(main_window.openrouter_fetch_button)
    rag_layout.addRow(main_window.openrouter_model_label, main_window.openrouter_model_combo)

    main_window.personality_input = QLineEdit()
    main_window.personality_input.setPlaceholderText("Optional: e.g., 'scientific', 'cheerful'")
    main_window.personality_label = QLabel("RAG Personality:")
    rag_layout.addRow(main_window.personality_label, main_window.personality_input)

    # Add Output Format Selection
    main_window.output_format_label = QLabel("Output Format:")
    main_window.output_format_combo = QComboBox()
    # Items will be populated from config in MainWindow's load_config method
    rag_layout.addRow(main_window.output_format_label, main_window.output_format_combo)

    config_tabs.addTab(rag_tab, "RAG")

    # Add the tab widget to the main left layout
    left_layout.addWidget(config_tabs)

    # Execution Button (remains below the tabs)
    main_window.run_button = QPushButton("Run Search")
    main_window.run_button.setObjectName("run_button") # Set object name
    # Remove inline style - will be handled by QSS
    # main_window.run_button.setStyleSheet("...")
    left_layout.addWidget(main_window.run_button)

    # Add Progress Bar
    main_window.progress_bar = QProgressBar()
    main_window.progress_bar.setVisible(False) # Initially hidden
    main_window.progress_bar.setRange(0, 100) # Default range, will be set to 0,0 for indeterminate
    main_window.progress_bar.setTextVisible(False) # Hide percentage text
    left_layout.addWidget(main_window.progress_bar)

    # Add Cancel Button
    main_window.cancel_button = QPushButton("Cancel Search")
    main_window.cancel_button.setObjectName("cancel_button") # Set object name
    # Remove inline style - will be handled by QSS
    # main_window.cancel_button.setStyleSheet("...")
    main_window.cancel_button.setVisible(False) # Initially hidden
    main_window.cancel_button.setEnabled(False) # Initially disabled
    left_layout.addWidget(main_window.cancel_button)

    left_layout.addStretch() # Push elements to the top

    # --- Wrap Left Panel in Scroll Area ---
    left_scroll_area = QScrollArea()
    left_scroll_area.setWidget(left_panel)
    left_scroll_area.setWidgetResizable(True)
    left_scroll_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding) # Apply size policy to scroll area

    # --- Right Panel: Status & Results ---
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0) # Let group boxes handle inner margins
    right_layout.setSpacing(10) # Space between status and results groups
    right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


    # Status Log
    status_group = QGroupBox("Status Log")
    status_layout = QVBoxLayout()
    status_layout.setContentsMargins(10, 10, 10, 10) # Margins inside status group
    main_window.status_log = QTextEdit()
    main_window.status_log.setReadOnly(True)
    main_window.status_log.setPlaceholderText("Search progress will appear here...")
    status_layout.addWidget(main_window.status_log)
    status_group.setLayout(status_layout)
    right_layout.addWidget(status_group)


    # Results
    results_group = QGroupBox("Results")
    results_layout = QFormLayout()
    results_layout.setContentsMargins(10, 10, 10, 10) # Margins inside results group
    results_layout.setVerticalSpacing(8) # Space between rows
    results_layout.setHorizontalSpacing(10) # Space between label/widget
    main_window.report_path_label = QLabel("Report will appear here.")
    main_window.report_path_label.setWordWrap(True)
    main_window.report_path_label = QLabel("Report will appear here.")
    main_window.report_path_label.setWordWrap(True)
    main_window.open_report_button = QPushButton("Open Report")
    main_window.open_folder_button = QPushButton("Open Folder") # Shortened text
    main_window.share_email_button = QPushButton("Share") # Shortened text
    # Add Icons for Results Buttons
    report_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_FileIcon)
    folder_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_DirIcon)
    share_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_DialogYesButton) # Using a generic 'action' icon, replace if better exists
    main_window.open_report_button.setIcon(report_icon)
    main_window.open_folder_button.setIcon(folder_icon)
    main_window.share_email_button.setIcon(share_icon)
    # Set Tooltips for icon buttons
    main_window.open_report_button.setToolTip("Open the generated report file")
    main_window.open_folder_button.setToolTip("Open the folder containing the results")
    main_window.share_email_button.setToolTip("Share report via email")

    main_window.open_report_button.setEnabled(False)
    main_window.open_folder_button.setEnabled(False)
    main_window.share_email_button.setEnabled(False) # Initially disabled

    results_layout.addRow("Report Path:", main_window.report_path_label)
    button_layout = QHBoxLayout()
    button_layout.setSpacing(8) # Space between result buttons
    button_layout.addWidget(main_window.open_report_button)
    button_layout.addWidget(main_window.open_folder_button)
    button_layout.addWidget(main_window.share_email_button) # Added email button to layout
    results_layout.addRow(button_layout)

    results_group.setLayout(results_layout)
    right_layout.addWidget(results_group)

    # --- Wrap Right Panel in Scroll Area ---
    right_scroll_area = QScrollArea()
    right_scroll_area.setWidget(right_panel)
    right_scroll_area.setWidgetResizable(True)
    right_scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Apply size policy to scroll area

    # Add panels (wrapped in scroll areas) to main layout
    main_layout.addWidget(left_scroll_area, 2) # Add scroll area instead of panel (Increased stretch)
    main_layout.addWidget(right_scroll_area, 1) # Add scroll area instead of panel (Decreased stretch)

    # --- Initial UI State (handled in MainWindow after setup) ---
    # e.g., setting initial combo box values based on config, hiding/showing elements
