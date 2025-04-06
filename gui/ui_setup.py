#!/usr/bin/env python3
# gui/ui_setup.py

"""
Module responsible for setting up the UI elements of the MainWindow.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QScrollArea, QTreeWidgetItemIterator,
    QSizePolicy, QProgressBar, QTabWidget, QApplication, # Added QTabWidget and QApplication
    QTreeView, QSplitter # Added QTreeView and QSplitter
)
from PyQt6.QtGui import QIcon, QStandardItemModel # Import QIcon, QStandardItemModel
from PyQt6.QtCore import QSize, Qt # Import QSize for icon sizing, Qt for splitter orientation

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
    main_window.corpus_clear_button = QPushButton("Clear") # New Clear button
    clear_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_DialogCancelButton) # Use a cancel/clear icon
    main_window.corpus_clear_button.setIcon(clear_icon)
    main_window.corpus_clear_button.setToolTip("Clear the selected corpus directory path") # Add tooltip
    corpus_layout.addWidget(main_window.corpus_dir_label)
    corpus_layout.addWidget(main_window.corpus_dir_button)
    corpus_layout.addWidget(main_window.corpus_clear_button) # Add clear button to layout
    general_layout.addRow("Local Corpus:", corpus_layout)

    main_window.max_depth_spinbox = QSpinBox()
    main_window.max_depth_spinbox.setRange(0, 10)
    main_window.max_depth_spinbox.setValue(1) # Default value
    general_layout.addRow("Max Recursion Depth:", main_window.max_depth_spinbox)

    main_window.top_k_spinbox = QSpinBox()
    main_window.top_k_spinbox.setRange(1, 50)
    main_window.top_k_spinbox.setValue(3) # Default value
    general_layout.addRow("Top K Results:", main_window.top_k_spinbox)

    # --- Cache Settings ---
    main_window.cache_enabled_checkbox = QCheckBox("Enable Caching (Web/Embed/Summary)")
    main_window.cache_enabled_checkbox.setToolTip("Check to use cached results for web downloads, embeddings, and summaries to speed up subsequent runs.")
    general_layout.addRow(main_window.cache_enabled_checkbox)

    main_window.clear_cache_button = QPushButton("Clear Cache")
    main_window.clear_cache_button.setToolTip("Deletes the cache database file.")
    # Add Icon for Clear Cache
    clear_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_TrashIcon)
    main_window.clear_cache_button.setIcon(clear_icon)
    general_layout.addRow(main_window.clear_cache_button)
    # --- End Cache Settings ---

    config_tabs.addTab(general_tab, "General")

    # -- Search Tab --
    search_tab = QWidget()
    # Main horizontal layout for the two columns
    search_main_layout = QHBoxLayout(search_tab)
    search_main_layout.setContentsMargins(10, 10, 10, 10)
    search_main_layout.setSpacing(10)

    # Left Column (Group Box with Form Layout for settings)
    left_column_group = QGroupBox("Search Settings")
    left_column_layout = QFormLayout(left_column_group) # Set layout for the group box
    # left_column_layout.setContentsMargins(10, 10, 10, 10) # Use default group box margins
    left_column_layout.setVerticalSpacing(8)
    left_column_layout.setHorizontalSpacing(10)

    main_window.search_provider_combo = QComboBox()
    main_window.search_provider_combo.addItems(["DuckDuckGo", "SearXNG"])
    left_column_layout.addRow("Search Provider:", main_window.search_provider_combo)

    # Iterative Search Checkbox
    main_window.iterative_search_checkbox = QCheckBox("Enable Iterative Search (Experimental)")
    left_column_layout.addRow(main_window.iterative_search_checkbox) # Add checkbox directly

    # Include Images/Maps Checkbox
    main_window.include_visuals_checkbox = QCheckBox("Include Images/Maps in Report")
    main_window.include_visuals_checkbox.setToolTip("Check this to instruct the LLM to include relevant images and static maps (using OpenStreetMap) in the final report.")
    left_column_layout.addRow(main_window.include_visuals_checkbox) # Add the new checkbox

    # SearXNG Specific Settings (visibility handled in MainWindow)
    main_window.searxng_base_url_label = QLabel("SearXNG URL:")
    main_window.searxng_base_url_input = QLineEdit()
    main_window.searxng_time_range_label = QLabel("Time Range:")
    main_window.searxng_time_range_input = QLineEdit()
    main_window.searxng_time_range_input.setPlaceholderText("Optional: day, week, month, year")
    main_window.searxng_categories_label = QLabel("Categories:")
    main_window.searxng_categories_input = QLineEdit()
    main_window.searxng_categories_input.setPlaceholderText("Optional: general,images,...")

    left_column_layout.addRow(main_window.searxng_base_url_label, main_window.searxng_base_url_input)
    left_column_layout.addRow(main_window.searxng_time_range_label, main_window.searxng_time_range_input)
    left_column_layout.addRow(main_window.searxng_categories_label, main_window.searxng_categories_input)

    # Right Column (Widget with Vertical Layout for Engine Selector)
    # Note: We keep the right column structure similar to the previous attempt
    right_column_widget = QWidget()
    right_column_layout = QVBoxLayout(right_column_widget)
    right_column_layout.setContentsMargins(0, 0, 0, 0) # No inner margins for the VBox itself
    right_column_layout.setSpacing(8)

    # SearXNG Engine Selection Widget
    main_window.searxng_engine_group = QGroupBox("SearXNG Engine Selection")
    searxng_engine_layout = QVBoxLayout() # Use QVBoxLayout for the group
    searxng_engine_layout.setContentsMargins(5, 5, 5, 5) # Slightly tighter margins for this specific group
    main_window.searxng_engine_selector = SearxngEngineSelector()
    searxng_engine_layout.addWidget(main_window.searxng_engine_selector)
    main_window.searxng_engine_group.setLayout(searxng_engine_layout)
    main_window.searxng_engine_group.setMinimumWidth(200) # Keep minimum width
    right_column_layout.addWidget(main_window.searxng_engine_group)
    right_column_layout.addStretch() # Push engine selector to the top

    # Add columns (left group box, right widget) to the main search tab layout
    search_main_layout.addWidget(left_column_group, 1) # Add the group box (stretch = 1)
    search_main_layout.addWidget(right_column_widget, 1) # Add the widget (stretch = 1)

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

    # -- Results Tab --
    results_tab_widget = QWidget()
    results_tab_layout = QVBoxLayout(results_tab_widget) # Main layout for the results tab
    results_tab_layout.setContentsMargins(10, 10, 10, 10)
    results_tab_layout.setSpacing(8)

    # --- Add Search Controls (Moved from Right Panel) ---
    search_controls_layout = QHBoxLayout()
    search_controls_layout.setSpacing(5)
    main_window.results_search_input = QLineEdit()
    main_window.results_search_input.setPlaceholderText("Search in results...")
    main_window.results_find_prev_button = QPushButton("Previous")
    main_window.results_find_next_button = QPushButton("Next")
    # Add Icons for Find buttons
    prev_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_ArrowUp)
    next_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_ArrowDown)
    main_window.results_find_prev_button.setIcon(prev_icon)
    main_window.results_find_next_button.setIcon(next_icon)
    main_window.results_find_prev_button.setToolTip("Find previous occurrence")
    main_window.results_find_next_button.setToolTip("Find next occurrence")

    search_controls_layout.addWidget(QLabel("Find:")) # Simple label
    search_controls_layout.addWidget(main_window.results_search_input, 1) # Input stretches
    search_controls_layout.addWidget(main_window.results_find_prev_button)
    search_controls_layout.addWidget(main_window.results_find_next_button)
    results_tab_layout.addLayout(search_controls_layout) # Add to results tab layout
    # --- End Search Controls ---

    # --- Splitter for TOC and Results Text (Moved from Right Panel) ---
    main_window.results_splitter = QSplitter(Qt.Orientation.Horizontal)

    # --- Left side of Splitter (TOC + Controls) ---
    toc_container_widget = QWidget()
    toc_container_layout = QVBoxLayout(toc_container_widget)
    toc_container_layout.setContentsMargins(0, 0, 0, 0) # No margins for the container layout
    toc_container_layout.setSpacing(5) # Space between filter, buttons, and tree

    # TOC Filter Input
    main_window.toc_filter_input = QLineEdit()
    main_window.toc_filter_input.setPlaceholderText("Filter ToC...")
    main_window.toc_filter_input.setClearButtonEnabled(True) # Add a clear button
    toc_container_layout.addWidget(main_window.toc_filter_input) # Add filter input first

    # TOC Control Buttons
    toc_buttons_layout = QHBoxLayout()
    toc_buttons_layout.setSpacing(5)
    main_window.toc_expand_all_button = QPushButton("Expand All")
    main_window.toc_collapse_all_button = QPushButton("Collapse All")
    # Add Icons (using standard icons if available, might need custom ones)
    expand_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_ArrowDown) # Placeholder icon
    collapse_icon = QApplication.style().standardIcon(QApplication.style().StandardPixmap.SP_ArrowUp) # Placeholder icon
    main_window.toc_expand_all_button.setIcon(expand_icon)
    main_window.toc_collapse_all_button.setIcon(collapse_icon)
    main_window.toc_expand_all_button.setToolTip("Expand all items in the Table of Contents")
    main_window.toc_collapse_all_button.setToolTip("Collapse all items in the Table of Contents")
    toc_buttons_layout.addWidget(main_window.toc_expand_all_button)
    toc_buttons_layout.addWidget(main_window.toc_collapse_all_button)
    toc_buttons_layout.addStretch() # Push buttons left

    toc_container_layout.addLayout(toc_buttons_layout) # Add button layout to container

    # TOC Tree Widget (Use the instance created in MainWindow)
    # main_window.toc_tree_widget = QTreeView() # Removed: Instance created in MainWindow
    # main_window.toc_tree_widget.setHeaderHidden(True) # Removed: Properties set in MainWindow
    # main_window.toc_tree_widget.setModel(QStandardItemModel()) # Removed: Model set in MainWindow
    main_window.toc_tree_view.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding) # Keep size policy setting if needed
    toc_container_layout.addWidget(main_window.toc_tree_view) # Add the instance from MainWindow

    main_window.results_splitter.addWidget(toc_container_widget) # Add container to splitter
    # --- End Left side of Splitter ---

    # --- Right side of Splitter (Results Text) ---
    # Results Text Edit
    main_window.results_text_edit = QTextEdit()
    main_window.results_text_edit.setReadOnly(True)
    main_window.results_text_edit.setPlaceholderText("Generated report content will appear here...")
    main_window.results_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    main_window.results_splitter.addWidget(main_window.results_text_edit)

    # Set initial splitter sizes (e.g., 1/4 for TOC, 3/4 for results)
    main_window.results_splitter.setSizes([150, 450]) # Adjust initial sizes as needed

    results_tab_layout.addWidget(main_window.results_splitter) # Add splitter to results tab layout
    # --- End Splitter ---

    # --- Report Path and Action Buttons (Moved from Right Panel) ---
    report_actions_layout = QFormLayout() # Use FormLayout for path label + buttons row
    report_actions_layout.setContentsMargins(0, 5, 0, 0) # Add some top margin
    report_actions_layout.setVerticalSpacing(8)
    report_actions_layout.setHorizontalSpacing(10)

    main_window.report_path_label = QLabel("Report path will appear here.")
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

    report_actions_layout.addRow("Report Path:", main_window.report_path_label)
    button_layout = QHBoxLayout()
    button_layout.setSpacing(8) # Space between result buttons
    button_layout.addWidget(main_window.open_report_button)
    button_layout.addWidget(main_window.open_folder_button)
    button_layout.addWidget(main_window.share_email_button) # Added email button to layout
    report_actions_layout.addRow(button_layout) # Add button row to the form layout

    results_tab_layout.addLayout(report_actions_layout) # Add the actions layout to results tab layout
    # --- End Report Path and Action Buttons ---

    config_tabs.addTab(results_tab_widget, "Results") # Add the new Results tab

    # Add the tab widget to the main left layout
    left_layout.addWidget(config_tabs)

    # Execution Button (remains below the tabs)
    main_window.run_button = QPushButton("Run Search")
    main_window.run_button.setObjectName("run_button") # Set object name
    # Remove inline style - will be handled by QSS
    # main_window.run_button.setStyleSheet("...")
    left_layout.addWidget(main_window.run_button)

    # --- Progress Visualization Area ---
    progress_group = QGroupBox("Progress")
    progress_layout = QVBoxLayout()
    progress_layout.setContentsMargins(5, 8, 5, 8) # Tighter margins for progress group
    progress_layout.setSpacing(5) # Space between progress elements

    # Add Status Label
    main_window.status_label = QLabel("Idle.")
    main_window.status_label.setObjectName("status_label") # Set object name for styling
    main_window.status_label.setWordWrap(True) # Allow wrapping for longer messages
    progress_layout.addWidget(main_window.status_label)

    # Add Overall Progress Bar (existing one)
    main_window.progress_bar = QProgressBar()
    main_window.progress_bar.setObjectName("overall_progress_bar")
    main_window.progress_bar.setVisible(False) # Initially hidden
    main_window.progress_bar.setRange(0, 100) # Default range, will be set to 0,0 for indeterminate
    main_window.progress_bar.setTextVisible(True) # Show percentage for overall
    main_window.progress_bar.setFormat("Overall: %p%") # Set format
    progress_layout.addWidget(main_window.progress_bar)

    # Add Phase-Specific Progress Bars
    main_window.phase_progress_bars = {}
    phases = {
        "web_search": "Web Search",
        "embedding": "Embedding",
        "summarization": "Summarization",
        "reporting": "Reporting"
        # Add more phases if needed (e.g., local_corpus, iterative_search)
    }
    for phase_key, phase_name in phases.items():
        bar = QProgressBar()
        bar.setObjectName(f"progress_bar_{phase_key}")
        bar.setFormat(f"{phase_name}: %p%")
        bar.setTextVisible(True)
        bar.setVisible(False) # Initially hidden
        bar.setRange(0, 100)
        bar.setValue(0)
        progress_layout.addWidget(bar)
        main_window.phase_progress_bars[phase_key] = bar

    progress_group.setLayout(progress_layout)
    left_layout.addWidget(progress_group) # Add the group box to the left layout
    # --- End Progress Visualization Area ---


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

    # --- Right Panel: Status Log Only ---
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0) # Let group boxes handle inner margins
    right_layout.setSpacing(10) # Space between status and results groups (now just status)
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
    right_layout.addWidget(status_group) # Status log remains on the right

    # Results Area (TOC and Content) - MOVED TO RESULTS TAB on Left Panel

    # --- Right Panel Scroll Area Removed ---

    # Add panels directly to main layout with adjusted stretch
    main_layout.addWidget(left_scroll_area, 3) # Give left panel more stretch (3)
    main_layout.addWidget(right_panel, 1)      # Add right panel directly (stretch 1)

    # --- Initial UI State (handled in MainWindow after setup) ---
    # e.g., setting initial combo box values based on config, hiding/showing elements
