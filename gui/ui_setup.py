#!/usr/bin/env python3
# gui/ui_setup.py

"""
Module responsible for setting up the UI elements of the MainWindow.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QScrollArea, # Added QScrollArea
    QSizePolicy, QProgressBar # Added QProgressBar
)
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

    # --- Left Panel: Configuration ---
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # Query Input
    query_group = QGroupBox("Search Query")
    query_layout = QVBoxLayout()
    main_window.query_input = QTextEdit()
    main_window.query_input.setPlaceholderText("Enter your research query here...")
    query_layout.addWidget(main_window.query_input)
    query_group.setLayout(query_layout)
    left_layout.addWidget(query_group)

    # Configuration Options
    config_group = QGroupBox("Configuration")
    config_layout = QFormLayout()

    main_window.web_search_checkbox = QCheckBox("Enable Web Search")
    config_layout.addRow(main_window.web_search_checkbox)

    corpus_layout = QHBoxLayout()
    main_window.corpus_dir_label = QLineEdit()
    main_window.corpus_dir_label.setPlaceholderText("Optional: Path to local documents")
    main_window.corpus_dir_label.setReadOnly(True)
    main_window.corpus_dir_button = QPushButton("Browse...")
    corpus_layout.addWidget(main_window.corpus_dir_label)
    corpus_layout.addWidget(main_window.corpus_dir_button)
    config_layout.addRow("Local Corpus:", corpus_layout)

    main_window.max_depth_spinbox = QSpinBox()
    main_window.max_depth_spinbox.setRange(0, 10)
    main_window.max_depth_spinbox.setValue(1) # Default value
    config_layout.addRow("Max Recursion Depth:", main_window.max_depth_spinbox)

    main_window.top_k_spinbox = QSpinBox()
    main_window.top_k_spinbox.setRange(1, 50)
    main_window.top_k_spinbox.setValue(3) # Default value
    config_layout.addRow("Top K Results:", main_window.top_k_spinbox)

    # --- Search Provider Configuration ---
    search_group = QGroupBox("Search Settings")
    search_layout = QFormLayout()
    main_window.search_provider_combo = QComboBox()
    main_window.search_provider_combo.addItems(["DuckDuckGo", "SearXNG"])
    # Initial value set from config in MainWindow.__init__ after this setup
    search_layout.addRow("Search Provider:", main_window.search_provider_combo)

    # --- SearXNG Specific Settings (visibility handled in MainWindow) ---
    # Initial values set from config in MainWindow.__init__ after this setup
    main_window.searxng_base_url_label = QLabel("SearXNG URL:")
    main_window.searxng_base_url_input = QLineEdit()
    main_window.searxng_time_range_label = QLabel("Time Range:")
    main_window.searxng_time_range_input = QLineEdit()
    main_window.searxng_time_range_input.setPlaceholderText("Optional: day, week, month, year")
    main_window.searxng_categories_label = QLabel("Categories:")
    main_window.searxng_categories_input = QLineEdit()
    main_window.searxng_categories_input.setPlaceholderText("Optional: general,images,...")
    main_window.searxng_engines_label = QLabel("Engines:")
    main_window.searxng_engines_input = QLineEdit()
    main_window.searxng_engines_input.setPlaceholderText("Optional: google,bing,!wikipedia")

    search_layout.addRow(main_window.searxng_base_url_label, main_window.searxng_base_url_input)
    search_layout.addRow(main_window.searxng_time_range_label, main_window.searxng_time_range_input)
    search_layout.addRow(main_window.searxng_categories_label, main_window.searxng_categories_input)
    search_layout.addRow(main_window.searxng_engines_label, main_window.searxng_engines_input)

    search_group.setLayout(search_layout)
    config_layout.addRow(search_group)


    # --- Embedding Configuration ---
    embedding_group = QGroupBox("Embedding Settings")
    embedding_layout = QFormLayout()

    main_window.device_combo = QComboBox()
    main_window.device_combo.addItems(["cpu", "cuda"])
    embedding_layout.addRow("Embedding Device:", main_window.device_combo)

    main_window.embedding_model_combo = QComboBox()
    # Initially populate based on default device ('cpu') - handled in MainWindow logic
    main_window.embedding_model_label = QLabel("Embedding Model:")
    embedding_layout.addRow(main_window.embedding_model_label, main_window.embedding_model_combo)

    embedding_group.setLayout(embedding_layout)
    config_layout.addRow(embedding_group)

    # --- RAG Configuration ---
    rag_group = QGroupBox("RAG Settings")
    rag_layout = QFormLayout()

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
    # Initial visibility handled in MainWindow logic

    rag_group.setLayout(rag_layout)
    config_layout.addRow(rag_group)

    config_group.setLayout(config_layout)
    left_layout.addWidget(config_group)


    # Execution Button
    main_window.run_button = QPushButton("Run Search")
    main_window.run_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #cccccc; }")
    left_layout.addWidget(main_window.run_button)

    # Add Progress Bar
    main_window.progress_bar = QProgressBar()
    main_window.progress_bar.setVisible(False) # Initially hidden
    main_window.progress_bar.setRange(0, 100) # Default range, will be set to 0,0 for indeterminate
    main_window.progress_bar.setTextVisible(False) # Hide percentage text
    left_layout.addWidget(main_window.progress_bar)

    # Add Cancel Button
    main_window.cancel_button = QPushButton("Cancel Search")
    main_window.cancel_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 8px; border-radius: 4px; } QPushButton:hover { background-color: #da190b; } QPushButton:disabled { background-color: #cccccc; }")
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
    right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


    # Status Log
    status_group = QGroupBox("Status Log")
    status_layout = QVBoxLayout()
    main_window.status_log = QTextEdit()
    main_window.status_log.setReadOnly(True)
    main_window.status_log.setPlaceholderText("Search progress will appear here...")
    status_layout.addWidget(main_window.status_log)
    status_group.setLayout(status_layout)
    right_layout.addWidget(status_group)


    # Results
    results_group = QGroupBox("Results")
    results_layout = QFormLayout()
    main_window.report_path_label = QLabel("Report will appear here.")
    main_window.report_path_label.setWordWrap(True)
    main_window.open_report_button = QPushButton("Open Report")
    main_window.open_folder_button = QPushButton("Open Results Folder")
    main_window.open_report_button.setEnabled(False)
    main_window.open_folder_button.setEnabled(False)

    results_layout.addRow("Report Path:", main_window.report_path_label)
    button_layout = QHBoxLayout()
    button_layout.addWidget(main_window.open_report_button)
    button_layout.addWidget(main_window.open_folder_button)
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
