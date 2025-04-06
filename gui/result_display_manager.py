#!/usr/bin/env python3
# gui/result_display_manager.py

import logging
import os
import re
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QModelIndex, QSortFilterProxyModel # Added QSortFilterProxyModel
from PyQt6.QtWidgets import QMenu, QInputDialog, QMessageBox, QApplication, QLineEdit, QPushButton # Added QLineEdit, QPushButton
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QTextDocument, QIcon # Added QTextDocument, QIcon

# Import the new highlighter
from .syntax_highlighter import MarkdownSyntaxHighlighter

class ResultDisplayManager(QObject):
    """
    Handles displaying the report content in the text area and managing
    the interactive Table of Contents (TOC) tree. Also initiates refinement.
    """
    # Signal to request refinement from the main controller/refinement controller
    refinement_requested = pyqtSignal(str, str) # anchor_id, instruction
    # Signal for status updates
    status_update = pyqtSignal(str)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.current_report_html = "" # Store the latest full HTML for refinement

        # --- Model Setup ---
        self.toc_source_model = QStandardItemModel()
        self.toc_proxy_model = QSortFilterProxyModel()
        self.toc_proxy_model.setSourceModel(self.toc_source_model)
        self.toc_proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.toc_proxy_model.setFilterKeyColumn(0) # Filter based on the text in the first column
        self.toc_proxy_model.setRecursiveFilteringEnabled(True) # Filter recursively
        self.main_window.toc_tree_view.setModel(self.toc_proxy_model) # Set proxy model on the view

        # Instantiate the syntax highlighter
        self.highlighter = MarkdownSyntaxHighlighter(self.main_window.results_text_edit.document())

        # Connect filter input signal
        self.main_window.toc_filter_input.textChanged.connect(self._handle_toc_filter_changed)

    def log_status(self, message):
        """Emit status update signal."""
        self.status_update.emit(message)

    def display_results_and_toc(self, report_path, final_answer_content, toc_tree_nodes):
        """
        Displays the final report content and populates the TOC tree.
        Called by the main controller upon successful search completion.
        """
        self.log_status(f"Displaying results for: {report_path}")
        # Update MainWindow state (moved from controller)
        self.main_window.current_report_path = report_path
        self.main_window.current_results_dir = os.path.dirname(report_path)
        self.main_window.report_path_label.setText(report_path)
        # Use the centralized state update method
        self.main_window.update_result_actions_state(True)

        # Store the HTML content
        self.current_report_html = final_answer_content

        # 1. Display report content
        self.main_window.results_text_edit.setMarkdown(final_answer_content) # Use setMarkdown for better rendering

        # 2. Populate TOC Tree
        self._populate_toc_tree(toc_tree_nodes)

        # 3. Connect TOC click signal (disconnect first)
        try:
            self.main_window.toc_tree_view.clicked.disconnect(self._handle_toc_click)
        except TypeError:
            pass # Signal not connected
        self.main_window.toc_tree_view.clicked.connect(self._handle_toc_click)

        # 4. Setup Context Menu for TOC (disconnect first)
        self.main_window.toc_tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        try:
             self.main_window.toc_tree_view.customContextMenuRequested.disconnect(self._handle_toc_context_menu)
        except TypeError:
             pass # Signal not connected
        self.main_window.toc_tree_view.customContextMenuRequested.connect(self._handle_toc_context_menu)

        self.log_status("Results displayed.")

        # 5. Connect Search Controls (disconnect first to avoid duplicates)
        try:
            self.main_window.results_search_input.returnPressed.disconnect(self._handle_find_next)
            self.main_window.results_find_next_button.clicked.disconnect(self._handle_find_next)
            self.main_window.results_find_prev_button.clicked.disconnect(self._handle_find_previous)
        except TypeError:
            pass # Signals not connected yet
        self.main_window.results_search_input.returnPressed.connect(self._handle_find_next)
        self.main_window.results_find_next_button.clicked.connect(self._handle_find_next)
        self.main_window.results_find_prev_button.clicked.connect(self._handle_find_previous)

        # 6. Connect TOC Expand/Collapse Controls (disconnect first)
        try:
            self.main_window.toc_expand_all_button.clicked.disconnect(self._handle_toc_expand_all)
            self.main_window.toc_collapse_all_button.clicked.disconnect(self._handle_toc_collapse_all)
        except TypeError:
            pass # Signals not connected yet
        self.main_window.toc_expand_all_button.clicked.connect(self._handle_toc_expand_all)
        self.main_window.toc_collapse_all_button.clicked.connect(self._handle_toc_collapse_all)


    def _populate_toc_tree(self, toc_nodes):
        """Recursively populates the QTreeView's source model from TOCNode data."""
        # Clear the source model directly
        self.toc_source_model.clear()

        # Define custom data roles (constants for clarity)
        AnchorIdRole = Qt.ItemDataRole.UserRole
        NodeDataRole = Qt.ItemDataRole.UserRole + 1

        def add_items(parent_item, nodes):
            for node in nodes:
                item = QStandardItem(node.query_text)
                # Store anchor ID
                item.setData(node.anchor_id, AnchorIdRole)
                # Store the whole node object for context menu actions
                item.setData(node, NodeDataRole)
                item.setEditable(False)

                # Set Icon based on depth
                style = QApplication.style()
                if node.depth == 1:
                    icon = style.standardIcon(style.StandardPixmap.SP_FileIcon) # Icon for top-level
                else:
                    icon = style.standardIcon(style.StandardPixmap.SP_ArrowRight) # Icon for children
                item.setIcon(icon)

                # Set tooltip with summary snippet
                if node.summary:
                    tooltip_text = node.summary[:150] + ("..." if len(node.summary) > 150 else "")
                    item.setToolTip(tooltip_text)
                else:
                    item.setToolTip("No summary available.")

                parent_item.appendRow(item)
                if node.children:
                    add_items(item, node.children)

        add_items(self.toc_source_model.invisibleRootItem(), toc_nodes)
        # self.main_window.toc_tree_view.expandToDepth(0) # Optional: Expand top level after populating

    def _handle_toc_click(self, index: QModelIndex):
        """Scrolls the results view to the anchor associated with the clicked TOC item."""
        if not index.isValid():
            return

        # Map proxy index to source index
        source_index = self.toc_proxy_model.mapToSource(index)
        if not source_index.isValid():
            return

        item = self.toc_source_model.itemFromIndex(source_index)
        if item:
            # Define custom data roles (constants for clarity) - must match _populate_toc_tree
            AnchorIdRole = Qt.ItemDataRole.UserRole
            anchor_id = item.data(AnchorIdRole)
            if anchor_id:
                self.log_status(f"Navigating to anchor: {anchor_id}")
                self.main_window.results_text_edit.scrollToAnchor(anchor_id)
            else:
                self.log_status(f"No anchor ID found for TOC item: {item.text()}")

    def _handle_toc_context_menu(self, position):
        """Shows a context menu for the TOC tree."""
        index = self.main_window.toc_tree_view.indexAt(position)
        if not index.isValid():
            return

        # Map proxy index to source index
        source_index = self.toc_proxy_model.mapToSource(index)
        if not source_index.isValid():
            return

        # Define custom data roles (constants for clarity) - must match _populate_toc_tree
        AnchorIdRole = Qt.ItemDataRole.UserRole
        NodeDataRole = Qt.ItemDataRole.UserRole + 1

        item = self.toc_source_model.itemFromIndex(source_index)
        if not item:
            return

        # Retrieve data using roles from the source item
        anchor_id = item.data(AnchorIdRole)
        toc_node = item.data(NodeDataRole) # Get the stored TOCNode object

        if not toc_node: # Check if we have the node data
             self.log_status(f"[Warning] Could not retrieve TOCNode data for item: {item.text()}")
             return

        menu = QMenu()

        # Refine Action (existing)
        refine_action = menu.addAction("Refine Section...")
        refine_action.setEnabled(bool(anchor_id)) # Only enable if anchor exists

        menu.addSeparator()

        # Copy Actions
        copy_query_action = menu.addAction("Copy Query Text")
        copy_summary_action = menu.addAction("Copy Section Summary")
        copy_summary_action.setEnabled(bool(toc_node.summary)) # Enable only if summary exists

        # Execute Menu
        action = menu.exec(self.main_window.toc_tree_view.viewport().mapToGlobal(position))

        # Handle Actions
        if action == refine_action and anchor_id:
            self._handle_refine_request(anchor_id)
        elif action == copy_query_action:
            clipboard = QApplication.clipboard()
            clipboard.setText(toc_node.query_text)
            self.log_status(f"Copied query text: '{toc_node.query_text}'")
        elif action == copy_summary_action:
            clipboard = QApplication.clipboard()
            clipboard.setText(toc_node.summary)
            self.log_status(f"Copied summary for query: '{toc_node.query_text}'")

    # --- TOC Filter Handler ---

    def _handle_toc_filter_changed(self, text):
        """Applies the filter text to the ToC proxy model."""
        self.toc_proxy_model.setFilterRegularExpression(text)
        # Optional: Automatically expand items when filtering?
        # if text:
        #     self.main_window.toc_tree_view.expandAll()
        # else:
        #     self.main_window.toc_tree_view.collapseAll() # Or restore previous state

    # --- TOC Expand/Collapse Handlers ---

    def _handle_toc_expand_all(self):
        """Expands all items in the TOC tree."""
        self.log_status("Expanding all TOC items.")
        self.main_window.toc_tree_view.expandAll()

    def _handle_toc_collapse_all(self):
        """Collapses all items in the TOC tree."""
        self.log_status("Collapsing all TOC items.")
        self.main_window.toc_tree_view.collapseAll()

    # --- Search Within Results Handlers ---

    def _handle_find_next(self):
        """Finds the next occurrence of the text in the results_search_input."""
        search_text = self.main_window.results_search_input.text()
        if not search_text:
            self.log_status("Find Next: No search text entered.")
            return

        found = self.main_window.results_text_edit.find(search_text)
        if not found:
            self.log_status(f"Find Next: Text '{search_text}' not found (reached end).")
            # Optional: Move cursor to the beginning to wrap search
            cursor = self.main_window.results_text_edit.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.main_window.results_text_edit.setTextCursor(cursor)
            # Try finding again from the start
            found_again = self.main_window.results_text_edit.find(search_text)
            if not found_again:
                 self.log_status(f"Find Next: Text '{search_text}' not found anywhere.")


    def _handle_find_previous(self):
        """Finds the previous occurrence of the text in the results_search_input."""
        search_text = self.main_window.results_search_input.text()
        if not search_text:
            self.log_status("Find Previous: No search text entered.")
            return

        found = self.main_window.results_text_edit.find(search_text, QTextDocument.FindFlag.FindBackward)
        if not found:
            self.log_status(f"Find Previous: Text '{search_text}' not found (reached beginning).")
            # Optional: Move cursor to the end to wrap search
            cursor = self.main_window.results_text_edit.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.main_window.results_text_edit.setTextCursor(cursor)
             # Try finding again from the end (backwards)
            found_again = self.main_window.results_text_edit.find(search_text, QTextDocument.FindFlag.FindBackward)
            if not found_again:
                 self.log_status(f"Find Previous: Text '{search_text}' not found anywhere.")

    # --- End Search Within Results Handlers ---


    def _handle_refine_request(self, anchor_id):
        """Handles the 'Refine Section' action by emitting a signal."""
        self.log_status(f"Refine requested for section with anchor: {anchor_id}")

        # Get User Instructions
        instruction, ok = QInputDialog.getText(self.main_window, "Refine Section",
                                               "How should this section be refined?\n(e.g., 'Summarize this', 'Make it simpler', 'Add bullet points')")
        if not ok or not instruction.strip():
            self.log_status("Refinement cancelled by user.")
            return

        # Emit signal to the main controller / refinement controller
        self.refinement_requested.emit(anchor_id, instruction)

    def update_refined_section(self, anchor_id, refined_content):
        """
        Updates the results display with the refined content for a specific anchor.
        Called by the main controller/refinement controller after successful refinement.
        """
        self.log_status(f"Updating display for refined section '{anchor_id}'...")

        # Use the stored full HTML from the last successful display
        current_full_html = self.current_report_html

        # Find the original segment using the anchor ID
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

        start_replace_index = start_match.start()
        if next_match:
            end_replace_index = next_match.start()
        else:
            end_replace_index = len(current_full_html) # Replace to the end

        # Construct the replacement HTML (Original Anchor + Refined Content)
        replacement_html = original_anchor_tag + "\n" + refined_content

        # Create the new full HTML
        new_full_html = current_full_html[:start_replace_index] + replacement_html + current_full_html[end_replace_index:]

        # Update the stored HTML and the display
        self.current_report_html = new_full_html
        self.main_window.results_text_edit.setHtml(new_full_html) # Use setHtml
        self.log_status(f"Display updated for refined section '{anchor_id}'.")

    def clear_results(self):
        """Clears the results text edit and TOC."""
        self.log_status("Clearing results display.")
        self.main_window.results_text_edit.clear()
        # Clear the source model, proxy model updates automatically
        self.toc_source_model.clear()
        self.main_window.toc_filter_input.clear() # Clear filter input as well
        self.current_report_html = "" # Clear stored HTML
        # Reset report path labels etc. in MainWindow
        self.main_window.report_path_label.setText("No report generated yet.")
        self.main_window.current_report_path = None
        self.main_window.current_results_dir = None
        # Use the centralized state update method
        self.main_window.update_result_actions_state(False)
