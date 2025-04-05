#!/usr/bin/env python3
# gui/result_display_manager.py

import logging
import os
import re
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QModelIndex
from PyQt6.QtWidgets import QMenu, QInputDialog, QMessageBox
from PyQt6.QtGui import QStandardItemModel, QStandardItem

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
        self.main_window.open_report_button.setEnabled(True)
        self.main_window.open_folder_button.setEnabled(True)
        self.main_window.share_email_button.setEnabled(True)

        # Store the HTML content
        self.current_report_html = final_answer_content

        # 1. Display report content
        self.main_window.results_text_edit.setMarkdown(final_answer_content) # Use setMarkdown for better rendering

        # 2. Populate TOC Tree
        self._populate_toc_tree(toc_tree_nodes)

        # 3. Connect TOC click signal (disconnect first)
        try:
            self.main_window.toc_tree_widget.clicked.disconnect(self._handle_toc_click)
        except TypeError:
            pass # Signal not connected
        self.main_window.toc_tree_widget.clicked.connect(self._handle_toc_click)

        # 4. Setup Context Menu for TOC (disconnect first)
        self.main_window.toc_tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        try:
             self.main_window.toc_tree_widget.customContextMenuRequested.disconnect(self._handle_toc_context_menu)
        except TypeError:
             pass # Signal not connected
        self.main_window.toc_tree_widget.customContextMenuRequested.connect(self._handle_toc_context_menu)

        self.log_status("Results displayed.")

    def _populate_toc_tree(self, toc_nodes):
        """Recursively populates the QTreeView model from TOCNode data."""
        model = self.main_window.toc_tree_widget.model()
        if not isinstance(model, QStandardItemModel):
            model = QStandardItemModel()
            self.main_window.toc_tree_widget.setModel(model)
        model.clear()

        def add_items(parent_item, nodes):
            for node in nodes:
                item = QStandardItem(node.query_text)
                item.setData(node.anchor_id, Qt.ItemDataRole.UserRole)
                item.setEditable(False)
                parent_item.appendRow(item)
                if node.children:
                    add_items(item, node.children)

        add_items(model.invisibleRootItem(), toc_nodes)
        # self.main_window.toc_tree_widget.expandToDepth(0) # Optional: Expand top level

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
            return

        item = self.main_window.toc_tree_widget.model().itemFromIndex(index)
        anchor_id = item.data(Qt.ItemDataRole.UserRole) if item else None

        if not anchor_id:
            return

        menu = QMenu()
        refine_action = menu.addAction("Refine Section...")
        action = menu.exec(self.main_window.toc_tree_widget.viewport().mapToGlobal(position))

        if action == refine_action:
            self._handle_refine_request(anchor_id)

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
        model = self.main_window.toc_tree_widget.model()
        if isinstance(model, QStandardItemModel):
            model.clear()
        self.current_report_html = "" # Clear stored HTML
        # Reset report path labels etc. in MainWindow
        self.main_window.report_path_label.setText("No report generated yet.")
        self.main_window.current_report_path = None
        self.main_window.current_results_dir = None
        self.main_window.open_report_button.setEnabled(False)
        self.main_window.open_folder_button.setEnabled(False)
        self.main_window.share_email_button.setEnabled(False)
