#!/usr/bin/env python3
# gui/ui_components/searxng_selector.py

"""
Custom QWidget for selecting SearXNG engines using a hierarchical tree view.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QGroupBox, QLabel, QTreeWidgetItemIterator
)
from PyQt6.QtCore import Qt, pyqtSignal
from gui.searxng_engines import SEARXNG_ENGINE_DATA

class SearxngEngineSelector(QWidget):
    """
    A widget for displaying and selecting SearXNG engines hierarchically.
    """
    # Signal emitted when the selection changes, passing the list of selected engine keys
    selectionChanged = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._engine_key_map = {} # Maps engine_key to QTreeWidgetItem
        # self._item_key_map removed as it's redundant with the new _engine_key_map structure
        self._setup_ui()
        self._populate_tree()

    def _setup_ui(self):
        """Sets up the UI elements of the widget."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # Use container margins

        # --- Filter and Actions ---
        filter_layout = QHBoxLayout()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter engines by name, !bang, or category...")
        self.filter_input.textChanged.connect(self._filter_tree)
        filter_layout.addWidget(self.filter_input)

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        filter_layout.addWidget(self.reset_button)

        # TODO: Add more bulk actions (Select/Deselect All, etc.) if needed

        main_layout.addLayout(filter_layout)

        # --- Tree Widget ---
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("SearXNG Engines")
        self.tree_widget.itemChanged.connect(self._handle_item_change) # Connect signal for checkbox changes
        main_layout.addWidget(self.tree_widget)

    def _populate_tree(self):
        """Populates the tree widget with SearXNG engine data."""
        self.tree_widget.clear()
        self._engine_key_map.clear()
        # self._item_key_map.clear() # Removed

        for tab_name, tab_data in SEARXNG_ENGINE_DATA.items():
            tab_item = QTreeWidgetItem(self.tree_widget, [f"Tab: {tab_name}"])
            tab_item.setFlags(tab_item.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
            tab_item.setCheckState(0, Qt.CheckState.Unchecked) # Start unchecked

            # Engines directly under the tab
            for name, bang, default_enabled in tab_data.get("engines", []):
                engine_key = self._get_engine_key(name)
                engine_item = QTreeWidgetItem(tab_item, [f"{name} ({bang})"])
                engine_item.setFlags(engine_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                engine_item.setCheckState(0, Qt.CheckState.Unchecked) # Start unchecked
                self._engine_key_map[engine_key] = engine_item
                # self._item_key_map[engine_key] = engine_item # Removed

            # Engines under groups
            for group_name, group_engines in tab_data.get("groups", {}).items():
                group_item = QTreeWidgetItem(tab_item, [f"Group: {group_name}"])
                group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                group_item.setCheckState(0, Qt.CheckState.Unchecked) # Start unchecked

                for name, bang, default_enabled in group_engines:
                    engine_key = self._get_engine_key(name)
                    engine_item = QTreeWidgetItem(group_item, [f"{name} ({bang})"])
                    engine_item.setFlags(engine_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    engine_item.setCheckState(0, Qt.CheckState.Unchecked) # Start unchecked
                    self._engine_key_map[engine_key] = engine_item
                    # self._item_key_map[engine_key] = engine_item # Removed

        self.tree_widget.expandAll() # Expand initially

    def _get_engine_key(self, name):
        """Derives the engine key used in SearXNG API calls from the display name."""
        key = name.split(" ")[0].lower()
        if "(" in key:
            key = key.split("(")[0]
        return key

    def _handle_item_change(self, item, column):
        """Handles changes in checkbox state, propagating up and down the hierarchy."""
        if column == 0:
            # Block signals temporarily to prevent recursion during updates
            self.tree_widget.blockSignals(True)

            check_state = item.checkState(0)

            # Update children if parent state changed
            if item.childCount() > 0:
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.checkState(0) != check_state:
                        child.setCheckState(0, check_state)

            # Update parent state based on children
            parent = item.parent()
            if parent:
                all_checked = True
                any_checked = False
                for i in range(parent.childCount()):
                    child = parent.child(i)
                    child_state = child.checkState(0)
                    if child_state != Qt.CheckState.Checked:
                        all_checked = False
                    if child_state != Qt.CheckState.Unchecked:
                        any_checked = True

                if all_checked:
                    parent.setCheckState(0, Qt.CheckState.Checked)
                elif any_checked:
                    parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
                else:
                    parent.setCheckState(0, Qt.CheckState.Unchecked)

            # Unblock signals and emit the change
            self.tree_widget.blockSignals(False)
            self._emit_selection_change() # Emit signal after state stabilizes

    def _filter_tree(self, text):
        """Filters the tree view based on the input text."""
        search_term = text.lower().strip()
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            item = iterator.value()
            item_text = item.text(0).lower()
            # Check if the item itself or any ancestor matches
            matches = search_term in item_text
            parent = item.parent()
            while parent and not matches:
                 matches = search_term in parent.text(0).lower()
                 parent = parent.parent()

            # Also check children - if a child matches, the parent should be visible
            child_matches = False
            if item.childCount() > 0 and not matches:
                 child_iterator = QTreeWidgetItemIterator(item, QTreeWidgetItemIterator.IteratorFlag.All)
                 while child_iterator.value():
                     child = child_iterator.value()
                     if search_term in child.text(0).lower():
                         child_matches = True
                         break
                     child_iterator += 1


            item.setHidden(not (matches or child_matches))
            iterator += 1


    def _reset_to_defaults(self):
        """Sets the checkboxes based on the default enabled engines."""
        default_keys = set()
        for tab_data in SEARXNG_ENGINE_DATA.values():
            for group_engines in tab_data.get("groups", {}).values():
                for name, bang, enabled in group_engines:
                    if enabled:
                        default_keys.add(self._get_engine_key(name))
            for name, bang, enabled in tab_data.get("engines", []):
                if enabled:
                    default_keys.add(self._get_engine_key(name))

        self.setSelectedEngines(list(default_keys))

    def _emit_selection_change(self):
        """Emits the selectionChanged signal with the current list of selected engine keys."""
        self.selectionChanged.emit(self.getSelectedEngines())

    # --- Public Methods ---
    def getSelectedEngines(self):
        """Returns a list of engine keys for the currently checked engine items."""
        selected_keys = []
        # Iterate through the map where key is engine_key and value is item
        for key, item in self._engine_key_map.items():
            # Check only actual engine items (not group/tab items which aren't in the map)
            if item.childCount() == 0 and item.checkState(0) == Qt.CheckState.Checked:
                selected_keys.append(key)
        return sorted(list(set(selected_keys))) # Return unique sorted list

    def setSelectedEngines(self, engine_keys_to_select):
        """
        Sets the check state of the tree items based on the provided list of engine keys.
        """
        self.tree_widget.blockSignals(True) # Block signals during bulk update

        selected_set = set(engine_keys_to_select)

        # First, uncheck all engine items by iterating through the values (items) of the map
        for item in self._engine_key_map.values():
             item.setCheckState(0, Qt.CheckState.Unchecked)

        # Then, check the selected ones using the key to find the item in the map
        for key in selected_set:
            if key in self._engine_key_map:
                self._engine_key_map[key].setCheckState(0, Qt.CheckState.Checked)

        # Update parent states after all children are set
        iterator = QTreeWidgetItemIterator(self.tree_widget, QTreeWidgetItemIterator.IteratorFlag.All)
        processed_parents = set()
        while iterator.value():
            item = iterator.value()
            parent = item.parent()
            # Process parent updates only once after all its children are potentially set
            if parent and id(parent) not in processed_parents:
                all_checked = True
                any_checked = False
                for i in range(parent.childCount()):
                    child = parent.child(i)
                    child_state = child.checkState(0)
                    if child_state != Qt.CheckState.Checked:
                        all_checked = False
                    if child_state != Qt.CheckState.Unchecked:
                        any_checked = True

                if all_checked:
                    parent.setCheckState(0, Qt.CheckState.Checked)
                elif any_checked:
                    parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
                else:
                    parent.setCheckState(0, Qt.CheckState.Unchecked)
                processed_parents.add(id(parent)) # Mark parent as processed
            iterator += 1


        self.tree_widget.blockSignals(False) # Unblock signals
        self._emit_selection_change() # Emit signal after update


if __name__ == '__main__':
    # Example usage for testing the widget standalone
    from PyQt6.QtWidgets import QApplication, QMainWindow
    import sys

    app = QApplication(sys.argv)
    window = QMainWindow()
    selector = SearxngEngineSelector()

    # Example: Connect signal to a slot
    def on_selection_changed(selected_list):
        print("Selection changed:", selected_list)
    selector.selectionChanged.connect(on_selection_changed)

    # Example: Set initial selection
    initial_selection = ["google", "wikipedia", "duckduckgo", "ddg"] # ddg is duplicate, handled by set
    print("Setting initial selection:", initial_selection)
    selector.setSelectedEngines(initial_selection)

    # Example: Get current selection after setting
    print("Current selection after setting:", selector.getSelectedEngines())


    container = QGroupBox("SearXNG Engine Selection")
    layout = QVBoxLayout()
    layout.addWidget(selector)
    container.setLayout(layout)

    window.setCentralWidget(container)
    window.setWindowTitle("SearXNG Engine Selector Test")
    window.setGeometry(100, 100, 600, 700)
    window.show()

    # Trigger reset after a delay for testing
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(2000, lambda: print("\nResetting to defaults..."))
    QTimer.singleShot(2000, selector._reset_to_defaults)


    sys.exit(app.exec())
