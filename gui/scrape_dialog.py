#!/usr/bin/env python3
# gui/scrape_dialog.py

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox,
    QPushButton, QDialogButtonBox, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt

class ScrapeDialog(QDialog):
    """
    A dialog window for users to input URL and scraping options.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scrape Web Page")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # URL Input
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://example.com/page_to_scrape")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)

        # Options
        self.ignore_robots_checkbox = QCheckBox("Ignore robots.txt (Use with caution)")
        self.ignore_robots_checkbox.setToolTip(
            "Check this to scrape the URL even if disallowed by the website's robots.txt file.\n"
            "Please be respectful of website policies."
        )
        layout.addWidget(self.ignore_robots_checkbox)

        # Recursion Depth Input
        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setRange(0, 5) # Allow depth 0 (single page) up to 5 levels
        self.depth_spinbox.setValue(0) # Default to single page
        self.depth_spinbox.setToolTip("Set the maximum number of links to follow from the initial URL.\n0 means only the initial URL will be scraped.")
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Recursion Depth (0 = single page):"))
        depth_layout.addWidget(self.depth_spinbox)
        layout.addLayout(depth_layout)

        # Dialog Buttons (OK, Cancel)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.url_input.setFocus() # Set focus to URL input initially

    def get_values(self):
        """Returns the entered URL and options."""
        url = self.url_input.text().strip()
        ignore_robots = self.ignore_robots_checkbox.isChecked()
        depth = self.depth_spinbox.value()

        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a URL to scrape.")
            return None

        # Basic URL validation (can be improved)
        if not url.startswith("http://") and not url.startswith("https://"):
             QMessageBox.warning(self, "Input Error", "Please enter a valid URL starting with http:// or https://")
             return None

        return {
            "url": url,
            "ignore_robots": ignore_robots,
            "depth": depth
        }

    # Override accept to validate before closing
    def accept(self):
        if self.get_values():
            super().accept()

if __name__ == '__main__':
    # Example usage for testing the dialog standalone
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    dialog = ScrapeDialog()
    if dialog.exec():
        values = dialog.get_values()
        print("Dialog Accepted:")
        print(f"  URL: {values['url']}")
        print(f"  Ignore Robots: {values['ignore_robots']}")
        print(f"  Depth: {values['depth']}")
    else:
        print("Dialog Cancelled")
    sys.exit(0)
