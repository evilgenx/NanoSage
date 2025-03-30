#!/usr/bin/env python3
# gui/app.py

import sys
from PyQt6.QtWidgets import QApplication

# Import MainWindow using relative import
try:
    from .main_window import MainWindow
except ImportError as e:
    print(f"Error importing MainWindow in app.py: {e}")
    sys.exit(1)

# --- Application Entry Point ---

def main():
    """Initializes and runs the PyQt application."""
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
