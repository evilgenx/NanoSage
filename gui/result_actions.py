#!/usr/bin/env python3
# gui/result_actions.py

import os
import sys
import subprocess
import webbrowser
import urllib.parse
import logging
from PyQt6.QtWidgets import QMessageBox

def open_report(report_path, log_status_func):
    """Open the generated report file using the default system viewer."""
    if report_path and os.path.exists(report_path):
        try:
            if sys.platform == "win32":
                os.startfile(report_path)
            elif sys.platform == "darwin": # macOS
                subprocess.run(["open", report_path], check=True)
            else: # Linux and other Unix-like
                subprocess.run(["xdg-open", report_path], check=True)
            log_status_func(f"Attempting to open report: {report_path}")
        except Exception as e:
            error_msg = f"Error opening report: {e}"
            log_status_func(error_msg)
            QMessageBox.warning(None, "Open Error", f"Could not open the report file:\n{e}") # Use None for parent if called outside MainWindow context
    else:
        QMessageBox.warning(None, "File Not Found", "The report file does not exist or path is not set.")

def open_results_folder(results_dir, log_status_func):
    """Open the folder containing the results."""
    if results_dir and os.path.exists(results_dir):
        try:
            if sys.platform == "win32":
                 # Use explorer for Windows, safer for paths with spaces
                 subprocess.run(['explorer', results_dir])
            elif sys.platform == "darwin": # macOS
                subprocess.run(["open", results_dir], check=True)
            else: # Linux and other Unix-like
                subprocess.run(["xdg-open", results_dir], check=True)
            log_status_func(f"Attempting to open results folder: {results_dir}")
        except Exception as e:
            error_msg = f"Error opening results folder: {e}"
            log_status_func(error_msg)
            QMessageBox.warning(None, "Open Error", f"Could not open the results folder:\n{e}")
    else:
         QMessageBox.warning(None, "Folder Not Found", "The results directory does not exist or path is not set.")

def share_report_email(report_path, log_status_func):
    """Open the default email client with a pre-filled message."""
    if report_path and os.path.exists(report_path):
        try:
            subject = "NanoSage Research Report"
            body = (
                f"Please find the research report attached.\n\n"
                f"You can find the file at:\n{report_path}\n\n"
                f"(Please attach the file manually before sending)"
            )
            # URL encode subject and body
            encoded_subject = urllib.parse.quote(subject)
            encoded_body = urllib.parse.quote(body)

            mailto_url = f"mailto:?subject={encoded_subject}&body={encoded_body}"

            webbrowser.open(mailto_url)
            log_status_func(f"Attempting to open email client for report: {report_path}")
        except Exception as e:
            error_msg = f"Error opening email client: {e}"
            log_status_func(error_msg)
            QMessageBox.warning(None, "Email Error", f"Could not open the email client:\n{e}")
    else:
        QMessageBox.warning(None, "File Not Found", "The report file does not exist or path is not set. Cannot share.")
