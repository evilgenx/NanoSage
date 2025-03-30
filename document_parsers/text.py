# document_parsers/text.py
import os
from .base import BaseParser

class TextParser(BaseParser):
    """Parses plain text files."""

    def parse(self, file_path: str) -> str:
        """Reads content from a text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read text file {file_path}: {e}")
            raise # Re-raise the exception
