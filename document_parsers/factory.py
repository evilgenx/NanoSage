# document_parsers/factory.py
import os
from .base import BaseParser
from .text import TextParser
from .pdf import PdfParser
from .html import HtmlParser
from .docx import DocxParser # Import the new parser

# Cache parser instances to avoid re-initialization if needed (simple dict cache)
_parser_cache = {}

def get_parser(file_path: str) -> BaseParser | None:
    """
    Factory function to get the appropriate parser based on file extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        BaseParser | None: An instance of the appropriate parser, or None if the
                           file type is unsupported or the file doesn't exist.
    """
    if not os.path.isfile(file_path):
        print(f"[WARN] File not found, cannot determine parser: {file_path}")
        return None

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    parser_key = extension # Use extension as cache key

    # Check cache first
    if parser_key in _parser_cache:
        return _parser_cache[parser_key]

    # Determine parser type
    parser_instance = None
    if extension == ".txt":
        parser_instance = TextParser()
    elif extension == ".pdf":
        try:
            # PdfParser might raise ImportError if fitz is not installed
            parser_instance = PdfParser()
        except ImportError as e:
            print(f"[WARN] Cannot create PDF parser, dependency missing: {e}")
            return None # Cannot parse this file type
    elif extension in [".html", ".htm"]:
        try:
            # HtmlParser might raise ImportError if bs4/lxml are not installed
            parser_instance = HtmlParser()
        except ImportError as e:
            print(f"[WARN] Cannot create HTML parser, dependency missing: {e}")
            return None # Cannot parse this file type
    elif extension == ".docx":
        try:
            # DocxParser might raise ImportError if python-docx is not installed
            parser_instance = DocxParser()
        except ImportError as e:
            print(f"[WARN] Cannot create DOCX parser, dependency missing: {e}")
            return None # Cannot parse this file type
    # Add other parsers here if needed

    if parser_instance:
        _parser_cache[parser_key] = parser_instance # Cache the instance
        return parser_instance
    else:
        print(f"[INFO] Unsupported file type for parsing: {extension} ({file_path})")
        return None
