# document_parsers/pdf.py
import os
from .base import BaseParser

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

class PdfParser(BaseParser):
    """Parses PDF files using PyMuPDF."""

    def __init__(self, max_pages=None): # Allow limiting pages if needed
        if fitz is None:
            raise ImportError("PDF parsing requires PyMuPDF. Please install it: pip install PyMuPDF")
        self.max_pages = max_pages

    def parse(self, file_path: str) -> str:
        """Extracts text content from a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        extracted_text = ""
        doc = None # Initialize doc to None
        try:
            doc = fitz.open(file_path)
            num_pages_to_process = len(doc)
            if self.max_pages is not None and self.max_pages > 0:
                num_pages_to_process = min(num_pages_to_process, self.max_pages)

            for page_num in range(num_pages_to_process):
                page = doc.load_page(page_num)
                extracted_text += page.get_text("text") + "\n" # Use "text" for better layout preservation

            # Return the extracted text if successful
            return extracted_text
        except Exception as e:
            print(f"[ERROR] Failed to parse PDF file {file_path}: {e}")
            raise # Re-raise the exception
        finally:
            # Ensure the document is closed regardless of success or failure
            if doc:
                doc.close()
