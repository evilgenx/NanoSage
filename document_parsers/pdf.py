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

    def _format_table_data(self, extracted_data: list[list[str | None]]) -> str:
        """Formats extracted table data (list of lists) into a plain text string."""
        if not extracted_data:
            return ""
        formatted_table = "--- TABLE START ---\n"
        try:
            for row in extracted_data:
                # Ensure all cell contents are strings, handling None
                formatted_row = " | ".join(str(cell) if cell is not None else "" for cell in row)
                formatted_table += formatted_row + "\n"
            formatted_table += "--- TABLE END ---\n"
        except Exception as e:
            print(f"[WARN] Could not format table row data: {e}")
            # Return minimal info even if formatting fails mid-way
            formatted_table += "... (error formatting table) ...\n--- TABLE END ---\n"
        return formatted_table

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
                page_text = page.get_text("text") # Get standard text first
                tables_text = "" # Initialize table text for the page

                # --- Add Table Extraction ---
                try:
                    # Default strategy is "lines", can be tuned if needed
                    table_settings = {} 
                    table_finder = page.find_tables(strategy=table_settings.get("strategy", "lines"), 
                                                    vertical_strategy=table_settings.get("vertical_strategy", "lines"),
                                                    horizontal_strategy=table_settings.get("horizontal_strategy", "lines"))
                    if table_finder.tables:
                        # print(f"[DEBUG] Found {len(table_finder.tables)} tables on page {page_num + 1}") # Optional debug log
                        for table in table_finder.tables:
                            # Extract table data
                            extracted_data = table.extract()
                            if extracted_data:
                                # Format and append
                                tables_text += self._format_table_data(extracted_data) + "\n"
                except Exception as table_ex:
                    # Log warning but continue processing the page
                    print(f"[WARN] Error processing tables on page {page_num + 1} of {file_path}: {table_ex}")
                # --- End Table Extraction ---

                # Append page text and then any extracted table text
                extracted_text += page_text + "\n" + tables_text

            # Return the combined extracted text if successful
            return extracted_text.strip() # Use strip() to remove potential trailing newline
        except Exception as e:
            print(f"[ERROR] Failed to parse PDF file {file_path}: {e}")
            raise # Re-raise the exception
        finally:
            # Ensure the document is closed regardless of success or failure
            if doc:
                doc.close()
