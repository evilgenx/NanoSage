# document_parsers/docx.py
import os
import docx # python-docx library
from .base import BaseParser

class DocxParser(BaseParser):
    """Parses DOCX files using python-docx."""

    def parse(self, file_path: str) -> str:
        """
        Parses the content of a DOCX file into a single text string.

        Args:
            file_path (str): The path to the DOCX file to parse.

        Returns:
            str: The extracted text content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For any parsing errors.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        try:
            document = docx.Document(file_path)
            full_text = []
            for para in document.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text) # Join paragraphs with newline

        except ImportError:
             # Should not happen if requirements are installed, but good practice
             raise ImportError("python-docx library not found. Please install it to parse DOCX files.") from None
        except Exception as e:
            # Catch potential errors from python-docx (e.g., corrupted file)
            raise Exception(f"Error parsing DOCX file {file_path}: {e}") from e
