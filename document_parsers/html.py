# document_parsers/html.py
import os
from bs4 import BeautifulSoup
from .base import BaseParser

class HtmlParser(BaseParser):
    """Parses HTML files using BeautifulSoup."""

    def parse(self, file_path: str) -> str:
        """
        Parses the content of an HTML file into a single text string.

        Args:
            file_path (str): The path to the HTML file to parse.

        Returns:
            str: The extracted text content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For any parsing errors.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"HTML file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml') # Use lxml parser

            # Extract text content; this is a basic approach.
            # It might need refinement depending on the desired level of detail
            # (e.g., handling specific tags, preserving some structure).
            text_content = soup.get_text(separator=' ', strip=True)
            return text_content

        except ImportError:
            # Handle case where lxml is not installed (though it's in requirements)
            print("[WARN] lxml parser not found, trying html.parser for HTML parsing.")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f, 'html.parser') # Fallback parser
                text_content = soup.get_text(separator=' ', strip=True)
                return text_content
            except Exception as e:
                 raise Exception(f"Error parsing HTML file {file_path} with fallback parser: {e}") from e
        except Exception as e:
            raise Exception(f"Error parsing HTML file {file_path}: {e}") from e
