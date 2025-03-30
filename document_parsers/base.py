# document_parsers/base.py
import abc

class BaseParser(abc.ABC):
    """Abstract base class for document parsers."""

    @abc.abstractmethod
    def parse(self, file_path: str) -> str:
        """
        Parses the content of a file into a single text string.

        Args:
            file_path (str): The path to the file to parse.

        Returns:
            str: The extracted text content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For any parsing errors specific to the file type.
        """
        pass
