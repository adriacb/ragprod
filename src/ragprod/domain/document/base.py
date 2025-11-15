from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDocument(ABC):
    """Interface for documents in a RAG pipeline."""

    @property
    @abstractmethod
    def content(self) -> str:
        """Main text content of the document."""

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Metadata like source, title, author, etc."""

    @abstractmethod
    def __str__(self) -> str:
        """Representation of the document."""
        if self.metadata:
            return f"Document(page_content: {self.content} metadata: {self.metadata})"
        return f"Document(page_content: {self.content})"