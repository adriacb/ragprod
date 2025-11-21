from typing import List
from .base import BaseTextSplitter


class CharacterTextSplitter(BaseTextSplitter):
    """
    Simple character-based text splitter.
    
    Splits text into chunks of a fixed character size with optional overlap.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "",
    ):
        """
        Initialize the character text splitter.
        
        Args:
            chunk_size: Maximum size of chunks in characters.
            chunk_overlap: Overlap between chunks in characters.
            separator: Separator to use (empty string means no separator).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of fixed character size.
        
        Args:
            text: The text to split.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we have a separator, try to split at separator boundary
            if self.separator and end < len(text):
                # Look for separator near the end
                sep_pos = text.rfind(self.separator, start, end)
                if sep_pos != -1:
                    end = sep_pos + len(self.separator)

            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

        return chunks

