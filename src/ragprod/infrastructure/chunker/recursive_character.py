from typing import List, Optional
from .base import BaseTextSplitter


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Recursive text splitter that splits text by trying different separators in order.
    
    Similar to LangChain's RecursiveCharacterTextSplitter, this splitter tries to split
    text by a list of separators in order, falling back to the next separator if chunks
    are too large.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: callable = len,
        keep_separator: bool = False,
    ):
        """
        Initialize the recursive character text splitter.
        
        Args:
            chunk_size: Maximum size of chunks (in characters or tokens).
            chunk_overlap: Overlap between chunks.
            separators: List of separators to try in order. Defaults to common separators.
            length_function: Function to measure text length. Defaults to len().
            keep_separator: Whether to keep the separator in the chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator

        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks recursively.
        
        Args:
            text: The text to split.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # If text is already small enough, return as single chunk
        if self.length_function(text) <= self.chunk_size:
            return [text]

        chunks = []
        splits = self._split_text_with_separator(text, self.separators)

        for split in splits:
            if self.length_function(split) <= self.chunk_size:
                chunks.append(split)
            else:
                # Recursively split if still too large
                sub_chunks = self.split_text(split)
                chunks.extend(sub_chunks)

        return self._merge_splits(chunks)

    def _split_text_with_separator(
        self, text: str, separators: List[str]
    ) -> List[str]:
        """Split text using the first separator that works."""
        if not separators:
            return [text]

        separator = separators[0]
        
        # Handle empty separator (character-by-character split)
        if separator == "":
            # Split into characters, but group them back into chunks
            if len(separators) == 1:
                # Last separator, return as single chunk
                return [text]
            # Try next separator
            return self._split_text_with_separator(text, separators[1:])
        
        splits = text.split(separator)

        # If we got multiple splits, use this separator
        if len(splits) > 1:
            if self.keep_separator:
                # Add separator back to all but last split
                result = [
                    split + separator if i < len(splits) - 1 else split
                    for i, split in enumerate(splits)
                ]
            else:
                result = splits

            # Filter out empty splits
            return [s for s in result if s.strip()]

        # If separator didn't work, try next one
        return self._split_text_with_separator(text, separators[1:])

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits with overlap."""
        if not splits:
            return []

        merged = []
        current_chunk = ""

        for split in splits:
            # If adding this split would exceed chunk size
            if (
                current_chunk
                and self.length_function(current_chunk + split) > self.chunk_size
            ):
                # Save current chunk
                if current_chunk:
                    merged.append(current_chunk.strip())

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + split
                else:
                    current_chunk = split
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += split
                else:
                    current_chunk = split

        # Add final chunk
        if current_chunk:
            merged.append(current_chunk.strip())

        return merged

