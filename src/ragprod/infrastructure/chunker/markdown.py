import re
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseTextSplitter
from ragprod.domain.document.base import BaseDocument


class MarkdownHeaderTextSplitter(BaseTextSplitter):
    """
    Split Markdown documents based on header hierarchy.
    
    Similar to LangChain's MarkdownHeaderTextSplitter, this splitter divides
    Markdown documents into semantically meaningful chunks based on header hierarchy.
    It preserves metadata for each header relevant to any given chunk.
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        strip_headers: bool = True,
    ):
        """
        Initialize the Markdown header text splitter.
        
        Args:
            headers_to_split_on: List of tuples (header_level, header_name) to split on.
                Defaults to all headers (# through ######).
            strip_headers: Whether to strip headers from chunk content.
        """
        if headers_to_split_on is None:
            # Default: split on all header levels
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ]

        self.headers_to_split_on = headers_to_split_on
        self.strip_headers = strip_headers

        # Build regex pattern for headers
        header_patterns = []
        for header_level, _ in headers_to_split_on:
            # Escape special regex characters
            escaped = re.escape(header_level)
            header_patterns.append(f"^{escaped}\\s+(.+)$")

        self.header_pattern = re.compile("|".join(header_patterns), re.MULTILINE)

    def split_text(self, text: str) -> List[str]:
        """
        Split Markdown text based on headers.
        
        Args:
            text: The Markdown text to split.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # Find all header positions
        header_positions = self._find_header_positions(text)

        if not header_positions:
            # No headers found, return entire text as single chunk
            return [text]

        chunks = []
        for i, (pos, level, header_text, metadata) in enumerate(header_positions):
            # Determine end position (next header or end of text)
            if i + 1 < len(header_positions):
                end_pos = header_positions[i + 1][0]
            else:
                end_pos = len(text)

            # Extract chunk content
            chunk_content = text[pos:end_pos]

            if self.strip_headers:
                # Remove the header line from content
                lines = chunk_content.split("\n")
                if lines and lines[0].strip().startswith(level):
                    chunk_content = "\n".join(lines[1:])

            chunks.append(chunk_content.strip())

        return [chunk for chunk in chunks if chunk]

    def split_documents(
        self, documents: List[BaseDocument]
    ) -> List[BaseDocument]:
        """
        Split documents and preserve header metadata.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of chunked documents with header metadata preserved.
        """
        chunks = []
        for doc in documents:
            text_chunks, chunk_metadatas = self._split_text_with_metadata(
                doc.content
            )

            for i, (chunk_text, chunk_metadata) in enumerate(
                zip(text_chunks, chunk_metadatas)
            ):
                # Merge with original document metadata
                merged_metadata = {**doc.metadata, **chunk_metadata}
                merged_metadata["chunk_index"] = i
                merged_metadata["total_chunks"] = len(text_chunks)
                merged_metadata["chunk_length"] = len(chunk_text)

                chunk = self._create_document(chunk_text, merged_metadata)
                chunks.append(chunk)

        return chunks

    def _find_header_positions(
        self, text: str
    ) -> List[Tuple[int, str, str, Dict[str, Any]]]:
        """Find all header positions in the text."""
        positions = []
        lines = text.split("\n")

        current_headers = {}  # Track header hierarchy

        for line_num, line in enumerate(lines):
            for header_level, header_name in self.headers_to_split_on:
                if line.strip().startswith(header_level + " "):
                    # Found a header
                    header_text = line.strip()[len(header_level) :].strip()
                    pos = sum(len(l) + 1 for l in lines[:line_num])  # +1 for newline

                    # Update header hierarchy
                    # Clear headers at same or deeper level
                    level_num = len(header_level)
                    current_headers = {
                        k: v
                        for k, v in current_headers.items()
                        if len(k.split("_")[0]) < level_num
                    }
                    current_headers[header_level] = header_text

                    # Build metadata
                    metadata = {}
                    for h_level, h_text in current_headers.items():
                        metadata_key = f"header_{len(h_level)}"
                        metadata[metadata_key] = h_text

                    positions.append((pos, header_level, header_text, metadata))
                    break

        return positions

    def _split_text_with_metadata(
        self, text: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Split text and return chunks with their metadata."""
        header_positions = self._find_header_positions(text)

        if not header_positions:
            return [text], [{}]

        chunks = []
        metadatas = []

        for i, (pos, level, header_text, metadata) in enumerate(header_positions):
            # Determine end position
            if i + 1 < len(header_positions):
                end_pos = header_positions[i + 1][0]
            else:
                end_pos = len(text)

            # Extract chunk content
            chunk_content = text[pos:end_pos]

            if self.strip_headers:
                lines = chunk_content.split("\n")
                if lines and lines[0].strip().startswith(level):
                    chunk_content = "\n".join(lines[1:])

            chunks.append(chunk_content.strip())
            metadatas.append(metadata)

        return chunks, metadatas

