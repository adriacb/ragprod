from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ragprod.domain.document.base import BaseDocument


class BaseTextSplitter(ABC):
    """Base class for text splitters, similar to LangChain's approach."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into a list of strings.
        
        Args:
            text: The text to split.
            
        Returns:
            List of text chunks.
        """
        pass

    def split_documents(
        self, documents: List[BaseDocument]
    ) -> List[BaseDocument]:
        """
        Split documents into a list of documents.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of chunked documents with preserved metadata.
        """
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.content)
            # Build base metadata from document
            base_metadata = doc.metadata.copy() if doc.metadata else {}
            # Add source and title from document attributes
            if hasattr(doc, 'source'):
                base_metadata['source'] = doc.source
            if hasattr(doc, 'title'):
                base_metadata['title'] = doc.title
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = self._create_chunk_metadata(
                    base_metadata, i, len(text_chunks), chunk_text
                )
                chunk = self._create_document(chunk_text, chunk_metadata)
                chunks.append(chunk)
        return chunks

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[BaseDocument]:
        """
        Create documents from a list of texts.
        
        Args:
            texts: List of text strings.
            metadatas: Optional list of metadata dicts for each text.
            
        Returns:
            List of documents.
        """
        from ragprod.domain.document import Document

        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            # Extract source and title from metadata if present
            source = metadata.get("source", "Unknown")
            title = metadata.get("title", "Untitled")
            doc = Document(
                raw_text=text,
                source=source,
                title=title,
                metadata=metadata
            )
            documents.append(doc)

        return self.split_documents(documents)

    def _create_chunk_metadata(
        self,
        original_metadata: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        chunk_text: str,
    ) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        chunk_metadata = original_metadata.copy()
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["total_chunks"] = total_chunks
        chunk_metadata["chunk_length"] = len(chunk_text)
        return chunk_metadata

    def _create_document(
        self, text: str, metadata: Dict[str, Any]
    ) -> BaseDocument:
        """Create a document from text and metadata."""
        from ragprod.domain.document import Document

        # Extract source and title from metadata if present, but keep them in metadata
        source = metadata.get("source", "Unknown")
        title = metadata.get("title", "Untitled")
        
        doc = Document(
            raw_text=text,
            source=source,
            title=title,
            metadata=metadata,  # Pass metadata directly (includes source and title)
        )
        return doc


# Keep BaseChunker for backward compatibility
BaseChunker = BaseTextSplitter
