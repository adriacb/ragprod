from .base import BaseChunker
from typing import List
from pydantic import BaseModel
from ..document import Document

class Chunker(BaseChunker, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def chunk_fixed_size(self, Document: Document, chunk_size: int) -> List[Document]:
        """Chunk the document into fixed size chunks.
        
        Args:
            Document: The document to chunk.
            chunk_size: The size of the chunks.

        Returns:
            List[Document]: The chunks of the document.
        """
        text_words = Document.content.split()

        chunks = []

        for i in range(0, len(text_words), chunk_size):
            chunk_words = text_words[i: i+chunk_size]
            # add length of chunk to metadata
            Document.metadata["length"] = len(chunk_words)
            Document.metadata["index"] = i
            chunks.append(Document(content=" ".join(chunk_words), metadata=Document.metadata))

        return chunks

    def chunk_fixed_size_with_overlap(self, Document: Document, chunk_size: int, overlap: int) -> List[Document]:
        """Chunk the document into fixed size chunks with overlap.
        
        Args:
            Document: The document to chunk.
            chunk_size: The size of the chunks.
            overlap: The overlap between the chunks.
            
        """
        text_words = Document.content.split()

        return [
            Document(
                content=" ".join(text_words[i:i+chunk_size]),
                metadata={
                    **Document.metadata, 
                    "length": len(text_words[i:i+chunk_size]),
                    "index": i
                    }
            )
            for i in range(0, len(text_words), chunk_size - overlap)
        ]

    def recursive_character_chunking(
            self,
            Document: Document,
            marker: list[str],
            min_length: int = 0
        ) -> List[Document]:
        """Recursively chunk a document by splitting on a sequence of character markers.

        The function splits the document content using the first marker in the list.
        Each resulting chunk becomes a new Document with updated metadata (including
        length and index). The process continues recursively with the remaining markers.
        Chunks shorter than `min_length` are excluded from the output.

        Args:
            Document (Document): The document to be chunked.
            marker (list[str]): A list of markers to split the text by, in order of application.
            min_length (int, optional): Minimum character length for a chunk to be included.
                Defaults to 0 (no filtering).

        Returns:
            List[Document]: A list of chunked Documents. Each chunk includes:
                - "length": number of characters in the chunk
                - "index": local index of the chunk at its recursion level
        """
        if not marker:
            return [Document] if len(Document.content) >= min_length else []

        parts = Document.content.split(marker[0])
        chunks = []
        for idx, p in enumerate(parts):
            if len(p.strip()) >= min_length:
                sub_docs = self.recursive_character_chunking(
                    Document(
                        content=p,
                        metadata={**Document.metadata, "length": len(p), "index": idx}
                    ),
                    marker[1:],
                    min_length=min_length
                )
                chunks.extend(sub_docs)
        return chunks


