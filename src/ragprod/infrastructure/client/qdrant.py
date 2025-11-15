from ragprod.domain.client.base import BaseClient
from ragprod.domain.embedding.base import EmbeddingModel
from ragprod.domain.document import Document

from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
import logging

from pydantic import BaseModel

class PointStruct(BaseModel):
    id: int
    vector: List[float]
    payload: Dict[str, Any]

    class Config:
        from_attributes = True

class QdrantRetriever(BaseClient):
    def __init__(self, host: str = "localhost", port: int = 6333, embedding_model: EmbeddingModel = None):
        self.host = host
        self.port = port
        self.client = self._connect()
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
    
    def _connect(self):
        try:
            return QdrantClient(host=self.host, port=self.port)
        except Exception as e:
            raise Exception(f"Failed to connect to Qdrant: {e}")

    def get_collection(self, collection_name: str):
        return self.client.get_collection(collection_name)
    
    def create_collection(self, collection_name: str, size: int = 1536, distance: models.Distance = models.Distance.COSINE):
        return self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=size,
                distance=distance
            )
            )

    def from_documents_to_qdrant(self, documents: List[Document], vectors: List[List[float]]) -> List[PointStruct]:
        """
        Converts documents and their vectors into Qdrant PointStructs.
        """
        points = []
        for idx, (doc, vector) in enumerate(zip(documents, vectors)):
            payload = {
                "raw_text": doc.raw_text,
                "source": doc.source,
                "title": doc.title,
                **(doc.metadata or {})
            }
            points.append(
                models.PointStruct(
                    id=idx,  # Optionally use a real ID if you have one
                    vector=vector,
                    payload=payload
                )
            )
        return points


    def get_embedding_size(self, model_name: str):
        return self.client.get_embedding_size(model_name)
    
    def delete_collection(self, collection_name: str):
        return self.client.delete_collection(collection_name)
    
    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def upload_collection(self, collection_name: str, documents: List[Document]):
        self.client.upload_collection(
            collection_name=collection_name,
            vectors=documents,
            ids=[document.id if document.id else idx for idx, document in enumerate(documents)],

        )

    def insert(self, collection_name: str, documents: List[Document], vectors: List[List[float]]):
        try:
            points = self.from_documents_to_qdrant(documents, vectors)
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
        except Exception as e:
            self.logger.error(f"Failed to insert documents: {e}")
            raise Exception(f"Failed to insert documents: {e}")

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        texts = [doc.raw_text for doc in documents]
        return self.embeddings.embed_documents(texts)

    def retrieve(self, query: str, collection_name: str, limit: int = 5) -> List[Document]:
        try:
            query_vector = self.embed_query(query)

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )

            # Convert payloads back to Document objects
            documents = []
            for hit in results:
                payload = hit.payload or {}
                documents.append(
                    Document(
                        raw_text=payload.get("raw_text", ""),
                        source=payload.get("source", "Unknown"),
                        title=payload.get("title", "Untitled"),
                        metadata={k: v for k, v in payload.items() if k not in {"raw_text", "source", "title"}}
                    )
                )

            return documents

        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            raise Exception(f"Failed to retrieve documents: {e}")

