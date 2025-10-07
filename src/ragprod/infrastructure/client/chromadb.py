import os
import logging
from typing import List, Optional
from ragprod.domain.document import Document
from ragprod.domain.embedding.huggingface_embedding import HuggingFaceEmbedder
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AsyncChromaDBClient:
    def __init__(
        self,
        persist_directory: str = "./chromadb",
        embedding_model: Optional[HuggingFaceEmbedder] = None,
        collection_name: str = "default"
    ):
        self.embedding_model = embedding_model
        self.persist_directory = os.path.abspath(persist_directory)
        self.collection_name = collection_name

        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info(f"Initializing ChromaDB with persistence at: {self.persist_directory}")

        # Initialize Chroma client
        self.client = chromadb.Client(
            path=self.persist_directory,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    async def add_documents(self, documents: List[Document], collection_name: str = None):
        if not self.embedding_model:
            raise ValueError("Embedding model is not set.")

        collection_name = collection_name or self.collection.name
        collection = self.client.get_or_create_collection(name=collection_name)

        texts = [doc.raw_text for doc in documents]
        # Ensure metadata is never empty
        metadatas = [
            dict(doc.metadata) if doc.metadata else {"_dummy": "none"}
            for doc in documents
        ]
        ids = [doc.id for doc in documents]

        embeddings = await self.embedding_model.embed_documents(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")

    async def retrieve(self, query: str, k: int = 5, collection_name: str = None) -> List[Document]:
        if not self.embedding_model:
            raise ValueError("Embedding model is not set.")

        collection_name = collection_name or self.collection.name
        collection = self.client.get_or_create_collection(name=collection_name)

        embedding = await self.embedding_model.embed_query(query)
        if isinstance(embedding[0], (int, float)):
            embedding = [embedding]  # wrap in 2D

        results = collection.query(query_embeddings=embedding, n_results=k)
        docs = []

        for i in range(len(results["documents"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i] or {}
            docs.append(Document(id=doc_id, raw_text=text, metadata=metadata))

        logger.info(f"Retrieved {len(docs)} documents from '{collection_name}'")
        return docs

    async def delete(self, ids: List[str], collection_name: str = None):
        collection_name = collection_name or self.collection.name
        collection = self.client.get_or_create_collection(name=collection_name)
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from collection '{collection_name}'")

    async def reset(self, collection_name: str = None):
        collection_name = collection_name or self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Reset collection '{collection_name}'")

    async def count(self, collection_name: str = None) -> int:
        collection_name = collection_name or self.collection.name
        collection = self.client.get_or_create_collection(name=collection_name)
        return collection.count()
