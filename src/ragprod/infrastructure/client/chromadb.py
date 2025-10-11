import os
import logging
from typing import List, Optional
from ragprod.domain.document import Document
from ragprod.domain.embedding import EmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_client(
    persist_directory: Optional[str] = None,
    api_host: Optional[str] = None,
    api_port: Optional[int] = None,
):
    """
    Returns an appropriate Chroma client depending on parameters:

      - If `api_host` is set -> use HttpClient (remote API mode)
      - If `persist_directory` is set -> use PersistentClient (local storage)
      - Else -> in-memory Client (no persistence)
    """
    from chromadb import Client, PersistentClient, HttpClient
    from chromadb.config import Settings

    # --- 1️⃣ Remote API mode ---
    if api_host:
        port = api_port or 8000
        logger.info(f"Using Chroma HTTP client at {api_host}:{port}")
        return HttpClient(host=api_host, port=port)

    # --- 2️⃣ Persistent local mode ---
    elif persist_directory:
        abs_path = os.path.abspath(persist_directory)
        os.makedirs(abs_path, exist_ok=True)
        logger.info(f"Using persistent Chroma client at: {abs_path}")
        return PersistentClient(path=abs_path)

    # --- 3️⃣ In-memory mode ---
    else:
        logger.info("Using in-memory Chroma client (no persistence).")
        return Client(Settings(anonymized_telemetry=False))

class AsyncChromaDBClient:
    def __init__(
        self,
        persist_directory: Optional[str] = "./chromadb",
        embedding_model: Optional["EmbeddingModel"] = None,
        collection_name: str = "default",
        api_host: Optional[str] = None,
        api_port: Optional[int] = None,
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.api_host = api_host
        self.api_port = api_port

        # Pick the appropriate client
        self.client = get_client(
            persist_directory=self.persist_directory,
            api_host=self.api_host,
            api_port=self.api_port,
        )

        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        mode = (
            "remote API"
            if self.api_host
            else "persistent"
            if self.persist_directory
            else "in-memory"
        )
        logger.info(f"ChromaDB initialized in {mode} mode.")


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
        print(results)
        docs = []

        for i in range(len(results["documents"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i] or {}
            distance = results["distances"][0][i]
            docs.append(Document(id=doc_id, raw_text=text, metadata=metadata, distance=distance))

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
