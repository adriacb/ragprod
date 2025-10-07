import asyncio
import uuid
import torch
from ragprod.infrastructure.client import AsyncChromaDBClient
from ragprod.core.embedding.huggingface_embedding import HuggingFaceEmbedder
from ragprod.core.document import Document

async def main():
    # Initialize embedder
    embedder = HuggingFaceEmbedder(
        model_name="jinaai/jina-code-embeddings-0.5b",
        model_kwargs={
            "device_map": "cuda",
            "dtype": torch.bfloat16,
        },
        tokenizer_kwargs={
            "padding_side": "left"
        }
    )

    # Initialize Chroma client
    client = AsyncChromaDBClient(
        embedding_model=embedder,
        persist_directory="./chromadb_test"
    )

    # Add documents to collection "test"
    documents = [
        Document(id=str(uuid.uuid4()), raw_text="Machine learning is fun", metadata={"topic": "ML"}),
        Document(id=str(uuid.uuid4()), raw_text="Deep learning uses neural networks", metadata={"topic": "DL"}),
        Document(id=str(uuid.uuid4()), raw_text="AI is the future", metadata={"topic": "AI"}),
    ]
    await client.add_documents(documents, collection_name="test")

    # Count documents
    count = await client.count(collection_name="test")
    print(f"Total documents in DB: {count}")

    # Retrieve from the same collection
    results = await client.retrieve(query="Machine learning", k=5, collection_name="test")
    for doc in results:
        print(f"ID: {doc.id}, Text: {doc.raw_text}, Metadata: {doc.metadata}")

asyncio.run(main())
