from ragprod.infrastructure.client import AsyncChromaDBClient
from ragprod.core.embedding.huggingface_embedding import HuggingFaceEmbedder
import torch

class MockAsyncEmbeddingModel:
    async def embed_documents(self, texts):
        # Simulate async delay
        return [[float(i)] * 768 for i in range(len(texts))]

    async def embed_query(self, query):
        return [0.5] * 768

from ragprod.core.document import Document

documents = [
    Document(raw_text="Machine learning is fun", metadata={"topic": "ML"}),
    Document(raw_text="Deep learning uses neural networks", metadata={"topic": "DL"}),
    Document(raw_text="AI is the future", metadata={"topic": "AI"}),
]


try:
    client = AsyncChromaDBClient(
        embedding_model=HuggingFaceEmbedder(
            model_name="jinaai/jina-code-embeddings-0.5b",
            model_kwargs={
                "device_map": "cuda",
                "dtype": torch.bfloat16,
                #"attn_implementation": "flash_attention_2",
            },
            tokenizer_kwargs={
                "padding_side": "left"
            }
        )
    )
except Exception as e:
    print(f"Error initializing client: {e}")
    raise e