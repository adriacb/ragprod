from ragprod.infrastructure.client import AsyncChromaDBClient
from ragprod.domain.embedding.huggingface_embedding import HuggingFaceEmbedder
import torch

# ----- Global singletons -----
embedder = HuggingFaceEmbedder(
    model_name="jinaai/jina-code-embeddings-0.5b",
    model_kwargs={"device_map": "cpu", "dtype": "bfloat16"},
)

try:
    clientDB = AsyncChromaDBClient(
        persist_directory="./chromadb_test",
        collection_name="test",
        embedding_model=HuggingFaceEmbedder(
            model_name="jinaai/jina-code-embeddings-0.5b",
            model_kwargs={
                "device_map": "cpu",
                "dtype": torch.bfloat16,
                #"attn_implementation": "flash_attention_2",
            },
            tokenizer_kwargs={
                "padding_side": "left"
            }
        )
    )
    print("Initialized.")
except Exception as e:
    print(f"Error initializing client: {e}")
    raise e