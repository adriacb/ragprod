from contextlib import asynccontextmanager
from fastmcp import FastMCP
from ragprod.presentation.mcp.client import client
from ragprod.infrastructure.client import AsyncChromaDBClient
from ragprod.core.embedding.huggingface_embedding import HuggingFaceEmbedder
import torch


# ----- Global singletons -----
embedder = HuggingFaceEmbedder(
    model_name="jinaai/jina-code-embeddings-0.5b",
    model_kwargs={"device_map": "cuda", "dtype": "bfloat16"},
)
client = AsyncChromaDBClient(embedding_model=embedder)

mcp = FastMCP(
    name="RAGProd 🚀",
    #lifespan=lifespan,
)

