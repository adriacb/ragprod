from fastmcp import FastMCP
from .client import clientDB, embedder
from ragprod.domain.embedding.huggingface_embedding import HuggingFaceEmbedder

mcp = FastMCP(
    name="RAGProd 🚀",
    version="0.0.1",
    #log_level=,
    debug=True
    #lifespan=lifespan,
)