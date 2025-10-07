from ragprod.core.embedding import HuggingFaceEmbedder
from ragprod.core.document import Document
from ragprod.infrastructure.client import AsyncChromaDBClient
from typing import List
from ragprod.presentation.mcp.server import mcp, client
from fastmcp import Context

@mcp.tool
async def rag_retrieve(query: str, limit: int = 5, ctx: Context = None):
    if client is None:
        raise RuntimeError("DB client not initialized")
    results = await client.retrieve(query, limit)
    await ctx.info(f"Retrieved {len(results)} documents")
    return results
