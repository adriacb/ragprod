from ragprod.domain.embedding import HuggingFaceEmbedder
from ragprod.domain.document import Document
from ragprod.infrastructure.client import AsyncChromaDBClient
from typing import List
from ragprod.presentation.mcp.server import mcp, clientDB
from fastmcp import Context

@mcp.tool
async def rag_retrieve(query: str, limit: int = 5, ctx: Context = None):
    if clientDB is None:
        raise RuntimeError("DB client not initialized")
    try:
        results = await clientDB.retrieve(query, limit)
    except Exception as e:
        await ctx.error(f"Retrieval failed: {e}")
        return []

    await ctx.info(f"Retrieved {len(results)} documents")
    return results
