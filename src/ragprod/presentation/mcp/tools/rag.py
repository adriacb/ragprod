from ragprod.domain.document import Document
from typing import List
from ragprod.presentation.mcp.server import mcp
from ..client import clientDB
from fastmcp import Context

@mcp.tool
async def rag_retrieve(
    query: str, 
    limit: int = 5, 
    ctx: Context = None
    ) -> List[Document]:
    """
    Retrieve documents from the RAG database.

    Args:
        query: The query to retrieve documents from the RAG database.
        limit: The number of documents to retrieve.
        ctx: The context of the tool call.
    
    Returns:
        A list of documents retrieved from the RAG database.
    """
    if clientDB is None:
        raise RuntimeError("DB client not initialized")
    try:
        results = await clientDB.retrieve(query, limit)
    except Exception as e:
        await ctx.error(f"Retrieval failed: {e}")
        return []

    await ctx.info(f"Retrieved {len(results)} documents")
    return results

@mcp.tool
async def add_documents(
    documents: List[Document], 
    ctx: Context = None
    ) -> List[Document]:
    if clientDB is None:
        raise RuntimeError("DB client not initialized")
    try:
        await clientDB.add_documents(documents)
    except Exception as e:
        await ctx.error(f"Failed to add documents: {e}")
        return []
    await ctx.info(f"Added {len(documents)} documents")
    return documents
