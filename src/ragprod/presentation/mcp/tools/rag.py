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
    chunker_name: str = "recursive_character",
    chunker_config: dict = None,
    ctx: Context = None
    ) -> List[Document]:
    """
    Add documents to the RAG database.
    
    Args:
        documents: List of documents to add.
        chunker_name: Name of the chunker to use (default: "recursive_character").
        chunker_config: Configuration for the chunker (default: {"chunk_size": 1000, "chunk_overlap": 200}).
        ctx: The context of the tool call.
        
    Returns:
        List of documents added (chunks).
    """
    if clientDB is None:
        raise RuntimeError("DB client not initialized")
    
    try:
        # Get global chunker service instance
        from ragprod.presentation.mcp.lifespan.service import get_chunker_service_instance
        chunker_service = get_chunker_service_instance()
        
        # Default config for recursive_character if not provided
        if chunker_config is None:
            chunker_config = {
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            
        # Get chunker instance
        chunker = chunker_service.get(chunker_name, chunker_config)
        
        # Split documents
        chunked_docs = chunker.split_documents(documents)
        
        # Add to database
        await clientDB.add_documents(chunked_docs)
        
    except Exception as e:
        await ctx.error(f"Failed to add documents: {e}")
        return []
        
    await ctx.info(f"Added {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs
