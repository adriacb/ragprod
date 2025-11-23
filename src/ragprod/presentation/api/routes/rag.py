from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from ragprod.domain.document import Document
from ragprod.presentation.api.lifespan.service import get_chunker_service_instance
from ragprod.presentation.mcp.client import clientDB
from ragprod.infrastructure.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class AddDocumentsRequest(BaseModel):
    """Request model for adding documents."""
    documents: List[dict] = Field(..., description="List of documents to add")
    chunker_name: str = Field(default="recursive_character", description="Name of the chunker to use")
    chunker_config: Optional[dict] = Field(default=None, description="Configuration for the chunker")


class AddDocumentsResponse(BaseModel):
    """Response model for adding documents."""
    message: str
    chunks_created: int


class RetrieveRequest(BaseModel):
    """Request model for retrieving documents."""
    query: str = Field(..., description="Query to search for")
    limit: int = Field(default=5, description="Number of documents to retrieve")


@router.post("/add_documents", response_model=AddDocumentsResponse)
async def add_documents(request: AddDocumentsRequest):
    """
    Add documents to the RAG database with chunking.
    
    Args:
        request: Request containing documents and chunking configuration
        
    Returns:
        Response with number of chunks created
    """
    try:
        # Get chunker service
        chunker_service = get_chunker_service_instance()
        
        # Convert dict to Document objects
        documents = [Document(**doc) for doc in request.documents]
        
        # Default config if not provided
        chunker_config = request.chunker_config or {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        # Get chunker instance
        logger.info(f"Using chunker: {request.chunker_name} with config: {chunker_config}")
        chunker = chunker_service.get(request.chunker_name, chunker_config)
        
        # Split documents
        chunked_docs = chunker.split_documents(documents)
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        # Add to database
        if clientDB is None:
            raise HTTPException(status_code=500, detail="Database client not initialized")
        
        await clientDB.add_documents(chunked_docs)
        
        return AddDocumentsResponse(
            message=f"Successfully added {len(chunked_docs)} chunks from {len(documents)} documents",
            chunks_created=len(chunked_docs)
        )
        
    except ValueError as e:
        logger.error(f"Invalid chunker configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """
    Retrieve documents from the RAG database.
    
    Args:
        request: Request containing query and limit
        
    Returns:
        List of retrieved documents
    """
    try:
        if clientDB is None:
            raise HTTPException(status_code=500, detail="Database client not initialized")
        
        logger.info(f"Retrieving documents for query: {request.query}")
        results = await clientDB.retrieve(request.query, request.limit)
        
        logger.info(f"Retrieved {len(results)} documents")
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
