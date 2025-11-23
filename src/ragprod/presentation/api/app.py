from fastapi import FastAPI
from ragprod.presentation.api.lifespan.manager import lifespan
from ragprod.presentation.api.routes import rag

# Create FastAPI application
app = FastAPI(
    title="RAGProd API",
    description="RAG Production API with document chunking and retrieval",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(rag.router, prefix="/rag", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to RAGProd API",
        "docs": "/docs",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
