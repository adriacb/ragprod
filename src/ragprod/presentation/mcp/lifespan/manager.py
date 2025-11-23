from contextlib import asynccontextmanager
from fastmcp import FastMCP
from .service import init_chunker_service

@asynccontextmanager
async def lifespan(server: FastMCP):
    """
    Lifespan context manager for the MCP server.
    Handles initialization and cleanup of resources.
    """
    # Startup
    init_chunker_service()
    yield
    # Shutdown (if needed)
