from typing import Optional
from ragprod.application.use_cases.get_chunker_service import GetChunkerService

_chunker_service_instance: Optional[GetChunkerService] = None

def init_chunker_service() -> None:
    """Initialize the global chunker service instance."""
    global _chunker_service_instance
    if _chunker_service_instance is None:
        _chunker_service_instance = GetChunkerService()

def get_chunker_service_instance() -> GetChunkerService:
    """Get the global chunker service instance."""
    if _chunker_service_instance is None:
        raise RuntimeError("Chunker service not initialized. Call init_chunker_service() first.")
    return _chunker_service_instance
