from typing import Optional
from types import SimpleNamespace
from ragprod.application.use_cases.get_chunker_service import GetChunkerService
from ragprod.infrastructure.config.settings import load_settings
from ragprod.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Global instances
_settings: Optional[SimpleNamespace] = None
_chunker_service_instance: Optional[GetChunkerService] = None


def init_services(env_path: str | None = None) -> None:
    """
    Initialize global services and settings.
    
    Args:
        env_path: Path to .env file to load. If None, uses process environment.
    """
    global _settings, _chunker_service_instance
    
    # Load settings from env file
    logger.info(f"Loading settings from: {env_path or 'process environment'}")
    _settings = load_settings(env_path)
    logger.info(f"Settings loaded for environment: {_settings.env}")
    
    # Initialize chunker service
    if _chunker_service_instance is None:
        logger.info("Initializing chunker service")
        _chunker_service_instance = GetChunkerService()
        logger.info("Chunker service initialized")


def get_settings() -> SimpleNamespace:
    """Get the global settings instance."""
    if _settings is None:
        raise RuntimeError("Settings not initialized. Call init_services() first.")
    return _settings


def get_chunker_service_instance() -> GetChunkerService:
    """Get the global chunker service instance."""
    if _chunker_service_instance is None:
        raise RuntimeError("Chunker service not initialized. Call init_services() first.")
    return _chunker_service_instance
