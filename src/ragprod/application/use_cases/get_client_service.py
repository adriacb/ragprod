from ragprod.domain.client.base import BaseClient
from ragprod.infrastructure.client import (
    AsyncChromaDBClient,
    # QdrantRetriever,
    # WeaviateClient
)
from typing import Dict, Type, Optional, List
import logging


class GetClientService:
    """
    Factory service for creating vector database client instances.
    
    This service provides a centralized way to instantiate different vector database
    clients (ChromaDB, Qdrant, Weaviate) based on configuration.
    """
    
    def __init__(self, enable_caching: bool = True):
        """
        Initialize the client service.
        
        Args:
            enable_caching: If True, cache client instances to avoid re-instantiation.
        """
        self._enable_caching = enable_caching
        self._cache: Dict[str, BaseClient] = {}
        self._logger = logging.getLogger(__name__)

    @property
    def registry(self) -> Dict[str, Type[BaseClient]]:
        """
        Registry mapping client names to their class implementations.
        
        Returns:
            Dictionary mapping client names to client classes.
        """
        return {
            "chroma": AsyncChromaDBClient,
            # "qdrant": QdrantRetriever,
            # "weaviate": WeaviateClient,
        }

    def get(self, client_name: str, config: Optional[dict] = None) -> BaseClient:
        """
        Get or create a client instance based on the client name and configuration.
        
        Args:
            client_name: Name of the client to instantiate (e.g., "chroma", "qdrant", "weaviate").
            config: Configuration dictionary passed to the client constructor.
                   If None, uses empty dict (client will use default values).
        
        Returns:
            An instance of the requested client.
        
        Raises:
            ValueError: If client_name is not supported or config is invalid.
            RuntimeError: If client instantiation fails.
        
        Example:
            >>> service = GetClientService()
            >>> client = service.get("chroma", {
            ...     "persist_directory": "./chromadb",
            ...     "collection_name": "my_collection"
            ... })
        """
        if config is None:
            config = {}
        
        client_name_lower = client_name.lower().strip()
        
        # Check if client is in registry
        if client_name_lower not in self.registry:
            available = ", ".join(self.registry.keys())
            raise ValueError(
                f"Unknown client '{client_name}'. "
                f"Available clients: {available}"
            )
        
        # Check cache if enabled
        if self._enable_caching:
            cache_key = self._generate_cache_key(client_name_lower, config)
            if cache_key in self._cache:
                self._logger.debug(
                    f"Returning cached {client_name_lower} client instance"
                )
                return self._cache[cache_key]
        
        # Get client class from registry
        client_class = self.registry[client_name_lower]
        
        try:
            # Instantiate client with config
            self._logger.info(f"Creating {client_name_lower} client with config: {self._sanitize_config(config)}")
            client_instance = client_class(**config)
            
            # Cache if enabled
            if self._enable_caching:
                cache_key = self._generate_cache_key(client_name_lower, config)
                self._cache[cache_key] = client_instance
                self._logger.debug(f"Cached {client_name_lower} client instance")
            
            return client_instance
            
        except TypeError as e:
            # Handle invalid constructor arguments
            raise ValueError(
                f"Invalid configuration for {client_name_lower} client. "
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            # Handle connection or other initialization errors
            raise RuntimeError(
                f"Failed to initialize {client_name_lower} client: {str(e)}"
            ) from e

    def _generate_cache_key(self, client_name: str, config: dict) -> str:
        """
        Generate a cache key from client name and configuration.
        
        Args:
            client_name: Name of the client.
            config: Configuration dictionary.
        
        Returns:
            String cache key.
        """
        # Sort config items to ensure consistent keys
        sorted_config = tuple(sorted(config.items()))
        return f"{client_name}:{sorted_config}"

    def _sanitize_config(self, config: dict) -> dict:
        """
        Sanitize configuration for logging (remove sensitive data).
        
        Args:
            config: Configuration dictionary.
        
        Returns:
            Sanitized configuration dictionary.
        """
        sensitive_keys = {"api_key", "password", "token", "secret"}
        sanitized = {}
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized

    def clear_cache(self):
        """Clear the client instance cache."""
        self._cache.clear()
        self._logger.debug("Client cache cleared")

    def list_available_clients(self) -> List[str]:
        """
        List all available client names.
        
        Returns:
            List of available client names.
        """
        return list(self.registry.keys())