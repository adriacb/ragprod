from ragprod.domain.embedding import EmbeddingModel
from ragprod.infrastructure.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    ColBERTEmbeddings
)
from typing import Optional, Dict, Type, List
import logging

logger = logging.getLogger(__name__)


class GetEmbeddingsService:
    """
    Factory service for creating embedding model instances.
    
    This service provides a centralized way to instantiate different embedding
    models (HuggingFace, OpenAI) based on configuration.
    """
    
    def __init__(self, enable_caching: bool = True):
        """
        Initialize the embedding service.
        
        Args:
            enable_caching: If True, cache embedding model instances to avoid re-instantiation.
        """
        self._enable_caching = enable_caching
        self._cache: Dict[str, EmbeddingModel] = {}
        self._logger = logger

    @property
    def registry(self) -> Dict[str, Type[EmbeddingModel]]:
        """
        Registry mapping embedding model names to their class implementations.
        
        Returns:
            Dictionary mapping model names to model classes.
        """
        return {
            "huggingface": HuggingFaceEmbeddings,
            "openai": OpenAIEmbeddings,
            "colbert": ColBERTEmbeddings
        }

    def get(self, embedding_model: str, config: Optional[dict] = None) -> EmbeddingModel:
        """
        Get or create an embedding model instance based on the model name and configuration.
        
        Args:
            embedding_model: Name of the embedding model to instantiate (e.g., "huggingface", "openai").
            config: Configuration dictionary passed to the model constructor.
                   If None, uses empty dict (model will use default values).
        
        Returns:
            An instance of the requested embedding model.
        
        Raises:
            ValueError: If embedding_model is not supported or config is invalid.
            RuntimeError: If model instantiation fails.
        
        Example:
            >>> service = GetEmbeddings()
            >>> model = service.get("huggingface", {
            ...     "model_name": "jinaai/jina-code-embeddings-0.5b",
            ...     "model_kwargs": {"device_map": "cpu"}
            ... })
        """
        if config is None:
            config = {}
        
        embedding_model_lower = embedding_model.lower().strip()
        
        # Check if model is in registry
        if embedding_model_lower not in self.registry:
            available = ", ".join(self.registry.keys())
            raise ValueError(
                f"Unknown embedding model '{embedding_model}'. "
                f"Available models: {available}"
            )
        
        # Check cache if enabled
        if self._enable_caching:
            cache_key = self._generate_cache_key(embedding_model_lower, config)
            if cache_key in self._cache:
                self._logger.debug(
                    f"Returning cached {embedding_model_lower} embedding model instance"
                )
                return self._cache[cache_key]
        
        # Get model class from registry
        model_class = self.registry[embedding_model_lower]
        
        try:
            # Instantiate model with config
            self._logger.info(
                f"Creating {embedding_model_lower} embedding model with config: "
                f"{self._sanitize_config(config)}"
            )
            model_instance = model_class(**config)
            
            # Cache if enabled
            if self._enable_caching:
                cache_key = self._generate_cache_key(embedding_model_lower, config)
                self._cache[cache_key] = model_instance
                self._logger.debug(f"Cached {embedding_model_lower} embedding model instance")
            
            return model_instance
            
        except TypeError as e:
            # Handle invalid constructor arguments
            raise ValueError(
                f"Invalid configuration for {embedding_model_lower} embedding model. "
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            # Handle model loading or other initialization errors
            raise RuntimeError(
                f"Failed to initialize {embedding_model_lower} embedding model: {str(e)}"
            ) from e

    def _generate_cache_key(self, model_name: str, config: dict) -> str:
        """
        Generate a cache key from model name and configuration.
        
        Args:
            model_name: Name of the embedding model.
            config: Configuration dictionary.
        
        Returns:
            String cache key.
        """
        # Sort config items to ensure consistent keys
        sorted_config = tuple(sorted(config.items()))
        return f"{model_name}:{sorted_config}"

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
        """Clear the embedding model instance cache."""
        self._cache.clear()
        self._logger.debug("Embedding model cache cleared")

    def list_available_models(self) -> List[str]:
        """
        List all available embedding model names.
        
        Returns:
            List of available embedding model names.
        """
        return list(self.registry.keys())