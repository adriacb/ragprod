from typing import Dict, Type, Optional, List
from ragprod.infrastructure.logger import get_logger
from ragprod.infrastructure.chunker import (
    BaseChunker,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SemanticChunker,
)


class GetChunkerService:
    """
    Factory service for creating chunker instances.
    
    This service provides a centralized way to instantiate different chunkers
    (Recursive, Markdown, Token, Semantic) based on configuration.
    """
    _logger = get_logger(__name__)
    
    def __init__(self, enable_caching: bool = True):
        """
        Initialize the chunker service.
        
        Args:
            enable_caching: If True, cache chunker instances to avoid re-instantiation.
        """
        self._enable_caching = enable_caching
        self._cache: Dict[str, BaseChunker] = {}

    @property
    def registry(self) -> Dict[str, Type[BaseChunker]]:
        """
        Registry mapping chunker names to their class implementations.
        
        Returns:
            Dictionary mapping chunker names to chunker classes.
        """
        return {
            "recursive_character": RecursiveCharacterTextSplitter,
            "markdown": MarkdownHeaderTextSplitter,
            "character": CharacterTextSplitter,
            "token": TokenTextSplitter,
            "semantic": SemanticChunker,
        }

    def get(self, chunker_name: str, config: Optional[dict] = None) -> BaseChunker:
        """
        Get or create a chunker instance based on the chunker name and configuration.
        
        Args:
            chunker_name: Name of the chunker to instantiate.
            config: Configuration dictionary passed to the chunker constructor.
                   If None, uses empty dict (chunker will use default values).
        
        Returns:
            An instance of the requested chunker.
        
        Raises:
            ValueError: If chunker_name is not supported or config is invalid.
            RuntimeError: If chunker instantiation fails.
        """
        if config is None:
            config = {}
        
        chunker_name_lower = chunker_name.lower().strip()
        
        # Check if chunker is in registry
        if chunker_name_lower not in self.registry:
            available = ", ".join(self.registry.keys())
            raise ValueError(
                f"Unknown chunker '{chunker_name}'. "
                f"Available chunkers: {available}"
            )
        
        # Check cache if enabled
        if self._enable_caching:
            cache_key = self._generate_cache_key(chunker_name_lower, config)
            if cache_key in self._cache:
                self._logger.debug(
                    f"Returning cached {chunker_name_lower} chunker instance"
                )
                return self._cache[cache_key]
        
        # Get chunker class from registry
        chunker_class = self.registry[chunker_name_lower]
        
        try:
            # Instantiate chunker with config
            self._logger.info(f"Creating {chunker_name_lower} chunker with config: {config}")
            chunker_instance = chunker_class(**config)
            
            # Cache if enabled
            if self._enable_caching:
                cache_key = self._generate_cache_key(chunker_name_lower, config)
                self._cache[cache_key] = chunker_instance
                self._logger.debug(f"Cached {chunker_name_lower} chunker instance")
            
            return chunker_instance
            
        except TypeError as e:
            # Handle invalid constructor arguments
            raise ValueError(
                f"Invalid configuration for {chunker_name_lower} chunker. "
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            # Handle other initialization errors
            raise RuntimeError(
                f"Failed to initialize {chunker_name_lower} chunker: {str(e)}"
            ) from e

    def _generate_cache_key(self, chunker_name: str, config: dict) -> str:
        """
        Generate a cache key from chunker name and configuration.
        
        Args:
            chunker_name: Name of the chunker.
            config: Configuration dictionary.
        
        Returns:
            String cache key.
        """
        # Sort config items to ensure consistent keys
        # Note: This assumes config values are hashable or stringifiable in a stable way
        sorted_config = tuple(sorted(config.items()))
        return f"{chunker_name}:{sorted_config}"

    def clear_cache(self):
        """Clear the chunker instance cache."""
        self._cache.clear()
        self._logger.debug("Chunker cache cleared")

    def list_available_chunkers(self) -> List[str]:
        """
        List all available chunker names.
        
        Returns:
            List of available chunker names.
        """
        return list(self.registry.keys())
