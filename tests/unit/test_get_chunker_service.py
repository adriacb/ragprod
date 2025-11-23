import pytest
from unittest.mock import Mock
from ragprod.application.use_cases.get_chunker_service import GetChunkerService
from ragprod.infrastructure.chunker import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SemanticChunker,
)

class TestGetChunkerService:
    """Test cases for GetChunkerService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = GetChunkerService()

    def test_get_recursive_character_chunker(self):
        """Test getting RecursiveCharacterTextSplitter."""
        config = {"chunk_size": 100, "chunk_overlap": 20}
        chunker = self.service.get("recursive_character", config)
        assert isinstance(chunker, RecursiveCharacterTextSplitter)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20

    def test_get_markdown_chunker(self):
        """Test getting MarkdownHeaderTextSplitter."""
        chunker = self.service.get("markdown")
        assert isinstance(chunker, MarkdownHeaderTextSplitter)

    def test_get_character_chunker(self):
        """Test getting CharacterTextSplitter."""
        config = {"chunk_size": 50, "chunk_overlap": 10}
        chunker = self.service.get("character", config)
        assert isinstance(chunker, CharacterTextSplitter)

    def test_get_token_chunker(self):
        """Test getting TokenTextSplitter."""
        config = {"chunk_size": 10, "chunk_overlap": 2}
        chunker = self.service.get("token", config)
        assert isinstance(chunker, TokenTextSplitter)

    def test_get_semantic_chunker(self):
        """Test getting SemanticChunker."""
        # SemanticChunker requires an embedding model, so we mock it if needed or pass a dummy config
        # Assuming SemanticChunker can be instantiated with just a mock embedding model in config
        mock_embedder = Mock()
        config = {"embedding_model": mock_embedder}
        chunker = self.service.get("semantic", config)
        assert isinstance(chunker, SemanticChunker)

    def test_get_unknown_chunker(self):
        """Test getting an unknown chunker raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunker"):
            self.service.get("unknown_chunker")

    def test_caching_behavior(self):
        """Test that caching works correctly."""
        service = GetChunkerService(enable_caching=True)
        config = {"chunk_size": 100}
        
        chunker1 = service.get("recursive_character", config)
        chunker2 = service.get("recursive_character", config)
        
        assert chunker1 is chunker2

    def test_disable_caching(self):
        """Test that caching can be disabled."""
        service = GetChunkerService(enable_caching=False)
        config = {"chunk_size": 100}
        
        chunker1 = service.get("recursive_character", config)
        chunker2 = service.get("recursive_character", config)
        
        assert chunker1 is not chunker2

    def test_cache_key_different_config(self):
        """Test that different configs produce different cache keys."""
        service = GetChunkerService(enable_caching=True)
        
        chunker1 = service.get("recursive_character", {"chunk_size": 100})
        chunker2 = service.get("recursive_character", {"chunk_size": 200})
        
        assert chunker1 is not chunker2

    def test_list_available_chunkers(self):
        """Test listing available chunkers."""
        available = self.service.list_available_chunkers()
        expected = [
            "recursive_character",
            "markdown",
            "character",
            "token",
            "semantic",
        ]
        assert sorted(available) == sorted(expected)

    def test_clear_cache(self):
        """Test clearing the cache."""
        service = GetChunkerService(enable_caching=True)
        config = {"chunk_size": 100}
        
        chunker1 = service.get("recursive_character", config)
        service.clear_cache()
        chunker2 = service.get("recursive_character", config)
        
        assert chunker1 is not chunker2
