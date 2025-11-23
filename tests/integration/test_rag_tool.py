import sys
from unittest.mock import MagicMock

# Mock fastmcp before any other imports
# Create a mock that makes the @mcp.tool decorator a pass-through
fastmcp_mock = MagicMock()
# Make the .tool decorator return the function unchanged (pass-through)
fastmcp_mock.Context = MagicMock
fastmcp_mock.FastMCP.return_value.tool = lambda func: func
sys.modules['fastmcp'] = fastmcp_mock

import pytest
from unittest.mock import AsyncMock, patch
from ragprod.domain.document import Document
from ragprod.presentation.mcp.lifespan.service import init_chunker_service


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Initialize and cleanup the chunker service for each test."""
    # Setup: Initialize the service
    init_chunker_service()
    yield
    # Teardown: Reset global state
    import ragprod.presentation.mcp.lifespan.service as service_module
    service_module._chunker_service_instance = None


@pytest.fixture
def mock_ctx():
    """Mock the MCP context."""
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


@pytest.mark.asyncio
@patch("ragprod.presentation.mcp.tools.rag.clientDB")
async def test_add_documents_default_chunker(mock_client_db, mock_ctx):
    """Test add_documents with default chunker configuration."""
    # Configure mock
    mock_client_db.add_documents = AsyncMock()
    
    from ragprod.presentation.mcp.tools.rag import add_documents
    
    doc = Document(
        raw_text="This is a test document. " * 50,
        source="test.txt"
    )
    
    
    await add_documents(
        documents=[doc],
        ctx=mock_ctx
    )
    
    # Verify add_documents was called
    assert mock_client_db.add_documents.called
    
    # Verify chunks were created (default chunk size is 1000)
    # The text length is approx 25 * 50 = 1250 chars
    # With overlap, we might get 1-2 chunks depending on splitting
    call_args = mock_client_db.add_documents.call_args[0][0]
    assert len(call_args) >= 1
    assert isinstance(call_args[0], Document)


@pytest.mark.asyncio
@patch("ragprod.presentation.mcp.tools.rag.clientDB")
async def test_add_documents_custom_config(mock_client_db, mock_ctx):
    """Test add_documents with custom configuration."""
    # Configure mock
    mock_client_db.add_documents = AsyncMock()
    
    from ragprod.presentation.mcp.tools.rag import add_documents
    
    doc = Document(
        raw_text="This is a test document. " * 50,
        source="test.txt"
    )
    
    # Use small chunk size to force many chunks
    config = {"chunk_size": 100, "chunk_overlap": 20}
    
    await add_documents(
        documents=[doc],
        chunker_name="recursive_character",
        chunker_config=config,
        ctx=mock_ctx
    )
    
    call_args = mock_client_db.add_documents.call_args[0][0]
    # 1250 chars / 100 chars/chunk ~= 13 chunks
    assert len(call_args) > 10


@pytest.mark.asyncio
@patch("ragprod.presentation.mcp.tools.rag.clientDB")
async def test_add_documents_different_chunker(mock_client_db, mock_ctx):
    """Test add_documents with a different chunker type."""
    # Configure mock
    mock_client_db.add_documents = AsyncMock()
    
    from ragprod.presentation.mcp.tools.rag import add_documents
    
    doc = Document(
        raw_text="This is a test document.",
        source="test.txt"
    )
    
    # Use token chunker
    config = {"chunk_size": 10, "chunk_overlap": 0}
    
    await add_documents(
        documents=[doc],
        chunker_name="token",
        chunker_config=config,
        ctx=mock_ctx
    )
    
    assert mock_client_db.add_documents.called
    call_args = mock_client_db.add_documents.call_args[0][0]
    assert len(call_args) >= 1


@pytest.mark.asyncio
@patch("ragprod.presentation.mcp.tools.rag.clientDB")
async def test_add_documents_error_handling(mock_client_db, mock_ctx):
    """Test error handling in add_documents."""
    # Configure mock
    mock_client_db.add_documents = AsyncMock()
    
    from ragprod.presentation.mcp.tools.rag import add_documents
    
    doc = Document(raw_text="Test", source="test.txt")
    
    # Force an error by passing invalid chunker name
    await add_documents(
        documents=[doc],
        chunker_name="invalid_chunker",
        ctx=mock_ctx
    )
    
    # Should log error to context
    assert mock_ctx.error.called
    assert "Failed to add documents" in mock_ctx.error.call_args[0][0]
    # Should not call DB
    assert not mock_client_db.add_documents.called
