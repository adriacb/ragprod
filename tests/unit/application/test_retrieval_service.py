"""Unit tests for RetrievalService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ragprod.application.use_cases.retrieval_service import (
    RetrievalService,
    create_retrieval_service,
)
from ragprod.domain.document import Document
from ragprod.domain.retrieval.entities import Query
from ragprod.domain.retrieval.exceptions import StrategyNotFoundError


@pytest.fixture
def mock_dense_store():
    """Mock dense store."""
    store = AsyncMock()
    store.retrieve = AsyncMock(
        return_value=[
            Document(
                id="1", raw_text="Python is a programming language", source="test", title="Python", score=0.9
            )
        ]
    )
    return store


@pytest.fixture
def mock_sparse_store():
    """Mock sparse store."""
    store = AsyncMock()
    store.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_score": 5.2,
                        "_source": {
                            "text": "Python programming",
                            "source": "test",
                            "title": "Python",
                            "metadata": {},
                        },
                    }
                ]
            }
        }
    )
    return store


@pytest.mark.asyncio
async def test_retrieval_service_dense_strategy(mock_dense_store):
    """Test retrieval service with dense strategy."""
    service = RetrievalService(dense_store=mock_dense_store, strategy="dense")

    query = Query(text="What is Python?")
    results = await service.retrieve(query, "test_collection", top_k=5)

    assert len(results) == 1
    assert results[0].retrieval_method == "dense"
    mock_dense_store.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_service_with_string_query(mock_dense_store):
    """Test retrieval service accepts string queries."""
    service = RetrievalService(dense_store=mock_dense_store, strategy="dense")

    results = await service.retrieve("What is Python?", "test_collection", top_k=5)

    assert len(results) == 1
    mock_dense_store.retrieve.assert_called_once()


def test_retrieval_service_dat_without_sparse_store(mock_dense_store):
    """Test DAT strategy requires sparse store."""
    with pytest.raises(StrategyNotFoundError, match="DAT strategy requires sparse_store"):
        RetrievalService(dense_store=mock_dense_store, strategy="dat")


def test_retrieval_service_unknown_strategy(mock_dense_store):
    """Test unknown strategy raises error."""
    with pytest.raises(StrategyNotFoundError, match="Unknown strategy"):
        RetrievalService(dense_store=mock_dense_store, strategy="unknown")


@patch("ragprod.application.use_cases.retrieval_service.AsyncOpenAI")
def test_create_retrieval_service_factory(mock_openai, mock_dense_store, mock_sparse_store):
    """Test factory function."""
    # Mock OpenAI client
    mock_openai.return_value = MagicMock()
    
    service = create_retrieval_service(
        dense_store=mock_dense_store,
        sparse_store=mock_sparse_store,
        strategy="dat",
        use_dynamic_tuning=False,
    )

    assert service.strategy_name == "dat"
    assert service.dat_config.use_dynamic_tuning is False

