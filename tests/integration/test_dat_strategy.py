"""Integration tests for DAT strategy with Elasticsearch and ChromaDB."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ragprod.domain.document import Document
from ragprod.domain.retrieval.entities import Query
from ragprod.domain.retrieval.strategies.dat import (
    AlphaTuner,
    DATConfig,
    DATStrategy,
    EffectivenessScorer,
)


@pytest.fixture
def mock_dense_store():
    """Mock ChromaDB client."""
    store = AsyncMock()
    store.retrieve = AsyncMock(
        return_value=[
            Document(
                id="1",
                raw_text="Python is a programming language",
                source="test",
                title="Python",
                score=0.9,
            ),
            Document(
                id="2",
                raw_text="List comprehensions in Python",
                source="test",
                title="Lists",
                score=0.7,
            ),
        ]
    )
    return store


@pytest.fixture
def mock_sparse_store():
    """Mock Elasticsearch client."""
    store = AsyncMock()
    store.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_score": 5.2,
                        "_source": {
                            "text": "Python is a programming language",
                            "source": "test",
                            "title": "Python",
                            "metadata": {},
                        },
                    },
                    {
                        "_id": "3",
                        "_score": 3.1,
                        "_source": {
                            "text": "Python syntax is simple",
                            "source": "test",
                            "title": "Syntax",
                            "metadata": {},
                        },
                    },
                ]
            }
        }
    )
    return store


@pytest.fixture
def mock_llm_client():
    """Mock OpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    
    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = '{"dense_score": 0.8, "sparse_score": 0.6}'
    
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


@pytest.mark.asyncio
async def test_dat_strategy_with_dynamic_tuning(
    mock_dense_store, mock_sparse_store, mock_llm_client
):
    """Test DAT strategy with dynamic alpha tuning."""
    # Setup
    config = DATConfig(use_dynamic_tuning=True, top_k_dense=5, top_k_sparse=5)
    scorer = EffectivenessScorer(mock_llm_client)
    tuner = AlphaTuner(scorer)
    strategy = DATStrategy(mock_dense_store, mock_sparse_store, tuner, config)

    # Execute
    query = Query(text="What is Python?")
    results = await strategy.retrieve(query, "test_collection", top_k=3)

    # Verify
    assert len(results) <= 3
    assert all(hasattr(r, "document") for r in results)
    assert all(hasattr(r, "score") for r in results)
    
    # Verify stores were called
    mock_dense_store.retrieve.assert_called_once()
    mock_sparse_store.search.assert_called_once()
    
    # Verify LLM was called for scoring
    mock_llm_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_dat_strategy_without_dynamic_tuning(
    mock_dense_store, mock_sparse_store, mock_llm_client
):
    """Test DAT strategy with fixed alpha."""
    # Setup
    config = DATConfig(use_dynamic_tuning=False, dense_weight_default=0.7)
    scorer = EffectivenessScorer(mock_llm_client)
    tuner = AlphaTuner(scorer)
    strategy = DATStrategy(mock_dense_store, mock_sparse_store, tuner, config)

    # Execute
    query = Query(text="Python programming")
    results = await strategy.retrieve(query, "test_collection", top_k=5)

    # Verify
    assert len(results) <= 5
    
    # Verify LLM was NOT called (dynamic tuning disabled)
    mock_llm_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_dat_strategy_handles_empty_results(mock_llm_client):
    """Test DAT strategy handles empty results gracefully."""
    # Setup with empty stores
    empty_dense = AsyncMock()
    empty_dense.retrieve = AsyncMock(return_value=[])
    
    empty_sparse = AsyncMock()
    empty_sparse.search = AsyncMock(return_value={"hits": {"hits": []}})
    
    config = DATConfig()
    scorer = EffectivenessScorer(mock_llm_client)
    tuner = AlphaTuner(scorer)
    strategy = DATStrategy(empty_dense, empty_sparse, tuner, config)

    # Execute
    query = Query(text="nonexistent query")
    results = await strategy.retrieve(query, "test_collection", top_k=5)

    # Verify
    assert len(results) == 0
