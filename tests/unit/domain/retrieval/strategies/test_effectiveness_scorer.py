"""Unit tests for EffectivenessScorer."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ragprod.domain.document import Document
from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.strategies.dat.effectiveness_scorer import EffectivenessScorer


@pytest.fixture
def mock_llm_client():
    """Mock OpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


@pytest.fixture
def scorer(mock_llm_client):
    """Create effectiveness scorer with mock LLM."""
    return EffectivenessScorer(mock_llm_client, model="gpt-4o-mini", temperature=0.0)


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return Query(text="What is Python?")


@pytest.fixture
def sample_dense_results():
    """Sample dense retrieval results."""
    return [
        RetrievalResult(
            document=Document(
                id="1",
                raw_text="Python is a high-level programming language",
                source="test",
                title="Python",
            ),
            score=0.9,
            retrieval_method="dense",
        )
    ]


@pytest.fixture
def sample_sparse_results():
    """Sample sparse retrieval results."""
    return [
        RetrievalResult(
            document=Document(
                id="2",
                raw_text="Python syntax and examples",
                source="test",
                title="Syntax",
            ),
            score=5.2,
            retrieval_method="sparse",
        )
    ]


@pytest.mark.asyncio
async def test_score_results_valid_response(
    scorer, mock_llm_client, sample_query, sample_dense_results, sample_sparse_results
):
    """Test scoring with valid LLM response."""
    # Mock LLM response - space-separated integers
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "4 3"
    
    mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Execute
    scores = await scorer.score_results(sample_query, sample_dense_results, sample_sparse_results)
    
    # Verify
    assert scores["dense_score"] == 4
    assert scores["sparse_score"] == 3
    assert isinstance(scores["dense_score"], int)
    assert isinstance(scores["sparse_score"], int)


@pytest.mark.asyncio
async def test_score_results_edge_cases(scorer, mock_llm_client, sample_query, sample_dense_results, sample_sparse_results):
    """Test scoring with edge case values."""
    test_cases = [
        ("0 0", 0, 0),  # Both fail
        ("5 0", 5, 0),  # Dense perfect, sparse fail
        ("0 5", 0, 5),  # Sparse perfect, dense fail
        ("5 5", 5, 5),  # Both perfect
    ]
    
    for response_text, expected_dense, expected_sparse in test_cases:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = response_text
        
        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        scores = await scorer.score_results(sample_query, sample_dense_results, sample_sparse_results)
        
        assert scores["dense_score"] == expected_dense
        assert scores["sparse_score"] == expected_sparse


@pytest.mark.asyncio
async def test_score_results_clamping(scorer, mock_llm_client, sample_query, sample_dense_results, sample_sparse_results):
    """Test that scores are clamped to [0, 5] range."""
    # Mock response with out-of-range values
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "10 -2"
    
    mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    scores = await scorer.score_results(sample_query, sample_dense_results, sample_sparse_results)
    
    # Should be clamped to valid range
    assert scores["dense_score"] == 5  # 10 clamped to 5
    assert scores["sparse_score"] == 0  # -2 clamped to 0


@pytest.mark.asyncio
async def test_score_results_invalid_format(scorer, mock_llm_client, sample_query, sample_dense_results, sample_sparse_results):
    """Test scoring with invalid LLM response format."""
    # Mock invalid response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "invalid response"
    
    mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    scores = await scorer.score_results(sample_query, sample_dense_results, sample_sparse_results)
    
    # Should fallback to neutral scores
    assert scores["dense_score"] == 2
    assert scores["sparse_score"] == 2


@pytest.mark.asyncio
async def test_score_results_empty_results(scorer, mock_llm_client, sample_query):
    """Test scoring with empty results."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "0 0"
    
    mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    scores = await scorer.score_results(sample_query, [], [])
    
    assert scores["dense_score"] == 0
    assert scores["sparse_score"] == 0


def test_build_evaluation_prompt(scorer, sample_query, sample_dense_results, sample_sparse_results):
    """Test prompt generation includes paper's template."""
    prompt = scorer._build_evaluation_prompt(sample_query, sample_dense_results[0], sample_sparse_results[0])
    
    # Verify prompt contains key elements from paper
    assert "0 to 5" in prompt
    assert "Direct hit --> 5 points" in prompt
    assert "Good wrong result" in prompt
    assert "Bad wrong result" in prompt
    assert "Completely off-track --> 0 points" in prompt
    assert sample_query.text in prompt
    assert "space" in prompt.lower()  # Output format mentions space-separated
