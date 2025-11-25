"""Unit tests for AlphaTuner."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ragprod.domain.document import Document
from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.strategies.dat.alpha_tuner import AlphaTuner
from ragprod.domain.retrieval.strategies.dat.effectiveness_scorer import EffectivenessScorer


@pytest.fixture
def mock_scorer():
    """Mock effectiveness scorer."""
    scorer = MagicMock(spec=EffectivenessScorer)
    scorer.score_results = AsyncMock()
    return scorer


@pytest.fixture
def tuner(mock_scorer):
    """Create alpha tuner with mock scorer."""
    return AlphaTuner(mock_scorer, default_alpha=0.5)


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return Query(text="What is Python?")


@pytest.fixture
def sample_dense_results():
    """Sample dense retrieval results."""
    return [
        RetrievalResult(
            document=Document(id="1", raw_text="Python programming", source="test", title="Python"),
            score=0.9,
            retrieval_method="dense",
        )
    ]


@pytest.fixture
def sample_sparse_results():
    """Sample sparse retrieval results."""
    return [
        RetrievalResult(
            document=Document(id="2", raw_text="Python syntax", source="test", title="Syntax"),
            score=5.2,
            retrieval_method="sparse",
        )
    ]


@pytest.mark.asyncio
async def test_calculate_alpha_case1_both_fail(tuner, mock_scorer, sample_query, sample_dense_results, sample_sparse_results):
    """Test Case 1: Both methods fail (scores = 0) → alpha = 0.5."""
    mock_scorer.score_results.return_value = {"dense_score": 0, "sparse_score": 0}
    
    alpha = await tuner.calculate_alpha(sample_query, sample_dense_results, sample_sparse_results)
    
    assert alpha == 0.5


@pytest.mark.asyncio
async def test_calculate_alpha_case2_dense_perfect(tuner, mock_scorer, sample_query, sample_dense_results, sample_sparse_results):
    """Test Case 2: Dense perfect (5), sparse imperfect → alpha = 1.0."""
    mock_scorer.score_results.return_value = {"dense_score": 5, "sparse_score": 3}
    
    alpha = await tuner.calculate_alpha(sample_query, sample_dense_results, sample_sparse_results)
    
    assert alpha == 1.0


@pytest.mark.asyncio
async def test_calculate_alpha_case3_sparse_perfect(tuner, mock_scorer, sample_query, sample_dense_results, sample_sparse_results):
    """Test Case 3: Sparse perfect (5), dense imperfect → alpha = 0.0."""
    mock_scorer.score_results.return_value = {"dense_score": 3, "sparse_score": 5}
    
    alpha = await tuner.calculate_alpha(sample_query, sample_dense_results, sample_sparse_results)
    
    assert alpha == 0.0


@pytest.mark.asyncio
async def test_calculate_alpha_case4_proportional(tuner, mock_scorer, sample_query, sample_dense_results, sample_sparse_results):
    """Test Case 4: Proportional weighting → alpha = dense/(dense+sparse)."""
    mock_scorer.score_results.return_value = {"dense_score": 4, "sparse_score": 3}
    
    alpha = await tuner.calculate_alpha(sample_query, sample_dense_results, sample_sparse_results)
    
    # 4 / (4 + 3) = 0.571... → 0.6 (rounded to 1 decimal)
    assert alpha == 0.6


@pytest.mark.asyncio
async def test_calculate_alpha_rounding(tuner, mock_scorer, sample_query, sample_dense_results, sample_sparse_results):
    """Test that alpha is rounded to 1 decimal place per paper."""
    test_cases = [
        ({"dense_score": 1, "sparse_score": 2}, 0.3),  # 1/3 = 0.333... → 0.3
        ({"dense_score": 2, "sparse_score": 1}, 0.7),  # 2/3 = 0.666... → 0.7
        ({"dense_score": 3, "sparse_score": 4}, 0.4),  # 3/7 = 0.428... → 0.4
    ]
    
    for scores, expected_alpha in test_cases:
        mock_scorer.score_results.return_value = scores
        alpha = await tuner.calculate_alpha(sample_query, sample_dense_results, sample_sparse_results)
        assert alpha == expected_alpha


@pytest.mark.asyncio
async def test_calculate_alpha_empty_dense_results(tuner, mock_scorer, sample_query, sample_sparse_results):
    """Test alpha calculation with empty dense results."""
    alpha = await tuner.calculate_alpha(sample_query, [], sample_sparse_results)
    
    # Should return 0.0 (use only sparse)
    assert alpha == 0.0


@pytest.mark.asyncio
async def test_calculate_alpha_empty_sparse_results(tuner, mock_scorer, sample_query, sample_dense_results):
    """Test alpha calculation with empty sparse results."""
    alpha = await tuner.calculate_alpha(sample_query, sample_dense_results, [])
    
    # Should return 1.0 (use only dense)
    assert alpha == 1.0


@pytest.mark.asyncio
async def test_calculate_alpha_both_empty(tuner, mock_scorer, sample_query):
    """Test alpha calculation with both empty."""
    alpha = await tuner.calculate_alpha(sample_query, [], [])
    
    # Should return default alpha
    assert alpha == 0.5


def test_apply_alpha_combines_scores(tuner, sample_dense_results, sample_sparse_results):
    """Test that apply_alpha correctly combines scores."""
    alpha = 0.7
    
    results = tuner.apply_alpha(alpha, sample_dense_results, sample_sparse_results)
    
    # Should have results from both methods
    assert len(results) == 2
    
    # Verify scores are weighted
    for result in results:
        assert hasattr(result, "score")
        assert hasattr(result, "document")
        assert hasattr(result, "metadata")
        assert "alpha" in result.metadata


def test_apply_alpha_handles_overlap(tuner):
    """Test that apply_alpha handles documents appearing in both results."""
    # Same document in both results
    doc = Document(id="1", raw_text="Python", source="test", title="Python")
    
    dense_results = [
        RetrievalResult(document=doc, score=0.9, retrieval_method="dense")
    ]
    sparse_results = [
        RetrievalResult(document=doc, score=5.0, retrieval_method="sparse")
    ]
    
    alpha = 0.6
    results = tuner.apply_alpha(alpha, dense_results, sparse_results)
    
    # Should have only 1 result (merged)
    assert len(results) == 1
    assert results[0].retrieval_method == "hybrid"
    
    # Score should be: (0.9 * 0.6) + (5.0 * 0.4) = 0.54 + 2.0 = 2.54
    expected_score = (0.9 * alpha) + (5.0 * (1 - alpha))
    assert abs(results[0].score - expected_score) < 0.01
