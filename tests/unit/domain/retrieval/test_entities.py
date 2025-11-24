"""Unit tests for retrieval entities."""

import pytest

from ragprod.domain.document import Document
from ragprod.domain.retrieval.entities import Query, RetrievalResult


def test_query_creation():
    """Test query creation."""
    query = Query(text="What is Python?", metadata={"user_id": "123"})

    assert query.text == "What is Python?"
    assert query.metadata == {"user_id": "123"}


def test_query_validation():
    """Test query validation."""
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        Query(text="")


def test_retrieval_result_creation():
    """Test retrieval result creation."""
    doc = Document(id="1", raw_text="Test content", source="test", title="Test")
    result = RetrievalResult(
        document=doc, score=0.95, retrieval_method="dense", metadata={"alpha": 0.5}
    )

    assert result.document == doc
    assert result.score == 0.95
    assert result.retrieval_method == "dense"
    assert result.metadata == {"alpha": 0.5}


def test_retrieval_result_validation():
    """Test retrieval result validation."""
    doc = Document(id="1", raw_text="Test", source="test", title="Test")

    with pytest.raises(ValueError, match="Score cannot be negative"):
        RetrievalResult(document=doc, score=-0.1)


def test_entities_immutability():
    """Test that entities are immutable."""
    query = Query(text="Test query")

    with pytest.raises(Exception):  # FrozenInstanceError
        query.text = "Modified"  # type: ignore
