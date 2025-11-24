"""Test configuration and fixtures for retrieval tests."""

import pytest

from ragprod.domain.document import Document
from ragprod.domain.retrieval.entities import Query


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            id="1",
            raw_text="Python is a high-level programming language.",
            source="test",
            title="Python Basics",
        ),
        Document(
            id="2",
            raw_text="List comprehensions provide concise syntax.",
            source="test",
            title="Python Features",
        ),
        Document(
            id="3",
            raw_text="Asyncio enables concurrent programming.",
            source="test",
            title="Python Async",
        ),
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return Query(text="What is Python?")
