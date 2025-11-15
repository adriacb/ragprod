import pytest
from ragprod.domain.document import Document

def test_metadata_property():
    doc = Document(
        id="123",
        raw_text="Sample text",
        source="UnitTest",
        title="Sample Doc",
        distance=0.5,
        score=0.9
    )

    # Set private metadata after initialization
    doc.metadata = {"author": "Tester"}

    # Check initial metadata
    assert doc.metadata == {"author": "Tester"}

    # Update metadata
    doc.metadata = {"author": "Updated Tester", "category": "test"}
    assert doc.metadata == {"author": "Updated Tester", "category": "test"}


def test_content_property():
    raw_text = "This is a test content."
    doc = Document(raw_text=raw_text)
    assert doc.content == raw_text

def test_repr_returns_empty_string(capsys):
    doc = Document(
        raw_text="Some content",
        source="pytest",
        title="Test Doc"
    )

    # Set metadata after creation
    doc.metadata = {"key": "value"}

    output = doc.__repr__()

    # __repr__ should return empty string
    assert output == ""

    # But it should print something to console (captured by capsys)
    captured = capsys.readouterr()
    assert "Some content" in captured.out
    assert "Metadata" in captured.out
    assert "key" in captured.out


def test_default_values():
    doc = Document(raw_text="Default test")
    assert doc.source == "Unknown"
    assert doc.title == "Untitled"
    assert doc.metadata == {}
    assert doc.distance is None
    assert doc.score is None
