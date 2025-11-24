"""Core entities for retrieval domain."""

from dataclasses import dataclass, field
from typing import Any

from ragprod.domain.document import Document


@dataclass(frozen=True)
class Query:
    """Represents a user query for retrieval."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate query fields."""
        if not self.text:
            raise ValueError("Query text cannot be empty")


@dataclass(frozen=True)
class RetrievalResult:
    """Represents a retrieval result with score and metadata."""

    document: Document
    score: float
    retrieval_method: str = "unknown"  # "dense", "sparse", "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate retrieval result fields."""
        if self.score < 0.0:
            raise ValueError("Score cannot be negative")
