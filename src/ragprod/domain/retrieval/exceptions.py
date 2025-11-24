"""Custom exceptions for retrieval domain."""


class RetrievalError(Exception):
    """Base exception for retrieval errors."""

    pass


class StrategyNotFoundError(RetrievalError):
    """Raised when a retrieval strategy is not found."""

    pass


class InvalidQueryError(RetrievalError):
    """Raised when a query is invalid."""

    pass
