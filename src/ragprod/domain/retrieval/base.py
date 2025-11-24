"""Base interfaces for retrieval strategies."""

from abc import ABC, abstractmethod
from typing import List

from ragprod.domain.retrieval.entities import Query, RetrievalResult


class BaseRetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    async def retrieve(
        self, query: Query, collection_name: str, top_k: int = 10
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for
            collection_name: Name of the collection to search
            top_k: Number of top results to return

        Returns:
            List of retrieval results ordered by relevance
        """
        pass
