"""Retrieval service use case for application layer."""

import logging
from typing import List

from openai import AsyncOpenAI

from ragprod.domain.retrieval.base import BaseRetrievalStrategy
from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.exceptions import RetrievalError, StrategyNotFoundError
from ragprod.domain.retrieval.strategies.dat import (
    AlphaTuner,
    DATConfig,
    DATStrategy,
    EffectivenessScorer,
)


class RetrievalService:
    """Application service for document retrieval with strategy selection."""

    def __init__(
        self,
        dense_store=None,
        sparse_store=None,
        strategy: str = "dense",
        dat_config: DATConfig | None = None,
    ):
        """Initialize retrieval service.

        Args:
            dense_store: Vector store client (e.g., ChromaDB)
            sparse_store: Sparse search client (e.g., Elasticsearch)
            strategy: Retrieval strategy to use ("dense", "dat")
            dat_config: Configuration for DAT strategy
        """
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.strategy_name = strategy
        self.dat_config = dat_config or DATConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize strategy
        self.strategy = self._create_strategy(strategy)

    def _create_strategy(self, strategy_name: str) -> BaseRetrievalStrategy | None:
        """Create retrieval strategy based on name.

        Args:
            strategy_name: Name of the strategy ("dense", "dat")

        Returns:
            Strategy instance or None for simple dense retrieval
        """
        if strategy_name == "dense":
            # Simple dense retrieval - no strategy needed
            return None

        elif strategy_name == "dat":
            if not self.sparse_store:
                raise StrategyNotFoundError(
                    "DAT strategy requires sparse_store (Elasticsearch)"
                )

            # Initialize DAT components
            llm_client = AsyncOpenAI()
            scorer = EffectivenessScorer(
                llm_client, model=self.dat_config.llm_model, temperature=self.dat_config.temperature
            )
            tuner = AlphaTuner(scorer, default_alpha=self.dat_config.dense_weight_default)

            return DATStrategy(
                dense_store=self.dense_store,
                sparse_store=self.sparse_store,
                alpha_tuner=tuner,
                config=self.dat_config,
            )

        else:
            raise StrategyNotFoundError(f"Unknown strategy: {strategy_name}")

    async def retrieve(
        self, query: Query | str, collection_name: str, top_k: int = 10
    ) -> List[RetrievalResult]:
        """Retrieve documents for a query.

        Args:
            query: Query object or query text
            collection_name: Name of the collection to search
            top_k: Number of results to return

        Returns:
            List of retrieval results

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Convert string to Query if needed
            if isinstance(query, str):
                query = Query(text=query)

            self.logger.info(
                f"Retrieving with strategy '{self.strategy_name}' for query: '{query.text}'"
            )

            # Use strategy if available, otherwise simple dense retrieval
            if self.strategy:
                results = await self.strategy.retrieve(query, collection_name, top_k)
            else:
                results = await self._simple_dense_retrieve(query, collection_name, top_k)

            self.logger.info(f"Retrieved {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval failed: {e}") from e

    async def _simple_dense_retrieve(
        self, query: Query, collection_name: str, top_k: int
    ) -> List[RetrievalResult]:
        """Simple dense retrieval using vector store.

        Args:
            query: Query object
            collection_name: Collection name
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        # Use dense store's retrieve method
        documents = await self.dense_store.retrieve(
            query=query.text, collection_name=collection_name, limit=top_k
        )

        # Convert to RetrievalResult
        return [
            RetrievalResult(
                document=doc,
                score=doc.score if doc.score is not None else 1.0,
                retrieval_method="dense",
            )
            for doc in documents
        ]


def create_retrieval_service(
    dense_store, sparse_store=None, strategy: str = "dense", **kwargs
) -> RetrievalService:
    """Factory function to create retrieval service.

    Args:
        dense_store: Vector store client
        sparse_store: Sparse search client (required for DAT)
        strategy: Strategy name ("dense", "dat")
        **kwargs: Additional configuration

    Returns:
        RetrievalService instance
    """
    dat_config = None
    if strategy == "dat":
        dat_config = DATConfig(**kwargs)

    return RetrievalService(
        dense_store=dense_store, sparse_store=sparse_store, strategy=strategy, dat_config=dat_config
    )
