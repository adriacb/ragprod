"""DAT (Dynamic Alpha Tuning) retrieval strategy implementation.

References:
    Hsu, H.-L., & Tzeng, J. (2025). "DAT: Dynamic Alpha Tuning for Hybrid Retrieval 
    in Retrieval-Augmented Generation". arXiv:2503.23013
    https://arxiv.org/abs/2503.23013
"""

import logging
from typing import Any, List

from ragprod.domain.retrieval.base import BaseRetrievalStrategy
from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.exceptions import RetrievalError
from ragprod.domain.retrieval.strategies.dat.alpha_tuner import AlphaTuner
from ragprod.domain.retrieval.strategies.dat.config import DATConfig


class DATStrategy(BaseRetrievalStrategy):
    """Dynamic Alpha Tuning hybrid retrieval strategy.

    Implements the complete DAT algorithm from the paper:
    1. Hybrid retrieval (dense + sparse)
    2. LLM-based effectiveness scoring (top-1 evaluation)
    3. Dynamic alpha calculation (case-aware)
    4. Final score fusion
    
    References:
        DAT Paper Section 4: "To overcome the limitations of static weighting in 
        hybrid retrieval, we introduce DAT, a query-adaptive framework that 
        dynamically adjusts the retrieval weighting coefficient based on the 
        effectiveness of each method for a given query."
    """

    def __init__(
        self,
        dense_store: Any,  # ChromaDB or other vector store client
        sparse_store: Any,  # Elasticsearch or other BM25 client
        alpha_tuner: AlphaTuner,
        config: DATConfig,
    ):
        """Initialize the DAT strategy.

        Args:
            dense_store: Vector store client for dense retrieval (e.g., ChromaDB)
            sparse_store: Sparse search client for BM25 retrieval (e.g., Elasticsearch)
            alpha_tuner: Alpha tuner for dynamic weighting
            config: DAT configuration
        """
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.alpha_tuner = alpha_tuner
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def retrieve(
        self, query: Query, collection_name: str, top_k: int = 10
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents using dynamic alpha tuning.
        
        Implements the complete DAT algorithm:
        - Step 1: Perform hybrid retrieval (dense + sparse)
        - Step 2: LLM evaluates top-1 from each method (via alpha_tuner)
        - Step 3: Calculate dynamic alpha (via alpha_tuner)
        - Step 4: Fuse scores with alpha weighting (via alpha_tuner)
        
        References:
            DAT Paper Section 4: Complete DAT workflow

        Args:
            query: The query to retrieve documents for
            collection_name: Name of the collection to search
            top_k: Number of top results to return

        Returns:
            List of retrieval results ordered by relevance
        """
        try:
            self.logger.info(f"DAT retrieval for query: '{query.text}' in collection: {collection_name}")

            # DAT Step 1: Perform hybrid retrieval
            # Paper: "we retrieve the top-1 result from both sparse and dense retrieval methods"
            dense_results = await self._dense_retrieve(query, collection_name, self.config.top_k_dense)
            sparse_results = await self._sparse_retrieve(query, collection_name, self.config.top_k_sparse)

            # DAT Steps 2-3: Calculate optimal alpha if dynamic tuning is enabled
            # (LLM scoring + alpha calculation happen in alpha_tuner)
            if self.config.use_dynamic_tuning:
                alpha = await self.alpha_tuner.calculate_alpha(query, dense_results, sparse_results)
            else:
                alpha = self.config.dense_weight_default
                self.logger.info(f"Using default alpha: {alpha}")

            # DAT Step 4: Combine results with alpha weighting (score fusion)
            combined_results = self.alpha_tuner.apply_alpha(alpha, dense_results, sparse_results)

            # Return top-k results
            final_results = combined_results[:top_k]
            self.logger.info(f"Returning {len(final_results)} results")

            return final_results

        except Exception as e:
            self.logger.error(f"DAT retrieval failed: {e}")
            raise RetrievalError(f"DAT retrieval failed: {e}") from e

    async def _dense_retrieve(
        self, query: Query, collection_name: str, top_k: int
    ) -> List[RetrievalResult]:
        """Perform dense (semantic) retrieval using vector store.

        Args:
            query: The query
            collection_name: Collection name
            top_k: Number of results

        Returns:
            List of retrieval results from dense retrieval
        """
        try:
            # Use vector store's retrieve method
            documents = await self.dense_store.retrieve(
                query=query.text, collection_name=collection_name, limit=top_k
            )

            # Convert to RetrievalResult
            results = [
                RetrievalResult(
                    document=doc,
                    score=doc.score if doc.score is not None else 1.0,
                    retrieval_method="dense",
                )
                for doc in documents
            ]

            self.logger.debug(f"Dense retrieval returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Dense retrieval failed: {e}")
            return []

    async def _sparse_retrieve(
        self, query: Query, collection_name: str, top_k: int
    ) -> List[RetrievalResult]:
        """Perform sparse (BM25) retrieval using Elasticsearch.

        Args:
            query: The query
            collection_name: Collection/index name
            top_k: Number of results

        Returns:
            List of retrieval results from sparse retrieval
        """
        try:
            # Query Elasticsearch for BM25 results
            response = await self.sparse_store.search(
                index=collection_name, query=query.text, size=top_k
            )

            # Convert Elasticsearch hits to RetrievalResult
            results = []
            for hit in response.get("hits", {}).get("hits", []):
                doc = self._es_hit_to_document(hit)
                results.append(
                    RetrievalResult(
                        document=doc, score=hit.get("_score", 0.0), retrieval_method="sparse"
                    )
                )

            self.logger.debug(f"Sparse retrieval returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Sparse retrieval failed: {e}")
            return []

    def _es_hit_to_document(self, hit: dict) -> Any:
        """Convert Elasticsearch hit to ragprod Document.

        Args:
            hit: Elasticsearch hit

        Returns:
            Document instance
        """
        from ragprod.domain.document import Document

        source = hit.get("_source", {})
        return Document(
            id=hit.get("_id"),
            raw_text=source.get("text", ""),
            source=source.get("source", "Unknown"),
            title=source.get("title", "Untitled"),
            metadata=source.get("metadata", {}),
            score=hit.get("_score", 0.0),
        )
