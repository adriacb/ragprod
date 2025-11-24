"""Dynamic alpha tuning for balancing dense and sparse retrieval."""

import logging

from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.strategies.dat.effectiveness_scorer import EffectivenessScorer


class AlphaTuner:
    """Dynamically calibrates the alpha weighting factor for hybrid retrieval."""

    def __init__(self, scorer: EffectivenessScorer, default_alpha: float = 0.5):
        """Initialize the alpha tuner.

        Args:
            scorer: Effectiveness scorer for evaluating retrieval methods
            default_alpha: Default alpha value (weight for dense retrieval)
        """
        self.scorer = scorer
        self.default_alpha = default_alpha
        self.logger = logging.getLogger(__name__)

    async def calculate_alpha(
        self, query: Query, dense_results: list[RetrievalResult], sparse_results: list[RetrievalResult]
    ) -> float:
        """Calculate optimal alpha (dense weight) for the given query.

        Args:
            query: The original query
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval

        Returns:
            Alpha value (0.0 to 1.0) representing weight for dense retrieval
            Sparse weight is (1 - alpha)
        """
        # If either method has no results, use the other exclusively
        if not dense_results and not sparse_results:
            return self.default_alpha
        if not dense_results:
            self.logger.info("No dense results, using only sparse retrieval")
            return 0.0  # Use only sparse
        if not sparse_results:
            self.logger.info("No sparse results, using only dense retrieval")
            return 1.0  # Use only dense

        # Get effectiveness scores from LLM
        scores = await self.scorer.score_results(query, dense_results, sparse_results)

        dense_score = scores["dense_score"]
        sparse_score = scores["sparse_score"]

        # Calculate alpha based on relative effectiveness
        total_score = dense_score + sparse_score

        if total_score == 0:
            self.logger.warning("Both scores are zero, using default alpha")
            return self.default_alpha

        # Alpha is the proportion of dense effectiveness
        alpha = dense_score / total_score

        self.logger.info(
            f"Calculated alpha={alpha:.3f} (dense_score={dense_score:.3f}, sparse_score={sparse_score:.3f})"
        )

        return alpha

    def apply_alpha(
        self, alpha: float, dense_results: list[RetrievalResult], sparse_results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Apply alpha weighting to combine dense and sparse results.

        Args:
            alpha: Weight for dense retrieval (0.0 to 1.0)
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval

        Returns:
            Combined and weighted results
        """
        # Create a dictionary to merge results by document ID
        combined: dict[str, RetrievalResult] = {}

        # Add dense results with alpha weighting
        for result in dense_results:
            doc_id = result.document.id
            weighted_score = result.score * alpha
            combined[doc_id] = RetrievalResult(
                document=result.document,
                score=weighted_score,
                retrieval_method="dense",
                metadata={"original_score": result.score, "alpha": alpha},
            )

        # Add or merge sparse results with (1-alpha) weighting
        sparse_weight = 1.0 - alpha
        for result in sparse_results:
            doc_id = result.document.id
            weighted_score = result.score * sparse_weight

            if doc_id in combined:
                # Chunk appears in both - combine scores
                existing = combined[doc_id]
                combined_score = existing.score + weighted_score
                combined[doc_id] = RetrievalResult(
                    document=result.document,
                    score=combined_score,
                    retrieval_method="hybrid",
                    metadata={
                        "dense_score": existing.score,
                        "sparse_score": weighted_score,
                        "alpha": alpha,
                    },
                )
            else:
                # Only in sparse results
                combined[doc_id] = RetrievalResult(
                    document=result.document,
                    score=weighted_score,
                    retrieval_method="sparse",
                    metadata={"original_score": result.score, "alpha": alpha},
                )

        # Sort by combined score
        sorted_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)

        self.logger.info(f"Combined {len(dense_results)} dense + {len(sparse_results)} sparse → {len(sorted_results)} total results")

        return sorted_results
