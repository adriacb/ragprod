"""Dynamic alpha tuning for balancing dense and sparse retrieval.

References:
    Hsu, H.-L., & Tzeng, J. (2025). "DAT: Dynamic Alpha Tuning for Hybrid Retrieval 
    in Retrieval-Augmented Generation". arXiv:2503.23013
    https://arxiv.org/abs/2503.23013
    
    Paper Section 4.2: Dynamic Alpha Calculation
    Paper Section 4.3: Final Score Fusion
"""

import logging

from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.strategies.dat.effectiveness_scorer import EffectivenessScorer


class AlphaTuner:
    """Dynamically calibrates the alpha weighting factor for hybrid retrieval.
    
    Implements DAT Algorithm Steps 3-4:
    - Step 3: Dynamic alpha calculation using case-aware rules
    - Step 4: Score fusion with calculated alpha
    
    References:
        DAT Paper Section 4.2: "Using the LLM-assigned scores, we compute the dynamic 
        weighting coefficient α(q) through a case-aware formulation that ensures robust 
        behavior across various retrieval outcomes."
    """

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
        
        DAT Algorithm Step 3: Dynamic Alpha Calculation
        
        Implements the case-aware formulation from the paper:
        - Case 1: Equal weighting (0.5) when both methods fail (both scores = 0)
        - Case 2: Exclusive preference (1.0) when dense has perfect score (5) and sparse doesn't
        - Case 3: Exclusive preference (0.0) when sparse has perfect score (5) and dense doesn't
        - Case 4: Proportional weighting otherwise: α(q) = S_v(q) / (S_v(q) + S_b(q))
        
        References:
            DAT Paper Section 4.2: "This rule-based approach ensures:
            - Equal weighting (0.5) when both retrieval methods fail to return relevant content.
            - Exclusive preference (1.0 or 0.0) when one method yields a perfect result and the other does not.
            - Proportional weighting when both methods return partially relevant results."

        Args:
            query: The original query
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval

        Returns:
            Alpha value (0.0 to 1.0) representing weight for dense retrieval.
            Sparse weight is (1 - alpha). Rounded to one decimal place per paper.
        """
        # Handle empty results
        if not dense_results and not sparse_results:
            return self.default_alpha
        if not dense_results:
            self.logger.info("No dense results, using only sparse retrieval")
            return 0.0  # Use only sparse
        if not sparse_results:
            self.logger.info("No sparse results, using only dense retrieval")
            return 1.0  # Use only dense

        # DAT Step 2: Get effectiveness scores from LLM (0-5 discrete range)
        scores = await self.scorer.score_results(query, dense_results, sparse_results)

        dense_score = scores["dense_score"]  # S_v(q) in paper
        sparse_score = scores["sparse_score"]  # S_b(q) in paper

        # DAT Step 3: Case-aware alpha calculation (from paper Section 4.2)
        
        # Case 1: Both methods fail (both scores = 0)
        # Paper: "Equal weighting (0.5) when both retrieval methods fail"
        if dense_score == 0 and sparse_score == 0:
            alpha = 0.5
            self.logger.info("Case 1: Both methods failed (scores=0), using equal weighting α=0.5")
        
        # Case 2: Dense has perfect score (5), sparse doesn't
        # Paper: "Exclusive preference (1.0 or 0.0) when one method yields a perfect result"
        elif dense_score == 5 and sparse_score < 5:
            alpha = 1.0
            self.logger.info(f"Case 2: Dense perfect (5), sparse imperfect ({sparse_score}), using α=1.0")
        
        # Case 3: Sparse has perfect score (5), dense doesn't
        elif sparse_score == 5 and dense_score < 5:
            alpha = 0.0
            self.logger.info(f"Case 3: Sparse perfect (5), dense imperfect ({dense_score}), using α=0.0")
        
        # Case 4: Proportional weighting for all other cases
        # Paper: "Proportional weighting when both methods return partially relevant results"
        # Formula: α(q) = S_v(q) / (S_v(q) + S_b(q))
        else:
            total_score = dense_score + sparse_score
            alpha = dense_score / total_score
            self.logger.info(
                f"Case 4: Proportional weighting: α={alpha:.3f} (dense={dense_score}, sparse={sparse_score})"
            )

        # Paper Section 4.2: "the final α(q) value is rounded to one decimal place"
        alpha = round(alpha, 1)
        
        self.logger.info(f"Final α={alpha} (dense_score={dense_score}, sparse_score={sparse_score})")
        return alpha

    def apply_alpha(
        self, alpha: float, dense_results: list[RetrievalResult], sparse_results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Apply alpha weighting to combine dense and sparse results.
        
        DAT Algorithm Step 4: Final Score Fusion
        
        References:
            DAT Paper Section 4.3: "With the dynamically determined α(q), we compute 
            the final hybrid ranking score by applying the weighted combination to the 
            normalized scores from both retrieval methods."
            
            Formula: R(q,d) = α(q) × score_dense(q,d) + (1-α(q)) × score_sparse(q,d)

        Args:
            alpha: Weight for dense retrieval (0.0 to 1.0)
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval

        Returns:
            Combined and weighted results sorted by hybrid score
        """
        # Create a dictionary to merge results by document ID
        combined: dict[str, RetrievalResult] = {}

        # DAT Step 4a: Apply alpha weighting to dense results
        # Paper formula: R(q,d) = α(q) × score_dense(q,d) + (1-α(q)) × score_sparse(q,d)
        for result in dense_results:
            doc_id = result.document.id
            weighted_score = result.score * alpha
            combined[doc_id] = RetrievalResult(
                document=result.document,
                score=weighted_score,
                retrieval_method="dense",
                metadata={"original_score": result.score, "alpha": alpha},
            )

        # DAT Step 4b: Apply (1-alpha) weighting to sparse results and merge
        sparse_weight = 1.0 - alpha
        for result in sparse_results:
            doc_id = result.document.id
            weighted_score = result.score * sparse_weight

            if doc_id in combined:
                # Document appears in both - combine scores (hybrid)
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

        # DAT Step 4c: Sort by final hybrid score
        # Paper: "Documents are then ranked based on R(q,d)"
        sorted_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)

        self.logger.info(
            f"Combined {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(sorted_results)} total results"
        )

        return sorted_results
