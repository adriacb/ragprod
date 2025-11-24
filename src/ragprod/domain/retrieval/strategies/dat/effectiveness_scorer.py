"""LLM-based effectiveness scorer for DAT."""

import json
import logging
from typing import Any

from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.exceptions import RetrievalError


class EffectivenessScorer:
    """Evaluates retrieval effectiveness using an LLM."""

    def __init__(self, llm_client: Any, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """Initialize the effectiveness scorer.

        Args:
            llm_client: LLM client (e.g., OpenAI client)
            model: Model name to use
            temperature: Temperature for generation
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    async def score_results(
        self, query: Query, dense_results: list[RetrievalResult], sparse_results: list[RetrievalResult]
    ) -> dict[str, float]:
        """Score the effectiveness of dense and sparse retrieval results.

        Args:
            query: The original query
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval

        Returns:
            Dictionary with 'dense_score' and 'sparse_score' (0.0 to 1.0)
        """
        try:
            # Get top result from each method for evaluation
            top_dense = dense_results[0] if dense_results else None
            top_sparse = sparse_results[0] if sparse_results else None

            prompt = self._build_evaluation_prompt(query, top_dense, top_sparse)

            # Call LLM to evaluate effectiveness
            response = await self._call_llm(prompt)
            scores = self._parse_scores(response)

            return scores

        except Exception as e:
            self.logger.error(f"Failed to score retrieval effectiveness: {e}")
            raise RetrievalError(f"Failed to score retrieval effectiveness: {e}") from e

    def _build_evaluation_prompt(
        self, query: Query, dense_result: RetrievalResult | None, sparse_result: RetrievalResult | None
    ) -> str:
        """Build prompt for LLM evaluation."""
        dense_text = dense_result.document.raw_text if dense_result else "No result"
        sparse_text = sparse_result.document.raw_text if sparse_result else "No result"

        return f"""You are evaluating the effectiveness of two retrieval methods for a query.

Query: "{query.text}"

Dense Retrieval (semantic) Top Result:
{dense_text}

Sparse Retrieval (keyword-based) Top Result:
{sparse_text}

Evaluate how relevant and useful each result is for answering the query.
Rate each method from 0.0 (completely irrelevant) to 1.0 (highly relevant and useful).

Respond ONLY with a JSON object in this exact format:
{{"dense_score": <float>, "sparse_score": <float>}}"""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the evaluation prompt."""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise RetrievalError(f"LLM call failed: {e}") from e

    def _parse_scores(self, response: str) -> dict[str, float]:
        """Parse scores from LLM response."""
        try:
            scores = json.loads(response)
            dense_score = float(scores.get("dense_score", 0.5))
            sparse_score = float(scores.get("sparse_score", 0.5))

            # Clamp scores to valid range
            dense_score = max(0.0, min(1.0, dense_score))
            sparse_score = max(0.0, min(1.0, sparse_score))

            return {"dense_score": dense_score, "sparse_score": sparse_score}
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise RetrievalError(f"Failed to parse LLM response: {e}") from e
