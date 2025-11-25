"""LLM-based effectiveness scorer for DAT.

References:
    Hsu, H.-L., & Tzeng, J. (2025). "DAT: Dynamic Alpha Tuning for Hybrid Retrieval 
    in Retrieval-Augmented Generation". arXiv:2503.23013
    https://arxiv.org/abs/2503.23013
    
    Paper Section 4.1: LLM-Based Retrieval Effectiveness Scoring
    Paper Appendix A: Prompt Template
"""

import logging
from typing import Any

from ragprod.domain.retrieval.entities import Query, RetrievalResult
from ragprod.domain.retrieval.exceptions import RetrievalError


class EffectivenessScorer:
    """Evaluates retrieval effectiveness using an LLM.
    
    Implements DAT Algorithm Step 2: LLM-based effectiveness evaluation.
    
    References:
        DAT Paper Section 4.1: "A key component of DAT is the use of LLMs as evaluators 
        of retrieval quality. We posit that LLMs, with their deep semantic understanding, 
        can assess the relevance of a retrieved document to the original query."
        
    Scoring Rubric (from paper Section 4.1):
        - 5 points: Direct hit - document directly answers the question
        - 3-4 points: Good wrong result - conceptually close, correct answer likely nearby
        - 1-2 points: Bad wrong result - loosely related but misleading
        - 0 points: Completely off-track - totally unrelated
    """

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
    ) -> dict[str, int]:
        """Score the effectiveness of dense and sparse retrieval results.
        
        DAT Algorithm Step 2: Effectiveness Evaluation
        - Evaluates top-1 result from each retrieval method
        - Returns effectiveness scores (0-5 discrete range) for normalization
        
        References:
            DAT Paper Section 4.1: "The LLM independently evaluates each of the top-1 
            documents and assigns scores: S_v(q) = S(q, d_v,1) for dense retrieval 
            and S_b(q) = S(q, d_b,1) for BM25."

        Args:
            query: The original query
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval

        Returns:
            Dictionary with 'dense_score' and 'sparse_score' (integers 0-5)
        """
        try:
            # DAT Step 2a: Extract top-1 results from each method
            # Paper: "we retrieve the top-1 result from both sparse and dense retrieval methods"
            top_dense = dense_results[0] if dense_results else None
            top_sparse = sparse_results[0] if sparse_results else None

            prompt = self._build_evaluation_prompt(query, top_dense, top_sparse)

            # DAT Step 2b: LLM evaluation
            response = await self._call_llm(prompt)
            scores = self._parse_scores(response)

            return scores

        except Exception as e:
            self.logger.error(f"Failed to score retrieval effectiveness: {e}")
            raise RetrievalError(f"Failed to score retrieval effectiveness: {e}") from e

    def _build_evaluation_prompt(
        self, query: Query, dense_result: RetrievalResult | None, sparse_result: RetrievalResult | None
    ) -> str:
        """Build prompt for LLM evaluation using DAT paper's exact template.
        
        References:
            DAT Paper Appendix A: Prompt Template
        """
        dense_text = dense_result.document.raw_text if dense_result else "No result"
        sparse_text = sparse_result.document.raw_text if sparse_result else "No result"

        # Exact prompt template from DAT paper Appendix A
        return f"""You are an evaluator assessing the retrieval effectiveness of dense retrieval (Cosine Distance) and BM25 retrieval for finding the correct answer.

## Task:
Given a question and two top1 search results (one from dense retrieval, one from BM25 retrieval), score each retrieval method from **0 to 5** based on whether the correct answer is likely to appear in top2, top3, etc.

### **Scoring Criteria:**
1. **Direct hit --> 5 points**
   - If the retrieved document directly answers the question, assign **5 points**.

2. **Good wrong result (High likelihood correct answer is nearby) --> 3-4 points**
   - If the top1 result is **conceptually close** to the correct answer (e.g., mentions relevant entities, related events, partial answer), it indicates the search method is in the right direction.
   - Give **4** if it's very close, **3** if somewhat close.

3. **Bad wrong result (Low likelihood correct answer is nearby) --> 1-2 points**
   - If the top1 result is **loosely related but misleading** (e.g., shares keywords but changes context), correct answers might not be in top2, top3.
   - Give **2** if there's a small chance correct answers are nearby, **1** if unlikely.

4. **Completely off-track --> 0 points**
   - If the result is **totally unrelated**, it means the retrieval method is failing.

---

### **Given Data:**
- **Question:** "{query.text}"
- **dense retrieval Top1 Result:** "{dense_text}"
- **BM25 retrieval Top1 Result:** "{sparse_text}"

---

### **Output Format:**
Return two integers separated by a space:
- **First number:** dense retrieval score.
- **Second number:** BM25 retrieval score.
- Example output: 3 4
  (Vector: 3, BM25: 4)

**Do not output any other text.**"""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the evaluation prompt."""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise RetrievalError(f"LLM call failed: {e}") from e

    def _parse_scores(self, response: str) -> dict[str, int]:
        """Parse scores from LLM response.
        
        Expected format from paper: "3 4" (space-separated integers)
        
        References:
            DAT Paper Appendix A: Output format is space-separated integers
        """
        try:
            # Parse space-separated integers
            parts = response.strip().split()
            if len(parts) >= 2:
                dense_score = int(parts[0])
                sparse_score = int(parts[1])
            else:
                raise ValueError(f"Expected 2 space-separated integers, got: {response}")

            # Clamp scores to valid range [0, 5] per paper
            dense_score = max(0, min(5, dense_score))
            sparse_score = max(0, min(5, sparse_score))

            self.logger.debug(f"Parsed scores: dense={dense_score}, sparse={sparse_score}")
            return {"dense_score": dense_score, "sparse_score": sparse_score}
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse LLM response '{response}': {e}")
            # Fallback to neutral scores (middle of range)
            return {"dense_score": 2, "sparse_score": 2}
