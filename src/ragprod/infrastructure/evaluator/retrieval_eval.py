import math
from typing import List, Dict, Optional
from ragprod.domain.document.base import BaseDocument
from ragprod.domain.evaluator.retrieval_eval import (
    BaseRetrievalEvaluator,
    RetrievalMetrics,
    BatchRetrievalMetrics,
)


class RetrievalEvaluator(BaseRetrievalEvaluator):
    """Implementation of retrieval evaluation metrics."""
    
    def evaluate_single(
        self,
        retrieved_documents: List[BaseDocument],
        relevant_document_ids: List[str],
        k: int = 5,
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance for a single query.
        
        Args:
            retrieved_documents: List of documents returned by the retriever
            relevant_document_ids: List of document IDs that are actually relevant
            k: Number of top documents to consider for @K metrics
            relevance_scores: Optional dict mapping document IDs to relevance scores (0-1 or 0-5)
        
        Returns:
            RetrievalMetrics object with all computed metrics
        """
        if not retrieved_documents:
            return self._empty_metrics(k, len(relevant_document_ids))
        
        if not relevant_document_ids:
            return self._empty_metrics(k, 0, len(retrieved_documents))
        
        # Get top k documents
        top_k_docs = retrieved_documents[:k]
        
        # Extract document IDs from retrieved documents
        retrieved_ids = [doc.id for doc in top_k_docs if doc.id is not None]
        
        # Convert relevant_document_ids to set for faster lookup
        relevant_set = set(relevant_document_ids)
        
        # Calculate binary relevance for retrieved documents
        retrieved_relevance = [
            1 if doc_id in relevant_set else 0
            for doc_id in retrieved_ids
        ]
        
        # Calculate graded relevance if provided
        if relevance_scores:
            retrieved_graded_relevance = [
                relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids
            ]
        else:
            retrieved_graded_relevance = retrieved_relevance
        
        # Calculate metrics
        precision_at_k = self._precision_at_k(retrieved_relevance, k)
        recall_at_k = self._recall_at_k(retrieved_relevance, len(relevant_set))
        f1_at_k = self._f1_score(precision_at_k, recall_at_k)
        mrr = self._mean_reciprocal_rank(retrieved_relevance)
        map_score = self._mean_average_precision(retrieved_relevance)
        ndcg_at_k = self._ndcg_at_k(retrieved_graded_relevance, k, relevance_scores, relevant_set)
        hit_rate = self._hit_rate(retrieved_relevance)
        
        num_relevant_retrieved = sum(retrieved_relevance)
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            mrr=mrr,
            map=map_score,
            ndcg_at_k=ndcg_at_k,
            hit_rate=hit_rate,
            k=k,
            num_retrieved=len(retrieved_documents),
            num_relevant=len(relevant_set),
            num_relevant_retrieved=num_relevant_retrieved,
        )
    
    def evaluate_batch(
        self,
        queries: List[str],
        retrieved_documents_list: List[List[BaseDocument]],
        relevant_document_ids_list: List[List[str]],
        k: int = 5,
        relevance_scores_list: Optional[List[Dict[str, float]]] = None,
    ) -> BatchRetrievalMetrics:
        """
        Evaluate retrieval performance for multiple queries.
        
        Args:
            queries: List of query strings
            retrieved_documents_list: List of lists, each containing retrieved documents for a query
            relevant_document_ids_list: List of lists, each containing relevant document IDs for a query
            k: Number of top documents to consider for @K metrics
            relevance_scores_list: Optional list of dicts, each mapping document IDs to relevance scores
        
        Returns:
            BatchRetrievalMetrics with aggregated metrics and per-query results
        """
        if len(queries) != len(retrieved_documents_list) or len(queries) != len(relevant_document_ids_list):
            raise ValueError(
                "queries, retrieved_documents_list, and relevant_document_ids_list must have the same length"
            )
        
        per_query_metrics = []
        
        for i, query in enumerate(queries):
            relevance_scores = (
                relevance_scores_list[i] if relevance_scores_list else None
            )
            metrics = self.evaluate_single(
                retrieved_documents=retrieved_documents_list[i],
                relevant_document_ids=relevant_document_ids_list[i],
                k=k,
                relevance_scores=relevance_scores,
            )
            per_query_metrics.append(metrics)
        
        # Calculate mean metrics
        def mean(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0
        
        mean_precision = mean([m.precision_at_k for m in per_query_metrics])
        mean_recall = mean([m.recall_at_k for m in per_query_metrics])
        mean_f1 = mean([m.f1_at_k for m in per_query_metrics])
        mean_mrr = mean([m.mrr for m in per_query_metrics])
        mean_map = mean([m.map for m in per_query_metrics])
        mean_ndcg = mean([m.ndcg_at_k for m in per_query_metrics])
        mean_hit_rate = mean([m.hit_rate for m in per_query_metrics])
        
        return BatchRetrievalMetrics(
            mean_precision_at_k=mean_precision,
            mean_recall_at_k=mean_recall,
            mean_f1_at_k=mean_f1,
            mean_mrr=mean_mrr,
            mean_map=mean_map,
            mean_ndcg_at_k=mean_ndcg,
            mean_hit_rate=mean_hit_rate,
            k=k,
            num_queries=len(queries),
            per_query_metrics=per_query_metrics,
        )
    
    def _precision_at_k(self, retrieved_relevance: List[int], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        top_k_relevance = retrieved_relevance[:k]
        if not top_k_relevance:
            return 0.0
        return sum(top_k_relevance) / len(top_k_relevance)
    
    def _recall_at_k(
        self, retrieved_relevance: List[int], total_relevant: int
    ) -> float:
        """Calculate Recall@K."""
        if total_relevant == 0:
            return 0.0
        num_relevant_retrieved = sum(retrieved_relevance)
        return num_relevant_retrieved / total_relevant
    
    def _f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1-Score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _mean_reciprocal_rank(self, retrieved_relevance: List[int]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)."""
        for i, rel in enumerate(retrieved_relevance):
            if rel == 1:
                return 1.0 / (i + 1)
        return 0.0
    
    def _mean_average_precision(self, retrieved_relevance: List[int]) -> float:
        """Calculate Mean Average Precision (MAP)."""
        if not retrieved_relevance or sum(retrieved_relevance) == 0:
            return 0.0
        
        num_relevant = sum(retrieved_relevance)
        precisions = []
        
        relevant_count = 0
        for i, rel in enumerate(retrieved_relevance):
            if rel == 1:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / num_relevant
    
    def _ndcg_at_k(
        self,
        retrieved_graded_relevance: List[float],
        k: int,
        relevance_scores: Optional[Dict[str, float]],
        relevant_set: set,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        if k == 0:
            return 0.0
        
        # Get top k graded relevance scores
        top_k_graded = retrieved_graded_relevance[:k]
        
        # Calculate DCG
        dcg = sum(
            rel / math.log2(i + 2) for i, rel in enumerate(top_k_graded)
        )
        
        # Calculate ideal DCG (IDCG)
        # Sort all relevant documents by their relevance scores in descending order
        if relevance_scores:
            # Use graded relevance scores
            ideal_relevance = sorted(
                [relevance_scores.get(doc_id, 0.0) for doc_id in relevant_set],
                reverse=True,
            )[:k]
        else:
            # Binary relevance: all relevant documents have score 1
            ideal_relevance = [1.0] * min(len(relevant_set), k)
        
        idcg = sum(
            rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance)
        )
        
        # Normalize
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def _hit_rate(self, retrieved_relevance: List[int]) -> float:
        """Calculate Hit Rate (whether at least one relevant document was retrieved)."""
        return 1.0 if any(retrieved_relevance) else 0.0
    
    def _empty_metrics(
        self, k: int, num_relevant: int = 0, num_retrieved: int = 0
    ) -> RetrievalMetrics:
        """Return metrics for edge cases (empty retrieval, no relevant docs, etc.)."""
        return RetrievalMetrics(
            precision_at_k=0.0,
            recall_at_k=0.0,
            f1_at_k=0.0,
            mrr=0.0,
            map=0.0,
            ndcg_at_k=0.0,
            hit_rate=0.0,
            k=k,
            num_retrieved=num_retrieved,
            num_relevant=num_relevant,
            num_relevant_retrieved=0,
        )

