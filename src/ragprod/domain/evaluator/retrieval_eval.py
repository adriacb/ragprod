from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from ragprod.domain.document.base import BaseDocument


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval evaluation."""
    
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    map: float
    ndcg_at_k: float
    hit_rate: float
    k: int
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int


@dataclass
class BatchRetrievalMetrics:
    """Aggregated metrics for batch evaluation."""
    
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_f1_at_k: float
    mean_mrr: float
    mean_map: float
    mean_ndcg_at_k: float
    mean_hit_rate: float
    k: int
    num_queries: int
    per_query_metrics: List[RetrievalMetrics]


class BaseRetrievalEvaluator(ABC):
    """Base interface for retrieval evaluation."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass

