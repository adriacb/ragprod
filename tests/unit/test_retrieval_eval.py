import pytest
from ragprod.infrastructure.evaluator import RetrievalEvaluator
from ragprod.domain.document import Document
from ragprod.domain.evaluator.retrieval_eval import RetrievalMetrics, BatchRetrievalMetrics


class TestRetrievalEvaluator:
    """Test cases for RetrievalEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = RetrievalEvaluator()

    def test_precision_at_k_perfect_retrieval(self):
        """Test Precision@K when all retrieved documents are relevant."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),
            Document(id="doc2", raw_text="Content 2"),
            Document(id="doc3", raw_text="Content 3"),
        ]
        relevant_ids = ["doc1", "doc2", "doc3"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        assert metrics.precision_at_k == 1.0
        assert metrics.num_relevant_retrieved == 3

    def test_precision_at_k_partial_relevance(self):
        """Test Precision@K when only some retrieved documents are relevant."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id="doc2", raw_text="Content 2"),  # Not relevant
            Document(id="doc3", raw_text="Content 3"),  # Relevant
        ]
        relevant_ids = ["doc1", "doc3", "doc4"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        assert metrics.precision_at_k == pytest.approx(2.0 / 3.0, abs=0.001)
        assert metrics.num_relevant_retrieved == 2

    def test_recall_at_k_perfect_recall(self):
        """Test Recall@K when all relevant documents are retrieved."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),
            Document(id="doc2", raw_text="Content 2"),
        ]
        relevant_ids = ["doc1", "doc2"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        assert metrics.recall_at_k == 1.0

    def test_recall_at_k_partial_recall(self):
        """Test Recall@K when only some relevant documents are retrieved."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id="doc2", raw_text="Content 2"),  # Not relevant
        ]
        relevant_ids = ["doc1", "doc3", "doc4", "doc5"]  # 4 relevant, 1 retrieved

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        assert metrics.recall_at_k == pytest.approx(1.0 / 4.0, abs=0.001)

    def test_f1_score_balanced(self):
        """Test F1-Score calculation."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id="doc2", raw_text="Content 2"),  # Relevant
            Document(id="doc3", raw_text="Content 3"),  # Not relevant
        ]
        relevant_ids = ["doc1", "doc2", "doc4"]  # 3 relevant total

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        # Precision = 2/3, Recall = 2/3
        # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2 * (4/9) / (4/3) = 8/9 / 4/3 = 2/3
        expected_f1 = 2.0 / 3.0
        assert metrics.f1_at_k == pytest.approx(expected_f1, abs=0.001)

    def test_mrr_first_relevant_at_position_1(self):
        """Test MRR when first relevant document is at position 1."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id="doc2", raw_text="Content 2"),
            Document(id="doc3", raw_text="Content 3"),
        ]
        relevant_ids = ["doc1", "doc3"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        assert metrics.mrr == 1.0

    def test_mrr_first_relevant_at_position_2(self):
        """Test MRR when first relevant document is at position 2."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Not relevant
            Document(id="doc2", raw_text="Content 2"),  # Relevant
            Document(id="doc3", raw_text="Content 3"),
        ]
        relevant_ids = ["doc2", "doc3"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        assert metrics.mrr == pytest.approx(1.0 / 2.0, abs=0.001)

    def test_mrr_no_relevant_documents(self):
        """Test MRR when no relevant documents are retrieved."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),
            Document(id="doc2", raw_text="Content 2"),
        ]
        relevant_ids = ["doc3", "doc4"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        assert metrics.mrr == 0.0

    def test_map_calculation(self):
        """Test Mean Average Precision calculation."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant (pos 1)
            Document(id="doc2", raw_text="Content 2"),  # Not relevant
            Document(id="doc3", raw_text="Content 3"),  # Relevant (pos 3)
            Document(id="doc4", raw_text="Content 4"),  # Not relevant
            Document(id="doc5", raw_text="Content 5"),  # Relevant (pos 5)
        ]
        relevant_ids = ["doc1", "doc3", "doc5"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        # MAP = (1/1 + 2/3 + 3/5) / 3 = (1 + 0.667 + 0.6) / 3 ≈ 0.756
        assert metrics.map > 0.7
        assert metrics.map < 0.8

    def test_ndcg_at_k_binary_relevance(self):
        """Test NDCG@K with binary relevance."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id="doc2", raw_text="Content 2"),  # Not relevant
            Document(id="doc3", raw_text="Content 3"),  # Relevant
        ]
        relevant_ids = ["doc1", "doc3", "doc4"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        # NDCG should be between 0 and 1
        assert 0.0 <= metrics.ndcg_at_k <= 1.0
        assert metrics.ndcg_at_k > 0.0  # Should be positive since we have relevant docs

    def test_ndcg_at_k_graded_relevance(self):
        """Test NDCG@K with graded relevance scores."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),
            Document(id="doc2", raw_text="Content 2"),
            Document(id="doc3", raw_text="Content 3"),
        ]
        relevant_ids = ["doc1", "doc3", "doc4"]
        relevance_scores = {
            "doc1": 0.9,  # Highly relevant
            "doc2": 0.1,  # Not very relevant
            "doc3": 0.8,  # Very relevant
            "doc4": 0.7,  # Relevant but not retrieved
        }

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
            relevance_scores=relevance_scores,
        )

        assert 0.0 <= metrics.ndcg_at_k <= 1.0

    def test_hit_rate_with_relevant_documents(self):
        """Test Hit Rate when relevant documents are retrieved."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id="doc2", raw_text="Content 2"),
        ]
        relevant_ids = ["doc1", "doc3"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        assert metrics.hit_rate == 1.0

    def test_hit_rate_no_relevant_documents(self):
        """Test Hit Rate when no relevant documents are retrieved."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),
            Document(id="doc2", raw_text="Content 2"),
        ]
        relevant_ids = ["doc3", "doc4"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        assert metrics.hit_rate == 0.0

    def test_empty_retrieval(self):
        """Test evaluation with empty retrieval results."""
        retrieved_docs = []
        relevant_ids = ["doc1", "doc2"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        assert metrics.precision_at_k == 0.0
        assert metrics.recall_at_k == 0.0
        assert metrics.f1_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.map == 0.0
        assert metrics.ndcg_at_k == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.num_retrieved == 0

    def test_no_relevant_documents(self):
        """Test evaluation when there are no relevant documents."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),
            Document(id="doc2", raw_text="Content 2"),
        ]
        relevant_ids = []

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        assert metrics.recall_at_k == 0.0
        assert metrics.num_relevant == 0
        assert metrics.num_relevant_retrieved == 0

    def test_k_smaller_than_retrieved(self):
        """Test evaluation when k is smaller than number of retrieved documents."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id="doc2", raw_text="Content 2"),  # Not relevant
            Document(id="doc3", raw_text="Content 3"),  # Relevant
            Document(id="doc4", raw_text="Content 4"),  # Not relevant
            Document(id="doc5", raw_text="Content 5"),  # Relevant
        ]
        relevant_ids = ["doc1", "doc3", "doc5"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,  # Only consider top 3
        )

        # Should only consider first 3 documents
        assert metrics.k == 3
        assert metrics.precision_at_k == pytest.approx(2.0 / 3.0, abs=0.001)

    def test_documents_without_ids(self):
        """Test evaluation when some documents don't have IDs."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),  # Relevant
            Document(id=None, raw_text="Content 2"),  # No ID, won't match
            Document(id="doc3", raw_text="Content 3"),  # Relevant
        ]
        relevant_ids = ["doc1", "doc3"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=3,
        )

        # Should still work, documents without IDs are ignored
        assert metrics.num_relevant_retrieved == 2

    def test_evaluate_batch_single_query(self):
        """Test batch evaluation with a single query."""
        queries = ["query1"]
        retrieved_documents_list = [
            [
                Document(id="doc1", raw_text="Content 1"),
                Document(id="doc2", raw_text="Content 2"),
            ]
        ]
        relevant_document_ids_list = [["doc1", "doc3"]]

        batch_metrics = self.evaluator.evaluate_batch(
            queries=queries,
            retrieved_documents_list=retrieved_documents_list,
            relevant_document_ids_list=relevant_document_ids_list,
            k=5,
        )

        assert batch_metrics.num_queries == 1
        assert len(batch_metrics.per_query_metrics) == 1
        assert batch_metrics.mean_precision_at_k > 0.0

    def test_evaluate_batch_multiple_queries(self):
        """Test batch evaluation with multiple queries."""
        queries = ["query1", "query2", "query3"]
        retrieved_documents_list = [
            [Document(id="doc1", raw_text="Content 1")],  # Query 1
            [Document(id="doc2", raw_text="Content 2")],  # Query 2
            [Document(id="doc3", raw_text="Content 3")],  # Query 3
        ]
        relevant_document_ids_list = [
            ["doc1", "doc4"],  # Query 1: 1 relevant retrieved
            ["doc2", "doc5"],  # Query 2: 1 relevant retrieved
            ["doc6", "doc7"],  # Query 3: 0 relevant retrieved
        ]

        batch_metrics = self.evaluator.evaluate_batch(
            queries=queries,
            retrieved_documents_list=retrieved_documents_list,
            relevant_document_ids_list=relevant_document_ids_list,
            k=5,
        )

        assert batch_metrics.num_queries == 3
        assert len(batch_metrics.per_query_metrics) == 3
        # Query 1: 1 relevant retrieved / 1 retrieved = 1.0
        # Query 2: 1 relevant retrieved / 1 retrieved = 1.0
        # Query 3: 0 relevant retrieved / 1 retrieved = 0.0
        # Mean = (1.0 + 1.0 + 0.0) / 3 = 2/3
        assert batch_metrics.mean_precision_at_k == pytest.approx(2.0 / 3.0, abs=0.001)
        assert batch_metrics.mean_hit_rate == pytest.approx(2.0 / 3.0, abs=0.001)

    def test_evaluate_batch_with_graded_relevance(self):
        """Test batch evaluation with graded relevance scores."""
        queries = ["query1", "query2"]
        retrieved_documents_list = [
            [Document(id="doc1", raw_text="Content 1")],
            [Document(id="doc2", raw_text="Content 2")],
        ]
        relevant_document_ids_list = [["doc1"], ["doc2"]]
        relevance_scores_list = [
            {"doc1": 0.9},
            {"doc2": 0.8},
        ]

        batch_metrics = self.evaluator.evaluate_batch(
            queries=queries,
            retrieved_documents_list=retrieved_documents_list,
            relevant_document_ids_list=relevant_document_ids_list,
            k=5,
            relevance_scores_list=relevance_scores_list,
        )

        assert batch_metrics.num_queries == 2
        assert all(m.ndcg_at_k > 0.0 for m in batch_metrics.per_query_metrics)

    def test_evaluate_batch_mismatched_lengths(self):
        """Test batch evaluation raises error for mismatched list lengths."""
        queries = ["query1", "query2"]
        retrieved_documents_list = [[Document(id="doc1", raw_text="Content 1")]]
        relevant_document_ids_list = [["doc1"], ["doc2"]]

        with pytest.raises(ValueError) as exc_info:
            self.evaluator.evaluate_batch(
                queries=queries,
                retrieved_documents_list=retrieved_documents_list,
                relevant_document_ids_list=relevant_document_ids_list,
                k=5,
            )

        assert "must have the same length" in str(exc_info.value)

    def test_metrics_dataclass_fields(self):
        """Test that RetrievalMetrics has all expected fields."""
        retrieved_docs = [
            Document(id="doc1", raw_text="Content 1"),
        ]
        relevant_ids = ["doc1"]

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        # Check all fields are present and have correct types
        assert isinstance(metrics.precision_at_k, float)
        assert isinstance(metrics.recall_at_k, float)
        assert isinstance(metrics.f1_at_k, float)
        assert isinstance(metrics.mrr, float)
        assert isinstance(metrics.map, float)
        assert isinstance(metrics.ndcg_at_k, float)
        assert isinstance(metrics.hit_rate, float)
        assert isinstance(metrics.k, int)
        assert isinstance(metrics.num_retrieved, int)
        assert isinstance(metrics.num_relevant, int)
        assert isinstance(metrics.num_relevant_retrieved, int)

    def test_batch_metrics_dataclass_fields(self):
        """Test that BatchRetrievalMetrics has all expected fields."""
        queries = ["query1"]
        retrieved_documents_list = [[Document(id="doc1", raw_text="Content 1")]]
        relevant_document_ids_list = [["doc1"]]

        batch_metrics = self.evaluator.evaluate_batch(
            queries=queries,
            retrieved_documents_list=retrieved_documents_list,
            relevant_document_ids_list=relevant_document_ids_list,
            k=5,
        )

        # Check all fields are present
        assert isinstance(batch_metrics.mean_precision_at_k, float)
        assert isinstance(batch_metrics.mean_recall_at_k, float)
        assert isinstance(batch_metrics.mean_f1_at_k, float)
        assert isinstance(batch_metrics.mean_mrr, float)
        assert isinstance(batch_metrics.mean_map, float)
        assert isinstance(batch_metrics.mean_ndcg_at_k, float)
        assert isinstance(batch_metrics.mean_hit_rate, float)
        assert isinstance(batch_metrics.k, int)
        assert isinstance(batch_metrics.num_queries, int)
        assert isinstance(batch_metrics.per_query_metrics, list)

    def test_realistic_scenario(self):
        """Test a realistic retrieval scenario."""
        # Simulate a realistic retrieval scenario
        retrieved_docs = [
            Document(id="doc1", raw_text="Python programming tutorial"),  # Relevant
            Document(id="doc2", raw_text="Cooking recipes"),  # Not relevant
            Document(id="doc3", raw_text="Python web frameworks"),  # Relevant
            Document(id="doc4", raw_text="Machine learning basics"),  # Somewhat relevant
            Document(id="doc5", raw_text="Python data structures"),  # Relevant
        ]
        relevant_ids = ["doc1", "doc3", "doc5", "doc6"]  # 4 relevant, 3 retrieved

        metrics = self.evaluator.evaluate_single(
            retrieved_documents=retrieved_docs,
            relevant_document_ids=relevant_ids,
            k=5,
        )

        # Check reasonable values
        assert 0.0 <= metrics.precision_at_k <= 1.0
        assert 0.0 <= metrics.recall_at_k <= 1.0
        assert 0.0 <= metrics.f1_at_k <= 1.0
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.map <= 1.0
        assert 0.0 <= metrics.ndcg_at_k <= 1.0
        assert metrics.hit_rate in [0.0, 1.0]

        # Specific checks for this scenario
        assert metrics.precision_at_k == pytest.approx(3.0 / 5.0, abs=0.001)  # 3 relevant out of 5
        assert metrics.recall_at_k == pytest.approx(3.0 / 4.0, abs=0.001)  # 3 retrieved out of 4 relevant
        assert metrics.mrr == 1.0  # First doc is relevant
        assert metrics.hit_rate == 1.0  # At least one relevant doc retrieved

