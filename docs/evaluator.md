# Retrieval Evaluation

The retrieval evaluation module provides comprehensive metrics to assess the performance of retrieval systems in RAG (Retrieval-Augmented Generation) applications.

## Overview

The `RetrievalEvaluator` class implements standard information retrieval metrics to evaluate how well a retrieval system finds relevant documents for given queries. This is crucial for optimizing RAG systems, as retrieval quality directly impacts the quality of generated responses.

## Metrics

The evaluator computes the following metrics:

### Precision@K
Measures the proportion of retrieved documents (in the top K) that are relevant to the query. Higher precision indicates fewer irrelevant documents in the results.

**Formula**: `Precision@K = (Number of relevant documents in top K) / K`

### Recall@K
Measures the proportion of all relevant documents that were successfully retrieved in the top K results. Higher recall indicates that fewer relevant documents were missed.

**Formula**: `Recall@K = (Number of relevant documents retrieved) / (Total number of relevant documents)`

### F1-Score@K
The harmonic mean of Precision@K and Recall@K, providing a balanced measure that considers both false positives and false negatives.

**Formula**: `F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)`

### Mean Reciprocal Rank (MRR)
Evaluates the position of the first relevant document in the retrieval results. A higher MRR indicates that relevant documents appear earlier in the list.

**Formula**: `MRR = 1 / (Position of first relevant document)`

### Mean Average Precision (MAP)
Calculates the average precision across all relevant documents, giving higher weight to relevant documents that appear earlier in the results.

### Normalized Discounted Cumulative Gain (NDCG@K)
Considers both the relevance and the position of documents in the retrieval results, assigning higher scores to relevant documents appearing earlier in the list. Supports both binary and graded relevance.

### Hit Rate@K
Measures the proportion of queries for which at least one relevant document is retrieved in the top K results.

**Formula**: `Hit Rate = 1 if at least one relevant document retrieved, else 0`

## Usage

### Basic Single Query Evaluation

```python
from ragprod.infrastructure.evaluator import RetrievalEvaluator
from ragprod.domain.document import Document

# Initialize the evaluator
evaluator = RetrievalEvaluator()

# Create sample retrieved documents
retrieved_docs = [
    Document(id="doc1", raw_text="Python is a programming language"),
    Document(id="doc2", raw_text="Machine learning uses algorithms"),
    Document(id="doc3", raw_text="Python has many libraries"),
    Document(id="doc4", raw_text="Cooking recipes for dinner"),
    Document(id="doc5", raw_text="Python web frameworks"),
]

# Define which documents are actually relevant for the query
relevant_document_ids = ["doc1", "doc3", "doc5"]

# Evaluate retrieval performance
metrics = evaluator.evaluate_single(
    retrieved_documents=retrieved_docs,
    relevant_document_ids=relevant_document_ids,
    k=5
)

# Access individual metrics
print(f"Precision@5: {metrics.precision_at_k:.3f}")
print(f"Recall@5: {metrics.recall_at_k:.3f}")
print(f"F1-Score@5: {metrics.f1_at_k:.3f}")
print(f"MRR: {metrics.mrr:.3f}")
print(f"MAP: {metrics.map:.3f}")
print(f"NDCG@5: {metrics.ndcg_at_k:.3f}")
print(f"Hit Rate: {metrics.hit_rate:.3f}")
```

### Evaluation with Graded Relevance

When you have relevance scores (e.g., 0-5 scale) instead of binary relevance:

```python
# Define relevance scores for documents (0.0 to 1.0 or 0 to 5)
relevance_scores = {
    "doc1": 1.0,  # Highly relevant
    "doc2": 0.3,  # Somewhat relevant
    "doc3": 0.8,  # Very relevant
    "doc4": 0.0,  # Not relevant
    "doc5": 0.9,  # Highly relevant
}

# Evaluate with graded relevance
metrics = evaluator.evaluate_single(
    retrieved_documents=retrieved_docs,
    relevant_document_ids=["doc1", "doc3", "doc5"],  # Still need binary IDs
    k=5,
    relevance_scores=relevance_scores
)

# NDCG will use the graded scores for more accurate evaluation
print(f"NDCG@5 (with graded relevance): {metrics.ndcg_at_k:.3f}")
```

### Batch Evaluation

Evaluate multiple queries at once to get aggregate metrics:

```python
# Prepare data for multiple queries
queries = [
    "What is Python?",
    "Machine learning basics",
    "Web development",
]

# Retrieved documents for each query
retrieved_documents_list = [
    [Document(id="doc1", raw_text="..."), Document(id="doc2", raw_text="...")],
    [Document(id="doc3", raw_text="..."), Document(id="doc4", raw_text="...")],
    [Document(id="doc5", raw_text="..."), Document(id="doc6", raw_text="...")],
]

# Relevant document IDs for each query
relevant_document_ids_list = [
    ["doc1", "doc2", "doc7"],  # Relevant docs for query 1
    ["doc3", "doc8"],           # Relevant docs for query 2
    ["doc5", "doc9", "doc10"],  # Relevant docs for query 3
]

# Evaluate batch
batch_metrics = evaluator.evaluate_batch(
    queries=queries,
    retrieved_documents_list=retrieved_documents_list,
    relevant_document_ids_list=relevant_document_ids_list,
    k=5
)

# Access aggregate metrics
print(f"Mean Precision@5: {batch_metrics.mean_precision_at_k:.3f}")
print(f"Mean Recall@5: {batch_metrics.mean_recall_at_k:.3f}")
print(f"Mean F1@5: {batch_metrics.mean_f1_at_k:.3f}")
print(f"Mean MRR: {batch_metrics.mean_mrr:.3f}")
print(f"Mean MAP: {batch_metrics.mean_map:.3f}")
print(f"Mean NDCG@5: {batch_metrics.mean_ndcg_at_k:.3f}")
print(f"Mean Hit Rate: {batch_metrics.mean_hit_rate:.3f}")

# Access per-query metrics
for i, query_metrics in enumerate(batch_metrics.per_query_metrics):
    print(f"\nQuery {i+1}: {queries[i]}")
    print(f"  Precision@5: {query_metrics.precision_at_k:.3f}")
    print(f"  Recall@5: {query_metrics.recall_at_k:.3f}")
```

### Integration with Retrieval System

Here's how to integrate evaluation with an actual retrieval system:

```python
from ragprod.application.use_cases import get_client_service
from ragprod.infrastructure.evaluator import RetrievalEvaluator

# Initialize your retrieval client
client = await get_client_service(
    mode="local_persistent",
    collection_name="my_documents"
)

# Initialize evaluator
evaluator = RetrievalEvaluator()

# Test queries with ground truth
test_cases = [
    {
        "query": "What is machine learning?",
        "relevant_ids": ["doc1", "doc5", "doc12"]
    },
    {
        "query": "Python programming basics",
        "relevant_ids": ["doc2", "doc8", "doc15"]
    },
]

# Evaluate each test case
results = []
for test_case in test_cases:
    # Retrieve documents
    retrieved = await client.retrieve(test_case["query"], k=5)
    
    # Evaluate
    metrics = evaluator.evaluate_single(
        retrieved_documents=retrieved,
        relevant_document_ids=test_case["relevant_ids"],
        k=5
    )
    
    results.append({
        "query": test_case["query"],
        "metrics": metrics
    })

# Print results
for result in results:
    print(f"\nQuery: {result['query']}")
    print(f"Precision@5: {result['metrics'].precision_at_k:.3f}")
    print(f"Recall@5: {result['metrics'].recall_at_k:.3f}")
```

## Understanding the Results

### Interpreting Metrics

- **Precision@K**: If Precision@5 = 0.8, it means 80% of the top 5 retrieved documents are relevant.
- **Recall@K**: If Recall@5 = 0.6, it means 60% of all relevant documents were found in the top 5 results.
- **F1-Score**: A balanced metric. Higher is better. Perfect score is 1.0.
- **MRR**: If MRR = 0.5, the first relevant document appears on average at position 2 (1/0.5 = 2).
- **MAP**: Average precision across all relevant documents. Higher is better.
- **NDCG@K**: Position-weighted relevance score. Perfect score is 1.0.
- **Hit Rate**: If Hit Rate = 1.0, at least one relevant document was retrieved for every query.

### Choosing K

The `k` parameter determines how many top results to consider:

- **Small K (e.g., 1-3)**: Focuses on top results, useful when users only look at the first few results
- **Medium K (e.g., 5-10)**: Balanced view, common in RAG systems
- **Large K (e.g., 20+)**: Evaluates broader retrieval performance

### Edge Cases

The evaluator handles edge cases gracefully:

- **Empty retrieval**: All metrics return 0.0
- **No relevant documents**: Recall and related metrics return 0.0
- **All documents relevant**: Precision and Hit Rate return 1.0
- **No retrieved documents match relevant IDs**: All metrics return 0.0

## Best Practices

1. **Use appropriate K values**: Match K to your actual use case (how many documents are typically used in RAG)

2. **Collect ground truth carefully**: Ensure your `relevant_document_ids` accurately reflect what documents should be retrieved for each query

3. **Use batch evaluation**: Evaluate on multiple queries to get reliable aggregate metrics

4. **Consider graded relevance**: If you have relevance scores, use them for more accurate NDCG evaluation

5. **Monitor over time**: Track these metrics as you improve your retrieval system (embedding models, chunking strategies, etc.)

6. **Combine with LLM metrics**: Use retrieval metrics alongside LLM performance metrics (faithfulness, relevancy) for comprehensive RAG evaluation

## Related Documentation

- [Architecture Documentation](architecture.md) - Overall system architecture
- LLM Performance Evaluation - Evaluating the generation component
- Human Annotation Evaluation - Manual evaluation methods

