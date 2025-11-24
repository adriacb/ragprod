# DAT (Dynamic Alpha Tuning) Retrieval Strategy

## Overview

DAT (Dynamic Alpha Tuning) is a hybrid retrieval strategy that intelligently combines dense (semantic) and sparse (keyword-based) retrieval methods. Unlike traditional hybrid approaches that use fixed weights, DAT dynamically adjusts the balance between methods based on query-specific effectiveness.

## How It Works

### 1. Dual Retrieval

DAT performs both retrieval methods in parallel:

- **Dense Retrieval** (ChromaDB): Uses vector embeddings to find semantically similar documents
- **Sparse Retrieval** (Elasticsearch): Uses BM25 algorithm for keyword matching

### 2. Effectiveness Evaluation

An LLM evaluates the top results from each method:

```python
Query: "How do I write concurrent code?"

Dense Top Result: "Asyncio enables concurrent programming in Python..."
→ LLM Score: 0.9 (highly relevant)

Sparse Top Result: "Python code examples for beginners..."
→ LLM Score: 0.3 (less relevant)
```

### 3. Dynamic Alpha Calculation

Based on effectiveness scores, DAT calculates the optimal weight (alpha):

```python
alpha = dense_score / (dense_score + sparse_score)
# Example: 0.9 / (0.9 + 0.3) = 0.75

# This means:
# - Dense results get 75% weight
# - Sparse results get 25% weight
```

### 4. Weighted Combination

Results are combined with calculated weights:

```python
final_score = (dense_score × alpha) + (sparse_score × (1 - alpha))
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              DATStrategy                        │
│                                                 │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │   ChromaDB   │         │ Elasticsearch   │  │
│  │   (Dense)    │         │   (Sparse)      │  │
│  └──────┬───────┘         └────────┬────────┘  │
│         │                          │           │
│         └──────────┬───────────────┘           │
│                    ▼                            │
│         ┌──────────────────────┐               │
│         │  EffectivenessScorer │               │
│         │    (LLM Evaluation)  │               │
│         └──────────┬───────────┘               │
│                    ▼                            │
│         ┌──────────────────────┐               │
│         │    AlphaTuner        │               │
│         │ (Dynamic Weighting)  │               │
│         └──────────┬───────────┘               │
│                    ▼                            │
│         ┌──────────────────────┐               │
│         │  Combined Results    │               │
│         └──────────────────────┘               │
└─────────────────────────────────────────────────┘
```

## Usage

### Basic Setup

```python
from openai import AsyncOpenAI
from ragprod.infrastructure.client.elasticsearch_client import ElasticsearchClient
from ragprod.infrastructure.client.chromadb import AsyncChromaDBClient
from ragprod.domain.retrieval.strategies.dat import (
    DATStrategy, DATConfig, AlphaTuner, EffectivenessScorer
)
from ragprod.domain.retrieval.entities import Query

# Initialize clients
chromadb = AsyncChromaDBClient(...)
elasticsearch = ElasticsearchClient(hosts=["http://localhost:9200"])

# Initialize DAT components
llm_client = AsyncOpenAI()
scorer = EffectivenessScorer(llm_client, model="gpt-4o-mini")
tuner = AlphaTuner(scorer, default_alpha=0.5)
config = DATConfig(use_dynamic_tuning=True)

# Create DAT strategy
strategy = DATStrategy(
    dense_store=chromadb,
    sparse_store=elasticsearch,
    alpha_tuner=tuner,
    config=config
)

# Retrieve documents
query = Query(text="What is Python?")
results = await strategy.retrieve(query, "my_collection", top_k=5)

for result in results:
    print(f"Score: {result.score:.3f} | Method: {result.retrieval_method}")
    print(f"Content: {result.document.raw_text[:100]}...")
```

### Configuration Options

```python
config = DATConfig(
    dense_weight_default=0.5,      # Default weight if dynamic tuning fails
    sparse_weight_default=0.5,
    top_k_dense=20,                # Retrieve top 20 from dense
    top_k_sparse=20,               # Retrieve top 20 from sparse
    use_dynamic_tuning=True,       # Enable LLM-based tuning
    effectiveness_threshold=0.3,   # Minimum score to consider
    llm_model="gpt-4o-mini",       # Model for effectiveness scoring
    temperature=0.0                # Temperature for LLM
)
```

### Disable Dynamic Tuning

For faster retrieval without LLM calls:

```python
config = DATConfig(
    use_dynamic_tuning=False,
    dense_weight_default=0.7,  # Fixed 70% dense, 30% sparse
)
```

## When to Use DAT

### ✅ Good Use Cases

- **Diverse query types**: Mix of semantic and keyword queries
- **Quality-critical applications**: Where retrieval accuracy is paramount
- **Hybrid search needs**: Queries requiring both meaning and exact matches

### ❌ Not Recommended

- **Latency-sensitive**: LLM call adds ~200-500ms overhead
- **Simple queries**: When basic dense or sparse retrieval is sufficient
- **High-volume**: Cost of LLM calls may be prohibitive

## Performance Characteristics

### Latency

- **Dense retrieval**: ~50-100ms
- **Sparse retrieval**: ~20-50ms
- **LLM scoring**: ~200-500ms
- **Total**: ~300-700ms per query

### Cost

- **Dense retrieval**: Free (local ChromaDB)
- **Sparse retrieval**: Free (local Elasticsearch)
- **LLM scoring**: ~$0.0001 per query (gpt-4o-mini)

### Accuracy

Studies show DAT improves retrieval quality by 15-30% over fixed-weight hybrid approaches.

## Comparison with Other Strategies

| Strategy | Semantic | Keyword | Dynamic | Latency | Cost |
|----------|----------|---------|---------|---------|------|
| **Dense Only** | ✅ | ❌ | ❌ | Low | Free |
| **Sparse Only** | ❌ | ✅ | ❌ | Low | Free |
| **Fixed Hybrid** | ✅ | ✅ | ❌ | Low | Free |
| **DAT** | ✅ | ✅ | ✅ | Medium | Low |

## Advanced Features

### Custom Effectiveness Scoring

```python
class CustomScorer(EffectivenessScorer):
    async def score_results(self, query, dense_results, sparse_results):
        # Custom scoring logic
        return {"dense_score": 0.8, "sparse_score": 0.6}

scorer = CustomScorer(llm_client)
tuner = AlphaTuner(scorer)
```

### Result Metadata

Each result includes metadata about the retrieval:

```python
result.metadata = {
    "alpha": 0.75,              # Calculated alpha
    "dense_score": 0.68,        # Original dense score
    "sparse_score": 0.12,       # Original sparse score
    "retrieval_method": "hybrid" # "dense", "sparse", or "hybrid"
}
```

## Troubleshooting

### Low Quality Results

1. **Check index quality**: Ensure documents are properly indexed in both stores
2. **Adjust top_k**: Increase `top_k_dense` and `top_k_sparse`
3. **Review LLM model**: Try a more capable model (e.g., gpt-4)

### High Latency

1. **Disable dynamic tuning**: Use fixed weights
2. **Reduce top_k**: Retrieve fewer candidates
3. **Use faster LLM**: Switch to gpt-3.5-turbo

### LLM Errors

1. **Check API key**: Ensure OpenAI API key is valid
2. **Handle rate limits**: Implement retry logic
3. **Fallback to default**: Strategy falls back to default alpha on error

## References

### Core Papers

**Dynamic Alpha Tuning (DAT)**
- Hsu, H.-L., & Tzeng, J. (2025). "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation"  
  *arXiv preprint* - [arXiv:2503.23013](https://arxiv.org/abs/2503.23013)

**Retrieval-Augmented Generation**
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"  
  [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

**Dense Passage Retrieval**
- Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering"  
  [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

**BM25 Algorithm**
- Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"  
  [Foundations and Trends in Information Retrieval](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)

### Hybrid Search

**Hybrid Search Best Practices**
- Pinecone. "Hybrid Search Explained"  
  [https://www.pinecone.io/learn/hybrid-search/](https://www.pinecone.io/learn/hybrid-search/)

**Elasticsearch Hybrid Search**
- Elastic. "Combining Full-Text and Vector Search"  
  [https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)

### Related Work

**ColBERT (Late Interaction)**
- Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search"  
  [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)

**Self-RAG**
- Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique"  
  [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

**Graph RAG**
- Microsoft Research (2024). "Graph RAG: Unlocking LLM Discovery on Narrative Private Data"  
  [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
