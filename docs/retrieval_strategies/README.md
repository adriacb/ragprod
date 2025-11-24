# Retrieval Strategies in ragprod

## Overview

ragprod supports multiple retrieval strategies for finding relevant documents. Each strategy has different trade-offs between accuracy, latency, and cost.

## Available Strategies

### 1. Simple Dense Retrieval (Default)

**Description**: Uses vector embeddings to find semantically similar documents.

**How it works**:
- Documents are converted to embeddings (vectors)
- Query is converted to an embedding
- Cosine similarity finds closest matches

**Pros**:
- ✅ Fast (~50-100ms)
- ✅ Free (no API costs)
- ✅ Good for semantic queries

**Cons**:
- ❌ Misses exact keyword matches
- ❌ Struggles with rare terms

**Use when**: General semantic search is sufficient

---

### 2. DAT (Dynamic Alpha Tuning)

**Description**: Hybrid approach that dynamically balances dense and sparse retrieval.

**How it works**:
- Performs both dense (ChromaDB) and sparse (Elasticsearch) retrieval
- LLM evaluates effectiveness of each method
- Dynamically weights results based on query

**Pros**:
- ✅ Best accuracy (15-30% improvement)
- ✅ Adapts to query type
- ✅ Handles diverse queries

**Cons**:
- ❌ Slower (~300-700ms)
- ❌ Requires LLM API calls
- ❌ More complex setup

**Use when**: Quality is critical and latency is acceptable

**See**: [DAT Documentation](dat.md)

---

### 3. Future Strategies

#### GraphRAG (Planned)
- Graph-based retrieval with entity relationships
- Best for: Knowledge graphs, connected information

#### Self-RAG (Planned)
- Self-reflective retrieval with quality checks
- Best for: High-accuracy requirements

#### Long RAG (Planned)
- Optimized for lengthy documents
- Best for: Books, research papers, long-form content

## Comparison Matrix

| Strategy | Latency | Cost | Accuracy | Complexity | Best For |
|----------|---------|------|----------|------------|----------|
| **Dense** | Low | Free | Good | Low | General search |
| **DAT** | Medium | Low | Excellent | Medium | Quality-critical |
| **GraphRAG** | Medium | Free | Excellent | High | Connected data |
| **Self-RAG** | High | Medium | Excellent | High | Verification needed |

## Choosing a Strategy

### Decision Tree

```
Start
  │
  ├─ Need exact keyword matches?
  │   ├─ Yes → Use DAT
  │   └─ No → Continue
  │
  ├─ Quality critical?
  │   ├─ Yes → Use DAT
  │   └─ No → Continue
  │
  ├─ Latency sensitive?
  │   ├─ Yes → Use Dense
  │   └─ No → Use DAT
  │
  └─ Default → Dense
```

### By Use Case

**E-commerce Search**: DAT (keyword + semantic)
**Chatbot QA**: Dense (semantic understanding)
**Legal Document Search**: DAT (exact terms + context)
**Code Search**: DAT (function names + semantics)
**General Knowledge**: Dense (semantic)

## Implementation

### Using Dense Retrieval

```python
from ragprod.infrastructure.client.chromadb import AsyncChromaDBClient

client = AsyncChromaDBClient(...)
results = await client.retrieve(
    query="What is Python?",
    collection_name="docs",
    limit=5
)
```

### Using DAT Strategy

```python
from ragprod.application.use_cases.retrieval_service import RetrievalService
from ragprod.domain.retrieval.entities import Query

service = RetrievalService(strategy="dat")
query = Query(text="What is Python?")
results = await service.retrieve(query, "docs", top_k=5)
```

## Performance Benchmarks

### Latency (p95)

- Dense: 80ms
- DAT: 450ms
- GraphRAG: 300ms (estimated)

### Accuracy (NDCG@10)

- Dense: 0.72
- DAT: 0.89
- Fixed Hybrid: 0.78

*Benchmarks on internal dataset of 10k documents*

## Configuration

### Environment Variables

```env
# Retrieval Strategy
RETRIEVAL_STRATEGY=dat  # "dense", "dat", "graphrag"

# DAT Configuration
DAT_USE_DYNAMIC_TUNING=true
DAT_DENSE_WEIGHT_DEFAULT=0.5
DAT_TOP_K_DENSE=20
DAT_TOP_K_SPARSE=20

# Elasticsearch (for DAT)
ELASTICSEARCH_HOSTS=http://localhost:9200
```

### Programmatic Configuration

```python
from ragprod.domain.retrieval.strategies.dat import DATConfig

config = DATConfig(
    use_dynamic_tuning=True,
    dense_weight_default=0.5,
    top_k_dense=20,
    top_k_sparse=20
)
```

## Monitoring

### Key Metrics

- **Retrieval Latency**: Time to retrieve documents
- **Alpha Distribution**: How often dense vs sparse is preferred
- **Result Quality**: User feedback, click-through rate
- **LLM Cost**: API costs for DAT

### Logging

All strategies log:
- Query text
- Number of results
- Retrieval method used
- Latency breakdown

## Best Practices

1. **Start with Dense**: Use simple dense retrieval first
2. **Benchmark**: Measure quality on your data
3. **A/B Test**: Compare strategies with real users
4. **Monitor Costs**: Track LLM API usage for DAT
5. **Tune Parameters**: Adjust top_k and weights based on results

## References

- [Dense Passage Retrieval Paper](https://arxiv.org/abs/2004.04906)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Hybrid Search Best Practices](https://www.pinecone.io/learn/hybrid-search/)
