# Retrieval Service Usage Examples

## Basic Usage

### Simple Dense Retrieval

```python
from ragprod.application.use_cases.retrieval_service import create_retrieval_service
from ragprod.infrastructure.client.chromadb import AsyncChromaDBClient

# Initialize ChromaDB client
chromadb = AsyncChromaDBClient(...)

# Create service with dense strategy
service = create_retrieval_service(
    dense_store=chromadb,
    strategy="dense"
)

# Retrieve documents
results = await service.retrieve(
    query="What is Python?",
    collection_name="docs",
    top_k=5
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.document.raw_text[:100]}...")
```

### DAT Hybrid Retrieval

```python
from ragprod.application.use_cases.retrieval_service import create_retrieval_service
from ragprod.infrastructure.client.chromadb import AsyncChromaDBClient
from ragprod.infrastructure.client.elasticsearch_client import ElasticsearchClient

# Initialize clients
chromadb = AsyncChromaDBClient(...)
elasticsearch = ElasticsearchClient(hosts=["http://localhost:9200"])

# Create service with DAT strategy
service = create_retrieval_service(
    dense_store=chromadb,
    sparse_store=elasticsearch,
    strategy="dat",
    use_dynamic_tuning=True,
    top_k_dense=20,
    top_k_sparse=20
)

# Retrieve documents
results = await service.retrieve(
    query="How do I write concurrent code in Python?",
    collection_name="docs",
    top_k=5
)

for result in results:
    print(f"Score: {result.score:.3f} | Method: {result.retrieval_method}")
    print(f"Content: {result.document.raw_text[:100]}...")
    if result.metadata:
        print(f"Alpha: {result.metadata.get('alpha', 'N/A')}")
```

## Advanced Configuration

### Custom DAT Configuration

```python
from ragprod.application.use_cases.retrieval_service import RetrievalService
from ragprod.domain.retrieval.strategies.dat import DATConfig

# Custom DAT config
config = DATConfig(
    dense_weight_default=0.7,      # Favor dense retrieval
    sparse_weight_default=0.3,
    top_k_dense=30,                # Retrieve more candidates
    top_k_sparse=30,
    use_dynamic_tuning=True,
    llm_model="gpt-4",             # Use more capable model
    temperature=0.0
)

service = RetrievalService(
    dense_store=chromadb,
    sparse_store=elasticsearch,
    strategy="dat",
    dat_config=config
)
```

### Disable Dynamic Tuning

For faster retrieval without LLM calls:

```python
service = create_retrieval_service(
    dense_store=chromadb,
    sparse_store=elasticsearch,
    strategy="dat",
    use_dynamic_tuning=False,      # Use fixed weights
    dense_weight_default=0.6       # 60% dense, 40% sparse
)
```

## Integration with MCP

### MCP Tool Integration

```python
# In your MCP tool
from ragprod.application.use_cases.retrieval_service import create_retrieval_service

async def rag_retrieve(query: str, limit: int = 5):
    """Retrieve documents using configured strategy."""
    service = create_retrieval_service(
        dense_store=chromadb_client,
        sparse_store=es_client,
        strategy=settings.retrieval_strategy  # From settings
    )
    
    results = await service.retrieve(query, "documents", top_k=limit)
    
    return [
        {
            "content": r.document.raw_text,
            "score": r.score,
            "method": r.retrieval_method,
            "source": r.document.source
        }
        for r in results
    ]
```

## Error Handling

```python
from ragprod.domain.retrieval.exceptions import RetrievalError, StrategyNotFoundError

try:
    service = create_retrieval_service(
        dense_store=chromadb,
        strategy="dat"  # Missing sparse_store!
    )
except StrategyNotFoundError as e:
    print(f"Strategy error: {e}")

try:
    results = await service.retrieve("query", "collection")
except RetrievalError as e:
    print(f"Retrieval failed: {e}")
    # Fallback to simple search
    results = await chromadb.retrieve("query", "collection")
```

## Performance Monitoring

```python
import time

start = time.time()
results = await service.retrieve(query, "docs", top_k=5)
latency = time.time() - start

print(f"Retrieval latency: {latency*1000:.0f}ms")
print(f"Results: {len(results)}")
print(f"Strategy: {service.strategy_name}")
```

## A/B Testing

```python
# Test different strategies
strategies = ["dense", "dat"]
results_by_strategy = {}

for strategy in strategies:
    service = create_retrieval_service(
        dense_store=chromadb,
        sparse_store=elasticsearch,
        strategy=strategy
    )
    
    results = await service.retrieve(query, "docs", top_k=5)
    results_by_strategy[strategy] = results

# Compare results
print("Dense:", [r.document.id for r in results_by_strategy["dense"]])
print("DAT:", [r.document.id for r in results_by_strategy["dat"]])
```

## Best Practices

1. **Reuse service instances**: Create once, use many times
2. **Handle errors gracefully**: Always catch `RetrievalError`
3. **Monitor latency**: Track retrieval times
4. **Start simple**: Use dense strategy first, upgrade to DAT if needed
5. **Configure appropriately**: Tune top_k based on your data
