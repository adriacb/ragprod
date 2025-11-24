# Alternative Sparse Search Clients for Hybrid RAG

This document provides Docker Compose configurations and client examples for alternative sparse search engines that can be used instead of Elasticsearch for BM25 retrieval in DAT.

## 1. Elasticsearch (Recommended)

**Best for**: Production systems, large-scale deployments, enterprise features

### Docker Compose

```yaml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
  container_name: ragprod-elasticsearch
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false
    - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
  ports:
    - "9200:9200"
  volumes:
    - elasticsearch_data:/usr/share/elasticsearch/data
  networks:
    - ragprod-network
```

### Client Usage

```python
from ragprod.infrastructure.client.elasticsearch_client import ElasticsearchClient

client = ElasticsearchClient(hosts=["http://localhost:9200"])
await client.index_documents("my_index", documents)
results = await client.search("my_index", "query text", size=10)
```

---

## 2. OpenSearch (AWS-Friendly Alternative)

**Best for**: AWS deployments, Elasticsearch alternative without licensing concerns

### Docker Compose

```yaml
opensearch:
  image: opensearchproject/opensearch:2.11.0
  container_name: ragprod-opensearch
  environment:
    - discovery.type=single-node
    - plugins.security.disabled=true
    - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
  ports:
    - "9200:9200"
    - "9600:9600"
  volumes:
    - opensearch_data:/usr/share/opensearch/data
  networks:
    - ragprod-network
```

### Client Implementation

```python
# ragprod/infrastructure/client/opensearch_client.py
from opensearchpy import AsyncOpenSearch
from typing import List, Dict, Any
from ragprod.domain.document import Document

class OpenSearchClient:
    """OpenSearch client for BM25 sparse retrieval."""
    
    def __init__(self, hosts: List[str] = None):
        self.hosts = hosts or ["http://localhost:9200"]
        self.client = AsyncOpenSearch(
            hosts=self.hosts,
            use_ssl=False,
            verify_certs=False
        )
    
    async def index_documents(self, index_name: str, documents: List[Document]):
        # Similar to Elasticsearch
        pass
    
    async def search(self, index: str, query: str, size: int = 10):
        return await self.client.search(
            index=index,
            body={"query": {"match": {"text": query}}, "size": size}
        )
```

**Dependency**: `pip install opensearch-py[async]`

---

## 3. Typesense (Lightweight & Fast)

**Best for**: Small to medium deployments, developer-friendly, low resource usage

### Docker Compose

```yaml
typesense:
  image: typesense/typesense:0.25.2
  container_name: ragprod-typesense
  environment:
    - TYPESENSE_DATA_DIR=/data
    - TYPESENSE_API_KEY=xyz  # Change this!
  ports:
    - "8108:8108"
  volumes:
    - typesense_data:/data
  networks:
    - ragprod-network
```

### Client Implementation

```python
# ragprod/infrastructure/client/typesense_client.py
import typesense
from typing import List
from ragprod.domain.document import Document

class TypesenseClient:
    """Typesense client for BM25-like sparse retrieval."""
    
    def __init__(self, host: str = "localhost", port: int = 8108, api_key: str = "xyz"):
        self.client = typesense.Client({
            'nodes': [{'host': host, 'port': str(port), 'protocol': 'http'}],
            'api_key': api_key,
            'connection_timeout_seconds': 2
        })
    
    async def create_collection(self, collection_name: str):
        schema = {
            'name': collection_name,
            'fields': [
                {'name': 'text', 'type': 'string'},
                {'name': 'source', 'type': 'string', 'facet': True},
                {'name': 'title', 'type': 'string'},
            ]
        }
        self.client.collections.create(schema)
    
    async def index_documents(self, collection_name: str, documents: List[Document]):
        docs = [
            {
                'id': doc.id,
                'text': doc.raw_text,
                'source': doc.source,
                'title': doc.title
            }
            for doc in documents
        ]
        self.client.collections[collection_name].documents.import_(docs)
    
    async def search(self, collection: str, query: str, size: int = 10):
        return self.client.collections[collection].documents.search({
            'q': query,
            'query_by': 'text,title',
            'per_page': size
        })
```

**Dependency**: `pip install typesense`

---

## 4. Meilisearch (Developer-Friendly)

**Best for**: Prototyping, developer experience, instant search UIs

### Docker Compose

```yaml
meilisearch:
  image: getmeili/meilisearch:v1.5
  container_name: ragprod-meilisearch
  environment:
    - MEILI_MASTER_KEY=masterKey  # Change this!
    - MEILI_ENV=development
  ports:
    - "7700:7700"
  volumes:
    - meilisearch_data:/meili_data
  networks:
    - ragprod-network
```

### Client Implementation

```python
# ragprod/infrastructure/client/meilisearch_client.py
import meilisearch
from typing import List
from ragprod.domain.document import Document

class MeilisearchClient:
    """Meilisearch client for BM25-like sparse retrieval."""
    
    def __init__(self, host: str = "http://localhost:7700", api_key: str = "masterKey"):
        self.client = meilisearch.Client(host, api_key)
    
    async def create_index(self, index_name: str):
        self.client.create_index(index_name, {'primaryKey': 'id'})
        # Configure searchable attributes
        self.client.index(index_name).update_searchable_attributes(['text', 'title'])
    
    async def index_documents(self, index_name: str, documents: List[Document]):
        docs = [
            {
                'id': doc.id,
                'text': doc.raw_text,
                'source': doc.source,
                'title': doc.title
            }
            for doc in documents
        ]
        self.client.index(index_name).add_documents(docs)
    
    async def search(self, index: str, query: str, size: int = 10):
        return self.client.index(index).search(query, {'limit': size})
```

**Dependency**: `pip install meilisearch`

---

## Comparison Matrix

| Feature | Elasticsearch | OpenSearch | Typesense | Meilisearch |
|---------|--------------|------------|-----------|-------------|
| **BM25 Support** | ✅ Native | ✅ Native | ✅ Similar | ✅ Similar |
| **Scalability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Resource Usage** | High | High | Low | Low |
| **Setup Complexity** | Medium | Medium | Easy | Easy |
| **Production Ready** | ✅ | ✅ | ✅ | ✅ |
| **Open Source** | ⚠️ SSPL | ✅ Apache 2.0 | ✅ GPL | ✅ MIT |
| **Best For** | Enterprise | AWS/Cloud | Small-Medium | Prototyping |

## Recommendation

- **Production**: Elasticsearch or OpenSearch
- **Development**: Typesense or Meilisearch
- **AWS**: OpenSearch
- **Low Resources**: Typesense

## Complete Docker Compose Example

```yaml
version: '3.8'

services:
  # Choose ONE of these sparse search engines
  
  # Option 1: Elasticsearch
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: ragprod-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - ragprod-network

  # Option 2: OpenSearch
  # opensearch:
  #   image: opensearchproject/opensearch:2.11.0
  #   ...

  # Option 3: Typesense
  # typesense:
  #   image: typesense/typesense:0.25.2
  #   ...

  # Option 4: Meilisearch
  # meilisearch:
  #   image: getmeili/meilisearch:v1.5
  #   ...

  # ChromaDB for dense retrieval
  chromadb:
    image: chromadb/chroma:latest
    container_name: ragprod-chromadb
    ports:
      - "8000:8000"
    volumes:
      - chromadb_data:/chroma/chroma
    networks:
      - ragprod-network

volumes:
  elasticsearch_data:
  # opensearch_data:
  # typesense_data:
  # meilisearch_data:
  chromadb_data:

networks:
  ragprod-network:
    driver: bridge
```

## Usage in DAT Strategy

All clients follow the same interface pattern:

```python
# In DATStrategy
class DATStrategy:
    def __init__(self, dense_store, sparse_store, ...):
        self.dense_store = dense_store  # ChromaDB
        self.sparse_store = sparse_store  # Any of: Elasticsearch, OpenSearch, Typesense, Meilisearch
    
    async def _sparse_retrieve(self, query, collection_name, top_k):
        # Works with any sparse store that implements search()
        results = await self.sparse_store.search(collection_name, query.text, size=top_k)
        return self._convert_to_retrieval_results(results)
```

The clean architecture allows easy swapping between different sparse search engines!
