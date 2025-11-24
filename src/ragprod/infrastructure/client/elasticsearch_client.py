"""Elasticsearch client for BM25 sparse retrieval."""

import logging
from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch

from ragprod.domain.document import Document


class ElasticsearchClient:
    """Elasticsearch client for BM25 sparse retrieval in hybrid RAG systems."""

    def __init__(self, hosts: List[str] = None):
        """Initialize Elasticsearch client.

        Args:
            hosts: List of Elasticsearch hosts. Defaults to ["http://localhost:9200"]
        """
        self.hosts = hosts or ["http://localhost:9200"]
        self.client = AsyncElasticsearch(hosts=self.hosts)
        self.logger = logging.getLogger(__name__)

    async def create_index(self, index_name: str) -> None:
        """Create index with BM25 settings.

        Args:
            index_name: Name of the index to create
        """
        if await self.client.indices.exists(index=index_name):
            self.logger.info(f"Index {index_name} already exists")
            return

        await self.client.indices.create(
            index=index_name,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "similarity": {"default": {"type": "BM25"}},
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "source": {"type": "keyword"},
                        "title": {"type": "text"},
                        "metadata": {"type": "object", "enabled": True},
                    }
                },
            },
        )
        self.logger.info(f"Created index {index_name} with BM25 similarity")

    async def index_documents(self, index_name: str, documents: List[Document]) -> None:
        """Index documents for BM25 search.

        Args:
            index_name: Name of the index
            documents: List of documents to index
        """
        await self.create_index(index_name)

        if not documents:
            self.logger.warning("No documents to index")
            return

        # Bulk index using the bulk helper
        actions = []
        for doc in documents:
            actions.append({"index": {"_index": index_name, "_id": doc.id}})
            actions.append(
                {
                    "text": doc.raw_text,
                    "source": doc.source,
                    "title": doc.title,
                    "metadata": doc.metadata or {},
                }
            )

        if actions:
            response = await self.client.bulk(operations=actions, refresh=True)
            if response.get("errors"):
                self.logger.error(f"Bulk indexing had errors: {response}")
            else:
                self.logger.info(f"Indexed {len(documents)} documents to {index_name}")

    async def search(self, index: str, query: str, size: int = 10) -> Dict[str, Any]:
        """Perform BM25 search.

        Args:
            index: Index name to search
            query: Search query text
            size: Number of results to return

        Returns:
            Elasticsearch response with hits
        """
        try:
            response = await self.client.search(
                index=index,
                body={"query": {"match": {"text": query}}, "size": size},
            )
            return response
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {"hits": {"hits": []}}

    async def delete_index(self, index_name: str) -> None:
        """Delete an index.

        Args:
            index_name: Name of the index to delete
        """
        if await self.client.indices.exists(index=index_name):
            await self.client.indices.delete(index=index_name)
            self.logger.info(f"Deleted index {index_name}")

    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index

        Returns:
            True if index exists, False otherwise
        """
        return await self.client.indices.exists(index=index_name)

    async def close(self) -> None:
        """Close the Elasticsearch client."""
        await self.client.close()
        self.logger.info("Elasticsearch client closed")
