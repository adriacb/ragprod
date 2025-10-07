from ragprod.domain.client.base import BaseClient
from ragprod.domain.document import Document
from typing import Literal, List
import weaviate
import logging


class WeaviateClient(BaseClient):
    def __init__(self, 
                    port: int = 8080, 
                    grpc_port: int = 8081,
                    connect_to: Literal["embedded", "local"] = "local"
                    ):
        self.port = port
        self.grpc_port = grpc_port
        self.client = self._connect()
        self.logger = logging.getLogger(__name__)

    def _connect(self, connect_to: Literal["embedded", "local"] = "local"):
        try:
            if connect_to == "embedded":
                return weaviate.connect_to_embedded(port=self.port, grpc_port=self.grpc_port)
            elif connect_to == "local":
                return weaviate.connect_to_local(port=self.port, grpc_port=self.grpc_port)
        except Exception as e:
            raise Exception(f"Failed to connect to Weaviate: {e}")
    
    def get_collection(self, collection_name: str):
        return self.client.collections.get(collection_name)
    
    def create_collection(self, collection_name: str):
        return self.client.collections.create(collection_name)
    
    def delete_collection(self, collection_name: str):
        return self.client.collections.delete(collection_name)

    def retrieve(self, query: str, collection_name: str, limit: int = 5) -> List[Document]:
        try:
            chunks = self.get_collection(collection_name)
        except Exception as e:
            raise Exception(f"Failed to get collection: {e}")
        
        try:
            result = chunks.query.near_text(query=query, limit=limit)
            return [
                Document(
                    content=obj.properties["content"], 
                    source=obj.properties["source"],
                    title=obj.properties["title"],
                    metadata=obj.properties) 
                    for obj in result.objects
                ]
        except Exception as e:
            raise Exception(f"Failed to query collection: {e}")
        finally:
            return result