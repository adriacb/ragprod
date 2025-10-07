from ragprod.domain.client.base import BaseClient
from ragprod.domain.document import Document
from ragprod.domain.embedding import EmbeddingModel
from typing import List, Union, Callable
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.base import AddableMixin, Docstore

class FaissClient(BaseClient):
    def __init__(self, 
            index: Any, 
            docstore: Docstore, 
            index_to_doc_map: dict,
            embedding_model: Union[
                    Callable[[str], List[float]],
                    EmbeddingModel] = None
            ):
        self.index_path = index_path
        self.index_url = index_url
        self.index = self._connect()
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
    
    def _connect(self):
        return faiss.read_index(self.index_path)

    def create_vector_store(self):
        return FAISS()
    
    def add_documents(self, vectors: List[List[float]]):
        self.index.add(vectors)
    
    def retrieve(self, 
            query: str, 
            k: int
            ) -> List[Document]:
        embedded_query = self.embedding_model.encode(query)
        embedded_query = embedded_query.astype(np.float32)
        distances, indices = self.index.search(query, k)
        return [self.index.reconstruct(i) for i in indices]
    
