from typing import List
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, query: str) -> List[float]: ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    
    @abstractmethod
    def get_dimension(self) -> int: ...
