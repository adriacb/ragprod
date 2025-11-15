from abc import ABC, abstractmethod
from typing import List
from ragprod.domain.document import Document

class BaseClient(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[Document]:
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        pass
