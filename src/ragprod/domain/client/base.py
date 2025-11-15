from abc import ABC, abstractmethod
from typing import List
from ragprod.domain.document.base import BaseDocument

class BaseClient(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> List[BaseDocument]:
        pass
    