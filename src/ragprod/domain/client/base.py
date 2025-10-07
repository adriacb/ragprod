from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ragprod.core.document import BaseDocument

class BaseClient(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> List[BaseDocument]:
        pass
    