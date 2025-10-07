from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, **kwargs) -> List[str]:
        pass