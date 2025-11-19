from .base import BaseChunk
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk(BaseChunk):
    id: str
    content: str
    children: List[BaseChunk]

    def get_children(self) -> List[BaseChunk]:
        return self.children