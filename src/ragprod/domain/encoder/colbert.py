from .base import BaseEncoder
from ..document import Document
from pydantic import BaseModel
from typing import List

class ColBERTEncoder(BaseEncoder, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, document: Document) -> List[float]:
        pass

