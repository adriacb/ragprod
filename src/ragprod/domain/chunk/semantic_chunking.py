from .base import BaseChunker
from typing import List
from pydantic import BaseModel
from ..document import Document

class SemanticChunker(BaseChunker, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)