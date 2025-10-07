from .base import BaseEmbeddings
from ..document import Document
from pydantic import BaseModel
from typing import List

class ColBERTEmbeddings(BaseEmbeddings, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
