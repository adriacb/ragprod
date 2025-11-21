from .base import BaseChunker
from pydantic import BaseModel
from typing import List
#from ..document import Document
from ragprod.domain.embedding import EmbeddingModel
from ragprod.domain.chunk import Chunk
from ragprod.infrastructure.logger import get_logger



class SemanticChunker(BaseChunker, BaseModel):
    
    logger = get_logger(__name__) 
    
    def __init__(self, 
            embedding_model: EmbeddingModel,
            max_tokens: int = 512,
            cluster_threshold: float = 0.5,
            similarity_threshold: float = 0.4
        ):
        
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.cluster_threshold = cluster_threshold
        self.similarity_threshold = similarity_threshold
        self.tokenizer = self.model.tokenizer if hasattr(self.model, "tokenizer") else None

    def get_embeddings(self, chunks: List[Chunk]) -> list:
        pass