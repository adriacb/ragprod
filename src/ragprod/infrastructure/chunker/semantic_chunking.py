from .base import BaseChunker
from pydantic import BaseModel
#from ..document import Document
from ragprod.domain.embedding import EmbeddingModel
from ragprod.domain.chunk import Chunk
import logging



class SemanticChunker(BaseChunker, BaseModel):
    
    logger = logging.getLogger(__name__) 
    
    def __init__(self, 
            embedding_model: EmbeddingModel,
            max_tokens: int = 512,
            cluster_threshold: float = 0.5,
            similatity_threshold: float = 0.4
        ):
        
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.cluster_threshold = cluster_threshold
        self.similarity_threshold = similarity_threshold
        self.tokenizer = self.model.tokenizer if hasattr(self.model, "tokenizer") else None

    def get_embeddings(self, chunks: List[Chunk]) -> List[Embeddings]:
        pass