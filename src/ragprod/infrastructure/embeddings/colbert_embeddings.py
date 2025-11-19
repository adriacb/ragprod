import torch
import asyncio
import logging
from typing import List
from ragprod.domain.embedding import EmbeddingModel
from colbert.modeling.colbert import ColBERT
from colbert.infra.config import ColBERTConfig

class ColBERTEmbeddings(EmbeddingModel):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        colbert_config: dict | None = None,
    ):
        super().__init__()

        self.model_name = model_name
        self._colbert_config = colbert_config or {}

        # lazy-loaded model placeholder
        self._model: ColBERT | None = None

        self.logger.info(f"Initialized ColBERTEmbeddings for '{self.model_name}' on device: {self.device}")

    # -------------------------------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Return the best available device."""
        device_checks = {
            "cuda": torch.cuda.is_available,
            "mps": torch.backends.mps.is_available,
            "cpu": lambda: True,
        }
        name = next(k for k, check in device_checks.items() if check())
        return torch.device(name)

    @property
    def model(self) -> ColBERT:
        """Lazy-load the ColBERT model on the correct device."""
        if self._model is None:
            config = ColBERTConfig(transformer=self.model_name, **self._colbert_config)
            # Direct instantiation instead of from_pretrained
            self._model = ColBERT(config)
            self._model.to(self.device)
            self._model.eval()
            self.logger.info(f"ColBERT model loaded on device: {self.device}")
        return self._model


    # -------------------------------------------------------------------------
    # METHODS
    # -------------------------------------------------------------------------
    async def embed_query(self, query: str) -> torch.Tensor:
        """Embed a single query string asynchronously on the correct device."""
        return await asyncio.to_thread(self.model.query, [query], device=self.device)

    async def embed_documents(self, texts: List[str]) -> torch.Tensor:
        """Embed multiple documents asynchronously on the correct device."""
        return await asyncio.to_thread(self.model.doc, texts, device=self.device)

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.dim

    def similarity(self, query: str, documents: List[str], batch_size: int = 64) -> List[float]:
        """
        Compute cosine similarity between a query and documents in a fully GPU-optimized way.
        
        Args:
            query: single query string
            documents: list of document strings
            batch_size: optional batch size to avoid GPU memory overflow
        
        Returns:
            List of cosine similarity scores
        """
        self.model.eval()
        query_emb = self.model.query([query]).to(self.device)
        query_emb = query_emb / (torch.norm(query_emb, dim=-1, keepdim=True) + 1e-10)

        scores = []
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                doc_embs = self.model.doc(batch_docs).to(self.device)
                doc_embs = doc_embs / (torch.norm(doc_embs, dim=-1, keepdim=True) + 1e-10)
                batch_scores = torch.matmul(doc_embs, query_emb.T).squeeze(-1)
                scores.extend(batch_scores.cpu().tolist())

        return scores
