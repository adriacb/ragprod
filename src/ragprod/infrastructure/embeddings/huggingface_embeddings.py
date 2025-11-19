import torch
import asyncio
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from ragprod.domain.embedding import EmbeddingModel


class HuggingFaceEmbeddings(EmbeddingModel):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "jinaai/jina-code-embeddings-0.5b",
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ):
        super().__init__()

        self.model_name = model_name
        self._user_model_kwargs = model_kwargs or {}
        self._user_tokenizer_kwargs = tokenizer_kwargs or {}

        # lazy-loaded model placeholder
        self._model: SentenceTransformer | None = None

        self.logger.info(f"Initialized HuggingFaceEmbeddings for '{self.model_name}' on device: {self.device}")

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
    def device_map(self) -> str:
        """Return HF-compatible device_map string."""
        mapping = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}
        if self.device.type not in mapping:
            raise RuntimeError(f"Unsupported device type: {self.device.type}")
        return mapping[self.device.type]

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model and ensure it's on the correct device."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                model_kwargs=self._build_model_kwargs(),
                tokenizer_kwargs=self._build_tokenizer_kwargs(),
            )

            # Ensure model is on the correct device
            self._model.to(self.device)
            self.logger.info(f"Model loaded on device: {self.device}")

        return self._model

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------
    def _build_model_kwargs(self) -> dict:
        """Merge default model kwargs with user-provided ones."""
        defaults = {"dtype": torch.bfloat16, "device_map": self.device_map}
        return {**defaults, **self._user_model_kwargs}

    def _build_tokenizer_kwargs(self) -> dict:
        """Merge default tokenizer kwargs with user-provided ones."""
        defaults = {"padding_side": "left"}
        return {**defaults, **self._user_tokenizer_kwargs}

    # -------------------------------------------------------------------------
    # METHODS
    # -------------------------------------------------------------------------
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string asynchronously."""
        return await asyncio.to_thread(self.model.encode, query, convert_to_numpy=True)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents asynchronously."""
        return await asyncio.to_thread(self.model.encode, texts, convert_to_numpy=True)

    def get_dimension(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()

    def similarity(self, query: str, documents: List[str]) -> List[float]:
        """Compute cosine similarity between a query and a list of documents."""
        query_vec = self.model.encode(query, convert_to_numpy=True)
        doc_vecs = self.model.encode(documents, convert_to_numpy=True)

        # Cosine similarity
        query_norm = query_vec / (torch.norm(torch.tensor(query_vec)) + 1e-10)
        doc_norms = doc_vecs / (torch.norm(torch.tensor(doc_vecs), dim=1, keepdim=True) + 1e-10)

        return (doc_norms @ torch.tensor(query_norm)).tolist()