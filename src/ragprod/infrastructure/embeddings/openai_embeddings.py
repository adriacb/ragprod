import logging
import asyncio
from typing import List
import openai
from ragprod.domain.embedding import EmbeddingModel

class OpenAIEmbeddings(EmbeddingModel):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        batch_size: int = 1000,
    ):
        super().__init__()

        self.model_name = model_name
        self.batch_size = batch_size
        if openai_api_key:
            openai.api_key = openai_api_key

        self.logger.info(f"Initialized OpenAIEmbeddings with model '{self.model_name}'")

    # -------------------------------------------------------------------------
    # METHODS
    # -------------------------------------------------------------------------
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string asynchronously using OpenAI API."""
        embeddings = await asyncio.to_thread(self._embed_texts, [query])
        return embeddings[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents asynchronously using batching."""
        return await asyncio.to_thread(self._embed_texts, texts)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Internal helper to call OpenAI embeddings API with batching."""
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = openai.Embedding.create(
                input=batch,
                model=self.model_name
            )
            batch_embeddings = [item["embedding"] for item in response["data"]]
            embeddings.extend(batch_embeddings)

        return embeddings

    def get_dimension(self) -> int:
        """
        Return the dimensionality of the embedding.
        Note: OpenAI embeddings have fixed dimensions depending on the model.
        """
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return model_dims.get(self.model_name, 1536)  # fallback to 1536
