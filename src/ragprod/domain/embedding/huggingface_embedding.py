from sentence_transformers import SentenceTransformer
from typing import List
import torch
import asyncio

class HuggingFaceEmbedder:
    def __init__(self, model_name: str = "jinaai/jina-code-embeddings-0.5b", model_kwargs: dict = None, tokenizer_kwargs: dict = None):
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={
                "dtype": torch.bfloat16,
                # "attn_implementation": "flash_attention_2",
                "device_map": "cuda"
            } if model_kwargs is None else model_kwargs,
            tokenizer_kwargs={
                "padding_side": "left"
            } if tokenizer_kwargs is None else tokenizer_kwargs,
        )

    async def embed_query(self, query: str, prompt_name: str = "nl2code_query") -> List[float]:
        return await asyncio.to_thread(self.model.encode, query, prompt_name=prompt_name)

    async def embed_documents(self, texts: List[str], prompt_name: str = "nl2code_query") -> List[List[float]]:
        return await asyncio.to_thread(self.model.encode, texts, prompt_name=prompt_name)

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def similarity(self, query: str, documents: List[str]) -> List[float]:
        return self.model.similarity(query, documents)
