import openai
from typing import List

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def embed_query(self, query: str) -> List[float]:
        response = openai.Embedding.create(input=query, model=self.model)
        return response['data'][0]['embedding']

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = openai.Embedding.create(input=texts, model=self.model)
        return [item['embedding'] for item in response['data']]

    def get_dimension(self) -> int:
        # These sizes are fixed per model and can be hardcoded or queried
        if self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        else:
            raise ValueError(f"Unknown embedding size for model {self.model}")
