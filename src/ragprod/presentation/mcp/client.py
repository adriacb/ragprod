from ragprod.application.use_cases import GetClientService, GetEmbeddingsService


embeddingsService = GetEmbeddingsService()
embedder = embeddingsService.get("huggingface", {
    "model_name": "jinaai/jina-code-embeddings-0.5b",
    "model_kwargs": {"device_map": "cpu", "dtype": "bfloat16"},
    "tokenizer_kwargs": {"padding_side": "left"},
    #"attn_implementation": "flash_attention_2",
})
service = GetClientService()


try:
    clientDB = service.get("chroma", {
        "persist_directory": "./chromadb_test",
        "collection_name": "test",
        "embedding_model": embedder
    })
    print("Initialized.")
except Exception as e:
    print(f"Error initializing client: {e}")
    raise e