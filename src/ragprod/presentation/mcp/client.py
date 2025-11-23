import os
from typing import Optional
from ragprod.application.use_cases import GetClientService, GetEmbeddingsService
from ragprod.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Global instances
_clientDB = None
_embedder = None


def init_database_client():
    """
    Initialize database client based on environment variables.
    
    Environment Variables:
        DB_TYPE: Type of database (chroma, weaviate)
        DB_MODE: Mode of operation (local, remote)
        
        For ChromaDB:
            CHROMA_PERSIST_DIRECTORY: Directory for local persistence
            CHROMA_COLLECTION_NAME: Collection name
            CHROMA_API_HOST: Host for remote ChromaDB
            CHROMA_API_PORT: Port for remote ChromaDB
            
        For Weaviate:
            WEAVIATE_URL: Weaviate server URL
            WEAVIATE_API_KEY: API key for Weaviate Cloud
            WEAVIATE_GRPC_PORT: gRPC port
            WEAVIATE_COLLECTION_NAME: Collection name
    """
    global _clientDB, _embedder
    
    if _clientDB is not None:
        logger.info("Database client already initialized")
        return _clientDB
    
    # Get database configuration from environment
    db_type = os.getenv("DB_TYPE", "chroma").lower()
    db_mode = os.getenv("DB_MODE", "local").lower()
    
    logger.info(f"Initializing database client: type={db_type}, mode={db_mode}")
    
    # Initialize embeddings
    if _embedder is None:
        _embedder = _init_embeddings()
    
    # Initialize database client based on type
    service = GetClientService()
    
    if db_type == "chroma":
        config = _get_chroma_config(db_mode)
        config["embedding_model"] = _embedder
        _clientDB = service.get("chroma", config)
        
    elif db_type == "weaviate":
        config = _get_weaviate_config(db_mode)
        config["embedding_model"] = _embedder
        _clientDB = service.get("weaviate", config)
        
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    logger.info(f"Database client initialized successfully: {db_type}")
    return _clientDB


def _init_embeddings():
    """Initialize embedding model from environment variables."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "jinaai/jina-code-embeddings-0.5b")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    dtype = os.getenv("EMBEDDING_DTYPE", "bfloat16")
    
    logger.info(f"Initializing embeddings: provider={provider}, model={model_name}")
    
    embeddings_service = GetEmbeddingsService()
    embedder = embeddings_service.get(provider, {
        "model_name": model_name,
        "model_kwargs": {"device_map": device, "dtype": dtype},
        "tokenizer_kwargs": {"padding_side": "left"},
    })
    
    logger.info("Embeddings initialized successfully")
    return embedder


def _get_chroma_config(mode: str) -> dict:
    """Get ChromaDB configuration based on mode."""
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ragprod")
    
    if mode == "remote":
        # Remote ChromaDB via HTTP
        api_host = os.getenv("CHROMA_API_HOST")
        api_port = int(os.getenv("CHROMA_API_PORT", "8000"))
        
        if not api_host:
            raise ValueError("CHROMA_API_HOST must be set for remote mode")
        
        logger.info(f"Using remote ChromaDB at {api_host}:{api_port}")
        return {
            "api_host": api_host,
            "api_port": api_port,
            "collection_name": collection_name,
        }
    else:
        # Local persistent ChromaDB
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chromadb_data")
        logger.info(f"Using local ChromaDB at {persist_dir}")
        return {
            "persist_directory": persist_dir,
            "collection_name": collection_name,
        }


def _get_weaviate_config(mode: str) -> dict:
    """Get Weaviate configuration based on mode."""
    collection_name = os.getenv("WEAVIATE_COLLECTION_NAME", "ragprod")
    
    if mode == "remote":
        # Remote Weaviate (cloud or self-hosted)
        url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")
        
        if not url:
            raise ValueError("WEAVIATE_URL must be set for remote mode")
        
        logger.info(f"Using remote Weaviate at {url}")
        config = {
            "url": url,
            "collection_name": collection_name,
        }
        
        if api_key:
            config["api_key"] = api_key
            logger.info("Using Weaviate with API key authentication")
        
        return config
    else:
        # Local Weaviate
        port = int(os.getenv("WEAVIATE_PORT", "8080"))
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        
        logger.info(f"Using local Weaviate at localhost:{port}")
        return {
            "port": port,
            "grpc_port": grpc_port,
            "connect_to": "local",
            "collection_name": collection_name,
        }


def get_client():
    """Get the initialized database client."""
    if _clientDB is None:
        return init_database_client()
    return _clientDB


# Initialize on module import for backward compatibility
try:
    clientDB = init_database_client()
    logger.info("Database client initialized on module import")
except Exception as e:
    logger.error(f"Failed to initialize database client: {e}")
    clientDB = None