# RAGProd 🚀

**RAGProd** is a production-ready **Retrieval-Augmented Generation (RAG)** system built with clean architecture principles. It provides a modular, extensible framework for building RAG applications with support for multiple vector databases, embedding models, chunking strategies, and flexible deployment options.

## 🎯 What is RAGProd?

RAGProd is a comprehensive RAG framework that enables you to:

- **Ingest and store documents** with automatic chunking using multiple strategies
- **Retrieve relevant documents** using semantic search across multiple vector database backends
- **Support multiple embedding models** (HuggingFace, OpenAI, ColBERT)
- **Deploy flexibly** via FastAPI REST API or Model Context Protocol (MCP)
- **Connect to local or cloud databases** (ChromaDB, Weaviate)
- **Monitor and evaluate** RAG performance with integrated observability tools

The system follows clean architecture patterns, making it easy to extend, test, and maintain while supporting production deployments.

## ✨ Key Features

### 🌐 Dual Presentation Layer
- **FastAPI** - RESTful API with automatic OpenAPI documentation
- **FastMCP** - Model Context Protocol for AI application integration
- Both support the same underlying RAG capabilities

### 🗄️ Multi-Vector Database Support
- **ChromaDB** - Local persistent storage or remote API
- **Weaviate** - Local, self-hosted, or Weaviate Cloud
- **FAISS** - High-performance similarity search
- **Qdrant** - Production-ready vector database

### ✂️ Advanced Document Chunking
- **Recursive Character** - Smart text splitting with separators (recommended)
- **Character** - Simple character-based splitting
- **Token** - Token-aware splitting for LLM context limits
- **Markdown** - Structure-aware markdown splitting
- **Semantic** - Similarity-based semantic chunking

### 🤖 Multiple Embedding Models
- **HuggingFace Transformers** - Open-source embedding models
- **OpenAI Embeddings** - GPT-based embeddings
- **ColBERT** - Contextualized late interaction embeddings

### 📊 Monitoring & Observability
- **Langfuse** - LLM observability and analytics
- **LangSmith** - LangChain monitoring
- **Phoenix** - Open-source LLM observability

### 🔍 Advanced Retrieval Strategies
- **Dense Retrieval** - Semantic search using vector embeddings (default)
- **DAT (Dynamic Alpha Tuning)** - Hybrid retrieval with LLM-based dynamic weighting
- **Future**: GraphRAG, Self-RAG, Long RAG support planned

See [Retrieval Strategies Documentation](docs/retrieval_strategies/) for details.

### 🏗️ Clean Architecture
- **Domain Layer** - Core business logic and entities
- **Infrastructure Layer** - External service integrations
- **Application Layer** - Use cases and services (GetChunkerService, GetClientService)
- **Presentation Layer** - FastAPI and MCP interfaces

### 🔧 Production-Ready Features
- Structured logging with `structlog`
- Environment-based configuration
- Async/await support throughout
- Comprehensive unit and integration tests
- Docker support with multiple deployment scenarios
- Automatic semantic versioning with CI/CD

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Presentation Layer (FastAPI & MCP)               │
│  ┌──────────────┐              ┌──────────────┐        │
│  │  FastAPI     │              │  FastMCP     │        │
│  │  :8000       │              │  :8002       │        │
│  │  - /rag/add  │              │  - add_docs  │        │
│  │  - /rag/ret  │              │  - retrieve  │        │
│  └──────────────┘              └──────────────┘        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Application Layer                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Use Cases & Services                             │  │
│  │  - GetChunkerService (factory + caching)          │  │
│  │  - GetClientService (factory + caching)           │  │
│  │  - GetEmbeddingsService                           │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Domain Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Documents   │  │  Embeddings   │  │   Chunkers    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Infrastructure Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  ChromaDB    │  │  Weaviate     │  │  HuggingFace  │ │
│  │  Clients     │  │  Clients      │  │  Embeddings   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ragprod
   ```

2. **Sync dependencies:**
   
   For CPU-only:
   ```bash
   make sync EXTRA=cpu
   ```
   
   For CUDA support:
   ```bash
   make sync EXTRA=cu124
   ```

3. **Install the project in editable mode:**
   ```bash
   make install
   ```

### Running the FastAPI Server

```bash
# Run with default settings
uv run python -m ragprod.presentation.api.run

# Run with specific environment file
uv run python -m ragprod.presentation.api.run --env envs/api.local.env

# Run with auto-reload for development
uv run python -m ragprod.presentation.api.run --reload

# Access API documentation
# Open http://localhost:8000/docs in your browser
```

### Running the MCP Server

```bash
python -m ragprod.presentation.mcp.run
```

## 🐳 Docker Deployment

RAGProd supports multiple deployment scenarios with Docker:

### Local Development (ChromaDB)
```bash
# Start FastAPI with local ChromaDB
docker-compose --profile api up

# Start MCP with local ChromaDB
docker-compose --profile mcp up
```

### Production (External Weaviate)
```bash
# Start FastAPI with external Weaviate
docker-compose --profile api-external up

# Start MCP with external Weaviate
docker-compose --profile mcp-external up
```

### Hybrid Setup
```bash
# API → Local ChromaDB, MCP → External Weaviate
docker-compose --profile api --profile mcp-external up
```

For detailed deployment options, see [Docker Deployment Guide](docs/docker_deployment.md) and [Quick Reference](docs/docker_quick_reference.md).

## 📖 Usage Examples

### FastAPI REST API

```bash
# Add documents with chunking
curl -X POST http://localhost:8000/rag/add_documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "raw_text": "Your document text here...",
        "source": "document.txt"
      }
    ],
    "chunker_name": "recursive_character",
    "chunker_config": {
      "chunk_size": 1000,
      "chunk_overlap": 200
    }
  }'

# Retrieve documents
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search query",
    "limit": 5
  }'
```

For more examples, see [API Usage Guide](docs/api_usage.md).

### Programmatic Usage

```python
from ragprod.application.use_cases import GetClientService, GetChunkerService
from ragprod.domain.document import Document

# Initialize services
client_service = GetClientService()
chunker_service = GetChunkerService()

# Get database client
client = client_service.get("chroma", {
    "persist_directory": "./chromadb_data",
    "collection_name": "my_documents"
})

# Get chunker
chunker = chunker_service.get("recursive_character", {
    "chunk_size": 1000,
    "chunk_overlap": 200
})

# Create and chunk documents
documents = [
    Document(
        raw_text="Your long document text here...",
        source="example.txt"
    )
]

chunks = chunker.split_documents(documents)
await client.add_documents(chunks)

# Retrieve similar documents
results = await client.retrieve("search query", limit=5)
for doc in results:
    print(f"{doc.source}: {doc.raw_text}")
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
make test

# Run specific test files
uv run --extra=dev pytest tests/unit/test_get_chunker_service.py -v
uv run --extra=dev pytest tests/integration/test_api.py -v

# Run with coverage
uv run --extra=dev pytest --cov=ragprod tests/
```

**Test Coverage:**
- ✅ Unit tests for GetChunkerService
- ✅ Integration tests for FastAPI endpoints
- ✅ Integration tests for MCP tools

## 🔧 Configuration

### Environment Files

RAGProd uses environment files for configuration. Examples are provided in `envs/`:

- `api.local.env` - FastAPI with local ChromaDB
- `api.external.env` - FastAPI with external Weaviate
- `mcp.local.env` - MCP with local ChromaDB
- `mcp.external.env` - MCP with external Weaviate

### Key Environment Variables

#### Database Configuration
```env
DB_TYPE=chroma              # Options: chroma, weaviate
DB_MODE=local               # Options: local, remote

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chromadb_data
CHROMA_COLLECTION_NAME=ragprod

# Weaviate
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-api-key
```

#### Embedding Configuration
```env
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=jinaai/jina-code-embeddings-0.5b
EMBEDDING_DEVICE=cpu
EMBEDDING_DTYPE=bfloat16
```

#### FastAPI Configuration
```env
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_WORKERS=1
FASTAPI_RELOAD=True
LOG_LEVEL=INFO
```

For complete configuration options, see the environment file examples in `envs/`.

## 🛠️ Development

### Project Structure

```
ragprod/
├── src/ragprod/
│   ├── domain/              # Core business logic
│   ├── infrastructure/      # External integrations
│   │   ├── client/          # Vector DB clients
│   │   ├── chunker/         # Text splitting strategies
│   │   ├── embeddings/      # Embedding providers
│   │   └── config/          # Configuration management
│   ├── application/         # Use cases and services
│   │   └── use_cases/       # GetChunkerService, GetClientService
│   └── presentation/        # API interfaces
│       ├── api/             # FastAPI application
│       └── mcp/             # FastMCP server
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── docs/                    # Documentation
├── docker/                  # Docker configurations
├── envs/                    # Environment files
└── notebooks/              # Jupyter notebooks
```

### Available Make Commands

- `make sync EXTRA=cpu` - Sync dependencies with CPU support
- `make sync EXTRA=cu124` - Sync dependencies with CUDA support
- `make install` - Install project in editable mode
- `make test` - Run tests
- `make lint` - Run linting
- `make clean` - Clean up environment

### Adding New Features

1. **Vector Database**: Implement `BaseClient` in `infrastructure/client/`
2. **Embedding Model**: Implement `BaseEmbedding` in `infrastructure/embeddings/`
3. **Chunking Strategy**: Implement `BaseChunker` in `infrastructure/chunker/`
4. **API Endpoint**: Add route in `presentation/api/routes/`
5. **MCP Tool**: Add tool in `presentation/mcp/tools/`

## 📚 Documentation

- [API Usage Guide](docs/api_usage.md) - Complete FastAPI usage examples
- [Retrieval Strategies](docs/retrieval_strategies/) - Advanced retrieval methods (DAT, hybrid search)
- [Docker Deployment](docs/docker_deployment.md) - Comprehensive deployment guide
- [Docker Quick Reference](docs/docker_quick_reference.md) - Quick commands and diagrams
- [Versioning & Release](docs/how_to_versioning_release.md) - CI/CD and semantic versioning
- [Architecture Documentation](docs/architecture.md) - Detailed architecture overview
- [Notebooks](notebooks/) - Example notebooks and tutorials

## 🧩 Technology Stack

- **Python 3.12+** - Programming language
- **UV** - Fast Python package manager
- **FastAPI** - Modern web framework for APIs
- **FastMCP** - Model Context Protocol framework
- **Pydantic** - Data validation
- **Structlog** - Structured logging
- **ChromaDB/Weaviate** - Vector databases
- **HuggingFace Transformers** - Embedding models
- **Docker & Docker Compose** - Containerization
- **pytest** - Testing framework
- **python-semantic-release** - Automated versioning

## 🚢 CI/CD

RAGProd includes automated semantic versioning:

- **Conventional Commits** - Use `feat:`, `fix:`, `refactor!:` for automatic version bumps
- **GitHub Actions** - Automated releases on push to `main`
- **Changelog Generation** - Automatic changelog updates

See [Versioning Guide](docs/how_to_versioning_release.md) for details.

## 📖 Academic References

RAGProd implements state-of-the-art retrieval techniques based on recent research:

### Retrieval-Augmented Generation
- **RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
  Lewis et al., 2020 - [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

### Dense Retrieval
- **Dense Passage Retrieval for Open-Domain Question Answering**  
  Karpukhin et al., 2020 - [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

### Sparse Retrieval (BM25)
- **Okapi BM25**  
  Robertson & Zaragoza, 2009 - [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)

### Hybrid Search & Dynamic Weighting
- **Dynamic Alpha Tuning (DAT)**  
  Implemented based on hybrid search best practices and LLM-based effectiveness evaluation

### Advanced RAG Techniques
- **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**  
  Asai et al., 2023 - [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

- **Graph RAG: Unlocking LLM Discovery on Narrative Private Data**  
  Microsoft Research, 2024 - [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)

### Embedding Models
- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction**  
  Khattab & Zaharia, 2020 - [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**  
  Reimers & Gurevych, 2019 - [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes using conventional commits (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

[Add your license information here]

## 👤 Author

**Adrià Cabello**
- Email: cabl.adria@gmail.com

---

For more information, see the [documentation](docs/).
