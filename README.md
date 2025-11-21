# RAGProd

**RAGProd** is a production-ready **Retrieval-Augmented Generation (RAG)** system built with clean architecture principles. It provides a modular, extensible framework for building RAG applications with support for multiple vector databases, embedding models, and monitoring tools.

## 🎯 What is RAGProd?

RAGProd is a comprehensive RAG framework that enables you to:

- **Ingest and store documents** in vector databases with automatic chunking and embedding
- **Retrieve relevant documents** using semantic search across multiple vector database backends
- **Support multiple embedding models** (HuggingFace, OpenAI, ColBERT)
- **Monitor and evaluate** RAG performance with integrated observability tools
- **Expose RAG capabilities** via Model Context Protocol (MCP) for easy integration with AI applications

The system follows clean architecture patterns, making it easy to extend, test, and maintain while supporting production deployments.

## ✨ Key Features

### 🗄️ Multi-Vector Database Support
- **ChromaDB** - Local persistent storage or remote API
- **FAISS** - High-performance similarity search
- **Qdrant** - Production-ready vector database
- **Weaviate** - Cloud-native vector database

### 🤖 Multiple Embedding Models
- **HuggingFace Transformers** - Open-source embedding models
- **OpenAI Embeddings** - GPT-based embeddings
- **ColBERT** - Contextualized late interaction embeddings

### 📊 Monitoring & Observability
- **Langfuse** - LLM observability and analytics
- **LangSmith** - LangChain monitoring
- **Phoenix** - Open-source LLM observability

### 🏗️ Clean Architecture
- **Domain Layer** - Core business logic and entities
- **Infrastructure Layer** - External service integrations
- **Application Layer** - Use cases and services
- **Presentation Layer** - MCP server and API interfaces

### 🔧 Production-Ready Features
- Structured logging with `structlog`
- Configuration management with environment variables
- Async/await support throughout
- Comprehensive unit tests
- Docker support for containerized deployments

## 🏛️ Architecture Overview

RAGProd follows a clean architecture pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│              Presentation Layer (MCP/API)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  MCP Server (FastMCP)                              │  │
│  │  - rag_retrieve()                                  │  │
│  │  - add_documents()                                 │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Application Layer                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Use Cases & Services                             │  │
│  │  - get_client_service()                           │  │
│  │  - get_embeddings()                               │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Domain Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Documents   │  │  Embeddings   │  │   Chunkers    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Evaluators  │  │   Encoders    │  │   Prompts     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Infrastructure Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Vector DBs  │  │  Embeddings   │  │  Monitoring   │ │
│  │  Clients     │  │  Providers    │  │  Tools        │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) package manager

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

### Configuration

1. **Copy environment file templates:**
   ```bash
   cp envs/api.env.example envs/api.env
   cp envs/langfuse.env.example envs/langfuse.env
   ```

2. **Configure your environment variables:**
   - Edit `envs/api.env` for API settings
   - Edit `envs/langfuse.env` for monitoring (optional)

### Running the MCP Server

Start the MCP server to expose RAG capabilities:

```bash
python -m ragprod.presentation.mcp.run
```

The server exposes tools for:
- `rag_retrieve(query, limit)` - Retrieve documents by semantic similarity
- `add_documents(documents)` - Add documents to the vector database

## 📖 Usage Examples

### Using the MCP Server

The MCP server can be integrated with AI applications that support the Model Context Protocol. It provides RAG retrieval capabilities as tools that can be called by LLMs.

### Programmatic Usage

```python
from ragprod.application.use_cases import get_client_service, get_embeddings
from ragprod.domain.document import Document

# Get a vector database client
client = await get_client_service(
    mode="local_persistent",
    collection_name="my_documents"
)

# Get an embedding model
embedder = get_embeddings(model_type="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and add documents
documents = [
    Document(
        raw_text="Your document text here",
        source="example",
        title="Example Document"
    )
]

await client.add_documents(documents)

# Retrieve similar documents
results = await client.retrieve("search query", limit=5)
for doc in results:
    print(f"{doc.title}: {doc.content}")
```

## 🧪 Testing

Run the test suite:

```bash
make test
```

Or run specific test files:

```bash
uv run --extra dev pytest tests/unit/test_logger.py -v
uv run --extra dev pytest tests/unit/test_config.py -v
```

## 🛠️ Development

### Project Structure

```
ragprod/
├── src/ragprod/
│   ├── domain/              # Core business logic
│   ├── infrastructure/      # External integrations
│   ├── application/         # Use cases and services
│   └── presentation/        # MCP server and API
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── notebooks/              # Jupyter notebooks
└── envs/                   # Environment configuration
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
4. **MCP Tool**: Add new tool in `presentation/mcp/tools/`

## 🔧 Configuration

### Environment Variables

#### API Configuration (`envs/api.env`)
- `FASTAPI_HOST` - API host (default: `0.0.0.0`)
- `FASTAPI_PORT` - API port (default: `8000`)
- `FASTAPI_WORKERS` - Number of workers (default: `1`)
- `FASTAPI_RELOAD` - Enable auto-reload (default: `True`)

#### Langfuse Configuration (`envs/langfuse.env`)
- `LANGFUSE_PUBLIC_KEY` - Langfuse public key (required)
- `LANGFUSE_SECRET_KEY` - Langfuse secret key (required)
- `LANGFUSE_HOST` - Langfuse host (default: `https://cloud.langfuse.com`)
- `LANGFUSE_TAGS` - Comma-separated tags or JSON array

#### Logging
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

## 📚 Documentation

- [Architecture Documentation](docs/architecture.md) - Detailed architecture overview
- [Notebooks](notebooks/) - Example notebooks and tutorials

## 🧩 Technology Stack

- **Python 3.12+** - Programming language
- **UV** - Fast Python package manager
- **Pydantic** - Data validation
- **Structlog** - Structured logging
- **FastMCP** - Model Context Protocol framework
- **ChromaDB/FAISS/Qdrant/Weaviate** - Vector databases
- **HuggingFace Transformers** - Embedding models
- **OpenAI** - GPT embeddings
- **Langfuse/LangSmith/Phoenix** - Monitoring tools

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license information here]

## 👤 Author

**Adrià Cabello**
- Email: cabl.adria@gmail.com

---

For more information, see the [architecture documentation](docs/architecture.md).
