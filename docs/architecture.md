# RAGProd Architecture

## Overview

RAGProd is a Retrieval-Augmented Generation (RAG) production system built with a clean architecture pattern. It provides a modular framework for document embedding, storage, and retrieval using vector databases.

## Directory Structure

```
ragprod/
├── src/
│   └── ragprod/
│       ├── __init__.py
│       │
│       ├── domain/                          # Core business logic
│       │   ├── chunk/                       # Document chunking strategies
│       │   │   ├── base.py                  # Base chunker interface
│       │   │   ├── chunker.py               # Chunker implementation
│       │   │   ├── semantic_chunking.py     # Semantic chunking
│       │   │   └── splitters/               # Text splitter utilities
│       │   │
│       │   ├── client/                      # Vector database client abstractions
│       │   │   ├── base.py                  # BaseClient interface
│       │   │   └── __init__.py
│       │   │
│       │   ├── document/                    # Document models
│       │   │   ├── base.py                  # BaseDocument interface
│       │   │   └── document.py              # Document implementation
│       │   │
│       │   ├── embedding/                   # Embedding model abstractions
│       │   │   ├── base.py                  # EmbeddingModel interface
│       │   │   ├── huggingface_embedding.py # HuggingFace implementation
│       │   │   ├── openai_embedding.py      # OpenAI implementation
│       │   │   ├── colbert_embeddings.py    # ColBERT implementation
│       │   │   └── quantization.py          # Quantization utilities
│       │   │
│       │   ├── encoder/                     # Encoder implementations
│       │   │   ├── bie.py                   # Bi-encoder
│       │   │   ├── colbert.py               # ColBERT encoder
│       │   │   └── cross.py                 # Cross-encoder
│       │   │
│       │   ├── evaluator/                   # Evaluation tools
│       │   │   ├── human_annot_eval.py      # Human annotation evaluation
│       │   │   ├── llm_as_judge_eval.py     # LLM-as-judge evaluation
│       │   │   ├── llm_performance_eval.py  # LLM performance evaluation
│       │   │   └── retrieval_eval.py        # Retrieval evaluation
│       │   │
│       │   └── prompt/                      # Prompt optimization
│       │       └── prompt_optimizer.py
│       │
│       ├── infrastructure/                  # External service integrations
│       │   ├── client/                      # Vector database clients
│       │   │   ├── chromadb.py              # ChromaDB async client
│       │   │   ├── faiss.py                 # FAISS client
│       │   │   ├── qdrant.py                # Qdrant client
│       │   │   ├── weaviate.py              # Weaviate client
│       │   │   └── __init__.py
│       │   │
│       │   └── monitoring/                  # Observability tools
│       │       ├── base.py                  # Base monitoring interface
│       │       ├── langfuse.py              # Langfuse integration
│       │       ├── langsmith.py             # LangSmith integration
│       │       └── phoenix.py               # Phoenix integration
│       │
│       ├── application/                     # Application services
│       │   ├── service.py                   # Application services
│       │   └── use_cases/                   # Use case implementations
│       │
│       └── presentation/                    # External interfaces
│           ├── api/                         # REST API (placeholder)
│           └── mcp/                         # Model Context Protocol server
│               ├── server.py                # FastMCP server instance
│               ├── client.py                # Database client initialization
│               ├── run.py                   # Server entry point
│               ├── add_docs.py              # Document addition utilities
│               ├── test_client.py           # Test client for MCP
│               └── tools/                   # MCP tools
│                   ├── rag.py               # RAG retrieval tool
│                   └── __init__.py
│
├── docs/                                    # Documentation
│   └── architecture.md                      # This file
│
├── notebooks/                               # Jupyter notebooks
│   ├── 01intro/                             # Introduction notebooks
│   ├── 03weviate/                           # Weaviate examples
│   ├── llm/                                 # LLM examples
│   ├── monitor/                             # Monitoring examples
│   └── *.ipynb                              # Various test notebooks
│
├── notes/                                   # Development notes
│   └── intro.md
│
├── chromadb_test/                           # ChromaDB test data
│
├── pyproject.toml                           # Project configuration
├── uv.lock                                  # Dependency lock file
├── Dockerfile                               # Docker configuration
├── docker-compose.yaml                      # Docker Compose configuration
├── Makefile                                 # Build automation
└── README.md                                # Project README
```

## Architecture Layers

### 1. Domain Layer (`domain/`)
Core business logic and entities. Contains:
- **Documents**: Document models with metadata
- **Embeddings**: Embedding model interfaces and implementations
- **Chunking**: Document chunking strategies
- **Evaluators**: Evaluation tools for RAG systems
- **Encoders**: Different encoder types (bi-encoder, cross-encoder, ColBERT)

### 2. Infrastructure Layer (`infrastructure/`)
External service integrations:
- **Vector Databases**: Clients for ChromaDB, FAISS, Qdrant, Weaviate
- **Monitoring**: Observability tools (Langfuse, LangSmith, Phoenix)

### 3. Application Layer (`application/`)
Application services and use cases (currently minimal, ready for expansion)

### 4. Presentation Layer (`presentation/`)
External interfaces:
- **MCP Server**: Model Context Protocol server exposing RAG tools
- **API**: REST API (placeholder for future development)

## System Flow

```mermaid
graph TB
    subgraph "Presentation Layer"
        MCP[MCP Server<br/>FastMCP]
        Tools[MCP Tools<br/>rag_retrieve]
    end
    
    subgraph "Application Layer"
        Service[Application Services]
        UseCases[Use Cases]
    end
    
    subgraph "Domain Layer"
        Document[Document Model]
        Embedding[Embedding Model<br/>HuggingFace/OpenAI]
        Chunker[Chunker<br/>Fixed/Semantic]
        Evaluator[Evaluators]
    end
    
    subgraph "Infrastructure Layer"
        ChromaDB[ChromaDB Client<br/>AsyncChromaDBClient]
        FAISS[FAISS Client]
        Qdrant[Qdrant Client]
        Weaviate[Weaviate Client]
        Monitor[Monitoring<br/>Langfuse/LangSmith]
    end
    
    subgraph "External Services"
        VectorDB[(Vector Database<br/>ChromaDB/FAISS/Qdrant)]
        EmbedModel[Embedding Models<br/>HuggingFace/OpenAI]
    end
    
    MCP --> Tools
    Tools --> ChromaDB
    Service --> UseCases
    UseCases --> Document
    UseCases --> Embedding
    UseCases --> Chunker
    
    ChromaDB --> Embedding
    ChromaDB --> VectorDB
    FAISS --> Embedding
    FAISS --> VectorDB
    Qdrant --> Embedding
    Qdrant --> VectorDB
    Weaviate --> Embedding
    Weaviate --> VectorDB
    
    Embedding --> EmbedModel
    ChromaDB --> Monitor
    FAISS --> Monitor
    Qdrant --> Monitor
    Weaviate --> Monitor
    
    style MCP fill:#e1f5ff
    style Tools fill:#e1f5ff
    style Document fill:#fff4e1
    style Embedding fill:#fff4e1
    style ChromaDB fill:#e8f5e9
    style VectorDB fill:#f3e5f5
    style EmbedModel fill:#f3e5f5
```

## Component Interaction Flow

### Document Ingestion Flow

```mermaid
sequenceDiagram
    participant User
    participant MCP as MCP Server
    participant Client as Vector DB Client
    participant Embedder as Embedding Model
    participant Chunker as Chunker
    participant VDB as Vector Database
    
    User->>MCP: Add documents
    MCP->>Chunker: Chunk documents
    Chunker->>MCP: Return chunks
    MCP->>Client: add_documents(chunks)
    Client->>Embedder: embed_documents(texts)
    Embedder-->>Client: embeddings
    Client->>VDB: Store documents + embeddings
    VDB-->>Client: Success
    Client-->>MCP: Confirmation
    MCP-->>User: Documents added
```

### Query Retrieval Flow

```mermaid
sequenceDiagram
    participant User
    participant MCP as MCP Server
    participant Tool as RAG Tool
    participant Client as Vector DB Client
    participant Embedder as Embedding Model
    participant VDB as Vector Database
    
    User->>MCP: Query request
    MCP->>Tool: rag_retrieve(query, limit)
    Tool->>Client: retrieve(query, k)
    Client->>Embedder: embed_query(query)
    Embedder-->>Client: query_embedding
    Client->>VDB: Query similar vectors
    VDB-->>Client: Retrieved documents
    Client-->>Tool: Documents with metadata
    Tool-->>MCP: Results
    MCP-->>User: Retrieved documents
```

## Key Design Patterns

1. **Clean Architecture**: Separation of concerns across layers
2. **Dependency Inversion**: Domain layer doesn't depend on infrastructure
3. **Strategy Pattern**: Multiple embedding models and vector databases
4. **Factory Pattern**: Client factory functions for different database modes
5. **Async/Await**: Asynchronous operations throughout the stack

## Technology Stack

- **Language**: Python 3.11
- **Vector Databases**: ChromaDB, FAISS, Qdrant, Weaviate
- **Embedding Models**: HuggingFace Transformers, OpenAI
- **MCP Framework**: FastMCP
- **Monitoring**: Langfuse, LangSmith, Phoenix
- **Package Management**: UV (with PyTorch variants: CPU, CUDA, ROCm)

## Configuration

The system supports multiple deployment modes:
- **Local Persistent**: ChromaDB with local file storage
- **Remote API**: ChromaDB via HTTP API
- **In-Memory**: Transient storage for testing
- **Multiple Vector DBs**: Swappable backends (ChromaDB, FAISS, Qdrant, Weaviate)

