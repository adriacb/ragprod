# RAGProd API Documentation

## Overview

The RAGProd API provides endpoints for document chunking and retrieval using various text splitting strategies. The API is built with FastAPI and includes automatic document chunking before storage.

## Getting Started

### Running the Server

#### Basic Usage
```bash
# Run with default settings (localhost:8000)
uv run python -m ragprod.presentation.api.run
```

#### With Environment File
```bash
# Load settings from a specific .env file
uv run python -m ragprod.presentation.api.run --env envs/api.env.example
```

#### Development Mode
```bash
# Enable auto-reload for development
uv run python -m ragprod.presentation.api.run --reload
```

#### Custom Host and Port
```bash
# Run on a specific host and port
uv run python -m ragprod.presentation.api.run --host 127.0.0.1 --port 8080
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Path to .env file | None (uses process environment) |
| `--host` | Host to bind to | 0.0.0.0 |
| `--port` | Port to bind to | 8000 |
| `--reload` | Enable auto-reload | False |

## API Endpoints

### Interactive Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Health Check

#### GET /health

Check if the API is running.

**Response:**
```json
{
  "status": "healthy"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### Add Documents

#### POST /rag/add_documents

Add documents to the RAG database with automatic chunking.

**Request Body:**
```json
{
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
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `documents` | array | Yes | - | List of documents to add |
| `documents[].raw_text` | string | Yes | - | The text content |
| `documents[].source` | string | Yes | - | Source identifier |
| `chunker_name` | string | No | `"recursive_character"` | Chunker type to use |
| `chunker_config` | object | No | `{"chunk_size": 1000, "chunk_overlap": 200}` | Chunker configuration |

**Supported Chunkers:**

- `recursive_character` - Splits recursively by separators (recommended)
- `character` - Simple character-based splitting
- `token` - Token-based splitting
- `markdown` - Markdown-aware splitting
- `semantic` - Semantic similarity-based splitting (requires embedding model)

**Response:**
```json
{
  "message": "Successfully added 5 chunks from 1 documents",
  "chunks_created": 5
}
```

**Example - Basic Usage:**
```bash
curl -X POST http://localhost:8000/rag/add_documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "raw_text": "This is a sample document that will be chunked automatically.",
        "source": "sample.txt"
      }
    ]
  }'
```

**Example - Custom Chunking:**
```bash
curl -X POST http://localhost:8000/rag/add_documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "raw_text": "Long document text...",
        "source": "long_doc.txt"
      }
    ],
    "chunker_name": "recursive_character",
    "chunker_config": {
      "chunk_size": 500,
      "chunk_overlap": 50
    }
  }'
```

**Example - Multiple Documents:**
```bash
curl -X POST http://localhost:8000/rag/add_documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "raw_text": "First document...",
        "source": "doc1.txt"
      },
      {
        "raw_text": "Second document...",
        "source": "doc2.txt"
      }
    ],
    "chunker_name": "token",
    "chunker_config": {
      "chunk_size": 100,
      "chunk_overlap": 10
    }
  }'
```

### Retrieve Documents

#### POST /rag/retrieve

Retrieve documents from the RAG database based on a query.

**Request Body:**
```json
{
  "query": "What is the meaning of life?",
  "limit": 5
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `limit` | integer | No | 5 | Maximum number of results |

**Response:**
```json
{
  "query": "What is the meaning of life?",
  "results": [
    {
      "raw_text": "Chunk of text...",
      "source": "document.txt",
      "metadata": {}
    }
  ],
  "count": 1
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "limit": 10
  }'
```

## Chunker Configuration

### Recursive Character (Recommended)

Best for general text. Splits recursively by separators.

```json
{
  "chunker_name": "recursive_character",
  "chunker_config": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

### Character

Simple character-based splitting.

```json
{
  "chunker_name": "character",
  "chunker_config": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separator": "\n\n"
  }
}
```

### Token

Token-based splitting (useful for LLM context limits).

```json
{
  "chunker_name": "token",
  "chunker_config": {
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

### Markdown

Preserves markdown structure.

```json
{
  "chunker_name": "markdown",
  "chunker_config": {
    "headers_to_split_on": [
      ["#", "Header 1"],
      ["##", "Header 2"]
    ]
  }
}
```

### Semantic

Splits based on semantic similarity (requires embedding model).

```json
{
  "chunker_name": "semantic",
  "chunker_config": {
    "embedding_model": "<embedding_model_instance>",
    "buffer_size": 1
  }
}
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid chunker configuration)
- `500` - Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Environment Configuration

Create a `.env` file in the `envs/` directory to configure the API:

```bash
# Example: envs/api.env
LOG_LEVEL=INFO
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
```

Load it when starting the server:
```bash
uv run python -m ragprod.presentation.api.run --env envs/api.env
```

## Python Client Example

```python
import requests

# Add documents
response = requests.post(
    "http://localhost:8000/rag/add_documents",
    json={
        "documents": [
            {
                "raw_text": "Your document text here",
                "source": "example.txt"
            }
        ],
        "chunker_name": "recursive_character",
        "chunker_config": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    }
)
print(response.json())

# Retrieve documents
response = requests.post(
    "http://localhost:8000/rag/retrieve",
    json={
        "query": "search query",
        "limit": 5
    }
)
print(response.json())
```

## Best Practices

1. **Chunk Size**: Start with 1000 characters and adjust based on your use case
2. **Overlap**: Use 10-20% of chunk size for overlap to maintain context
3. **Chunker Selection**: Use `recursive_character` for most text documents
4. **Error Handling**: Always check response status codes and handle errors
5. **Environment Files**: Use different `.env` files for dev/staging/production

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Verify the env file path is correct
- Check logs for initialization errors

### Documents not being added
- Verify the database client is initialized correctly
- Check the chunker configuration is valid
- Review server logs for detailed error messages

### Retrieval returns no results
- Ensure documents have been added successfully
- Check the query matches document content
- Verify the database is properly configured
