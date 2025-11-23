# RAGProd Docker Architecture

## Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│                     Deployment Scenarios                     │
└─────────────────────────────────────────────────────────────┘

Scenario 1: Local Development
┌──────────┐     ┌──────────┐     ┌─────────────┐
│ FastAPI  │────▶│ ChromaDB │     │  Weaviate   │
│  :8000   │     │  :8001   │     │  (unused)   │
└──────────┘     └──────────┘     └─────────────┘
┌──────────┐           │
│ FastMCP  │───────────┘
│  :8002   │
└──────────┘

Scenario 2: Production (External DB)
┌──────────┐     
│ FastAPI  │────────────────────┐
│  :8000   │                    │
└──────────┘                    ▼
┌──────────┐           ┌──────────────────┐
│ FastMCP  │──────────▶│ Weaviate Cloud   │
│  :8002   │           │ (External)       │
└──────────┘           └──────────────────┘

Scenario 3: Hybrid
┌──────────┐     ┌──────────┐
│ FastAPI  │────▶│ ChromaDB │
│  :8000   │     │  :8001   │
└──────────┘     └──────────┘
┌──────────┐     ┌──────────────────┐
│ FastMCP  │────▶│ Weaviate Cloud   │
│  :8002   │     │ (External)       │
└──────────┘     └──────────────────┘

Scenario 4: All Local
┌──────────┐     ┌──────────┐
│ FastAPI  │────▶│ ChromaDB │
│  :8000   │     │  :8001   │
└──────────┘     └──────────┘
┌──────────┐     ┌──────────┐
│ FastMCP  │────▶│ Weaviate │
│  :8002   │     │  :8080   │
└──────────┘     └──────────┘
```

## Commands Cheat Sheet

### Start Services

```bash
# Local development (API + ChromaDB)
docker-compose --profile api up

# Production (API + External DB)
docker-compose --profile api-external up

# Both API and MCP with local DB
docker-compose --profile api --profile mcp up

# Everything local
docker-compose --profile all up
```

### Environment Files

```bash
# API with local ChromaDB
envs/api.local.env

# API with external Weaviate
envs/api.external.env

# MCP with local ChromaDB
envs/mcp.local.env

# MCP with external Weaviate
envs/mcp.external.env
```

### Custom Configuration

```bash
# Use custom env file
API_ENV_FILE=envs/my-custom.env docker-compose --profile api up

# Change ports
API_PORT=8080 MCP_PORT=8003 docker-compose --profile api --profile mcp up
```

## Environment Variable Quick Reference

### Required for Each Database Type

**ChromaDB Local:**
```env
DB_TYPE=chroma
DB_MODE=local
CHROMA_PERSIST_DIRECTORY=./chromadb_data
CHROMA_COLLECTION_NAME=my_collection
```

**ChromaDB Remote:**
```env
DB_TYPE=chroma
DB_MODE=remote
CHROMA_API_HOST=chromadb.example.com
CHROMA_API_PORT=8000
CHROMA_COLLECTION_NAME=my_collection
```

**Weaviate Local:**
```env
DB_TYPE=weaviate
DB_MODE=local
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION_NAME=my_collection
```

**Weaviate Cloud:**
```env
DB_TYPE=weaviate
DB_MODE=remote
WEAVIATE_URL=https://cluster.weaviate.network
WEAVIATE_API_KEY=your-key-here
WEAVIATE_COLLECTION_NAME=my_collection
```
