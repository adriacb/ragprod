# Docker Deployment Guide

## Overview

RAGProd supports flexible deployment scenarios with both FastAPI and FastMCP presentations, connecting to either local or external vector databases.

## Supported Databases

- **ChromaDB**: Local persistent or remote HTTP
- **Weaviate**: Local, self-hosted, or Weaviate Cloud

## Deployment Scenarios

### 1. Local Development (ChromaDB)

Both API and MCP connect to a local ChromaDB instance running in Docker.

```bash
# Start API with local ChromaDB
docker-compose --profile api up

# Start MCP with local ChromaDB
docker-compose --profile mcp up

# Start both
docker-compose --profile api --profile mcp up
```

**Environment File**: `envs/api.local.env` or `envs/mcp.local.env`

```env
DB_TYPE=chroma
DB_MODE=local
CHROMA_PERSIST_DIRECTORY=./chromadb_data
CHROMA_COLLECTION_NAME=ragprod_local
```

### 2. Production (External Weaviate)

Connect to Weaviate Cloud or a self-hosted Weaviate instance.

```bash
# Start API with external Weaviate
docker-compose --profile api-external up

# Start MCP with external Weaviate
docker-compose --profile mcp-external up
```

**Environment File**: `envs/api.external.env` or `envs/mcp.external.env`

```env
DB_TYPE=weaviate
DB_MODE=remote
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-api-key-here
WEAVIATE_COLLECTION_NAME=ragprod_production
```

### 3. Hybrid Setup

API connects to local ChromaDB, MCP connects to external Weaviate.

```bash
docker-compose --profile api --profile mcp-external up
```

### 4. Local Weaviate Development

Run Weaviate locally in Docker.

```bash
# Start local Weaviate
docker-compose --profile weaviate-local up

# Update env file to point to local Weaviate
# DB_TYPE=weaviate
# DB_MODE=local
# WEAVIATE_PORT=8080
# WEAVIATE_GRPC_PORT=50051
```

## Environment Variables

### Database Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DB_TYPE` | Database type (`chroma`, `weaviate`) | `chroma` | No |
| `DB_MODE` | Connection mode (`local`, `remote`) | `local` | No |

### ChromaDB Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CHROMA_PERSIST_DIRECTORY` | Local storage directory | `./chromadb_data` | No |
| `CHROMA_COLLECTION_NAME` | Collection name | `ragprod` | No |
| `CHROMA_API_HOST` | Remote ChromaDB host | - | Yes (remote mode) |
| `CHROMA_API_PORT` | Remote ChromaDB port | `8000` | No |

### Weaviate Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `WEAVIATE_URL` | Weaviate server URL | - | Yes (remote mode) |
| `WEAVIATE_API_KEY` | API key for authentication | - | No |
| `WEAVIATE_COLLECTION_NAME` | Collection name | `ragprod` | No |
| `WEAVIATE_PORT` | Local Weaviate port | `8080` | No |
| `WEAVIATE_GRPC_PORT` | Local Weaviate gRPC port | `50051` | No |

### Embedding Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | Embedding provider | `huggingface` |
| `EMBEDDING_MODEL_NAME` | Model name | `jinaai/jina-code-embeddings-0.5b` |
| `EMBEDDING_DEVICE` | Device (`cpu`, `cuda`) | `cpu` |
| `EMBEDDING_DTYPE` | Data type | `bfloat16` |

## Docker Compose Profiles

| Profile | Description | Services Started |
|---------|-------------|------------------|
| `api` | FastAPI with local DB | `api`, `chromadb` |
| `api-external` | FastAPI with external DB | `api-external` |
| `mcp` | FastMCP with local DB | `mcp`, `chromadb` |
| `mcp-external` | FastMCP with external DB | `mcp-external` |
| `weaviate-local` | Local Weaviate server | `weaviate` |
| `all` | All services with local DBs | `api`, `mcp`, `chromadb`, `weaviate` |

## Port Mapping

| Service | Container Port | Host Port | Configurable |
|---------|---------------|-----------|--------------|
| FastAPI | 8000 | 8000 | `API_PORT` |
| FastMCP | 8000 | 8002 | `MCP_PORT` |
| ChromaDB | 8000 | 8001 | No |
| Weaviate HTTP | 8080 | 8080 | No |
| Weaviate gRPC | 50051 | 50051 | No |

## Examples

### Example 1: Local Development

```bash
# Create local env file (already provided)
cat envs/api.local.env

# Start services
docker-compose --profile api up -d

# Check logs
docker-compose logs -f api

# Access API
curl http://localhost:8000/health
```

### Example 2: Production with Weaviate Cloud

1. **Create Weaviate Cloud cluster** at https://console.weaviate.cloud

2. **Update environment file**:
```bash
# envs/api.production.env
DB_TYPE=weaviate
DB_MODE=remote
WEAVIATE_URL=https://my-cluster-abc123.weaviate.network
WEAVIATE_API_KEY=my-secret-key
WEAVIATE_COLLECTION_NAME=ragprod_prod
LOG_LEVEL=WARNING
FASTAPI_WORKERS=4
FASTAPI_RELOAD=False
```

3. **Deploy**:
```bash
API_ENV_FILE=envs/api.production.env docker-compose --profile api-external up -d
```

### Example 3: Mixed Environment

API uses local ChromaDB, MCP uses external Weaviate:

```bash
# Start API with local DB
API_ENV_FILE=envs/api.local.env docker-compose --profile api up -d

# Start MCP with external DB
MCP_ENV_FILE=envs/mcp.external.env docker-compose --profile mcp-external up -d
```

### Example 4: Custom Configuration

```bash
# Create custom env file
cp envs/api.local.env envs/api.custom.env

# Edit as needed
nano envs/api.custom.env

# Run with custom config
API_ENV_FILE=envs/api.custom.env docker-compose --profile api up
```

## Building Images

```bash
# Build with CPU support
docker-compose build --build-arg EXTRA=cpu

# Build with CUDA support
docker-compose build --build-arg EXTRA=cu124

# Build with ROCm support
docker-compose build --build-arg EXTRA=rocm
```

## Troubleshooting

### Database Connection Issues

**Problem**: Cannot connect to database

**Solutions**:
1. Check environment variables are set correctly
2. Verify database service is running: `docker-compose ps`
3. Check logs: `docker-compose logs chromadb` or `docker-compose logs weaviate`
4. Ensure network connectivity: `docker network inspect ragprod_network`

### Port Conflicts

**Problem**: Port already in use

**Solutions**:
```bash
# Change API port
API_PORT=8080 docker-compose --profile api up

# Change MCP port
MCP_PORT=8003 docker-compose --profile mcp up
```

### Volume Permissions

**Problem**: Permission denied accessing volumes

**Solutions**:
```bash
# Fix permissions
sudo chown -R $USER:$USER chromadb_data/

# Or use Docker volumes instead of bind mounts
```

### External Database Not Reachable

**Problem**: Cannot connect to external Weaviate/ChromaDB

**Solutions**:
1. Verify URL and credentials in env file
2. Check firewall rules
3. Test connection manually:
```bash
# Test Weaviate
curl https://your-cluster.weaviate.network/v1/meta

# Test ChromaDB
curl http://your-chromadb-host:8000/api/v1/heartbeat
```

## Best Practices

1. **Use different collections** for dev/staging/prod
2. **Never commit** `.env` files with real credentials
3. **Use Docker secrets** for production credentials
4. **Monitor resource usage** with `docker stats`
5. **Backup data volumes** regularly
6. **Use health checks** to ensure services are ready
7. **Set appropriate log levels** (DEBUG for dev, WARNING for prod)

## Data Persistence

### Local ChromaDB
Data is stored in Docker volume `chromadb_data`:
```bash
# Backup
docker run --rm -v ragprod_chromadb_data:/data -v $(pwd):/backup alpine tar czf /backup/chromadb-backup.tar.gz /data

# Restore
docker run --rm -v ragprod_chromadb_data:/data -v $(pwd):/backup alpine tar xzf /backup/chromadb-backup.tar.gz -C /
```

### External Databases
Follow the backup procedures for your external database service.

## Monitoring

```bash
# View logs
docker-compose logs -f api
docker-compose logs -f mcp

# Monitor resources
docker stats

# Check health
curl http://localhost:8000/health
```

## Scaling

### Horizontal Scaling

```bash
# Scale API service
docker-compose --profile api up -d --scale api=3

# Use a load balancer (nginx, traefik) in front
```

### Vertical Scaling

Update `docker-compose.yaml` to add resource limits:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```
