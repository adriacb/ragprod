import sys
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

# Mock fastmcp before any imports that might use it
sys.modules['fastmcp'] = MagicMock()

from ragprod.presentation.api.app import app
from ragprod.presentation.api.lifespan.service import init_services
from ragprod.domain.document import Document


@pytest.fixture(scope="module", autouse=True)
def setup_services():
    """Initialize services before running tests."""
    # Initialize without env file for testing
    init_services(env_path=None)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_client_db():
    """Mock the database client."""
    with patch("ragprod.presentation.api.routes.rag.clientDB") as mock_db:
        mock_db.add_documents = AsyncMock()
        mock_db.retrieve = AsyncMock()
        yield mock_db


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_add_documents_default_config(client, mock_client_db):
    """Test adding documents with default chunker configuration."""
    mock_client_db.add_documents.return_value = None
    
    request_data = {
        "documents": [
            {
                "raw_text": "This is a test document. " * 50,
                "source": "test.txt"
            }
        ]
    }
    
    response = client.post("/rag/add_documents", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "chunks_created" in data
    assert data["chunks_created"] >= 1
    
    # Verify database was called
    assert mock_client_db.add_documents.called


def test_add_documents_custom_config(client, mock_client_db):
    """Test adding documents with custom chunker configuration."""
    mock_client_db.add_documents.return_value = None
    
    request_data = {
        "documents": [
            {
                "raw_text": "This is a test document. " * 50,
                "source": "test.txt"
            }
        ],
        "chunker_name": "recursive_character",
        "chunker_config": {
            "chunk_size": 100,
            "chunk_overlap": 20
        }
    }
    
    response = client.post("/rag/add_documents", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_created"] > 10  # Small chunks should create many pieces


def test_add_documents_multiple(client, mock_client_db):
    """Test adding multiple documents."""
    mock_client_db.add_documents.return_value = None
    
    request_data = {
        "documents": [
            {
                "raw_text": "First document text.",
                "source": "doc1.txt"
            },
            {
                "raw_text": "Second document text.",
                "source": "doc2.txt"
            }
        ]
    }
    
    response = client.post("/rag/add_documents", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_created"] >= 2


def test_add_documents_invalid_chunker(client, mock_client_db):
    """Test adding documents with invalid chunker name."""
    request_data = {
        "documents": [
            {
                "raw_text": "Test document",
                "source": "test.txt"
            }
        ],
        "chunker_name": "invalid_chunker"
    }
    
    response = client.post("/rag/add_documents", json=request_data)
    
    assert response.status_code == 400
    assert "detail" in response.json()


def test_add_documents_token_chunker(client, mock_client_db):
    """Test adding documents with token chunker."""
    mock_client_db.add_documents.return_value = None
    
    request_data = {
        "documents": [
            {
                "raw_text": "This is a test document for token chunking.",
                "source": "test.txt"
            }
        ],
        "chunker_name": "token",
        "chunker_config": {
            "chunk_size": 10,
            "chunk_overlap": 0
        }
    }
    
    response = client.post("/rag/add_documents", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_created"] >= 1


def test_retrieve_documents(client, mock_client_db):
    """Test retrieving documents."""
    # Mock the retrieve response
    mock_results = [
        Document(raw_text="Result 1", source="doc1.txt"),
        Document(raw_text="Result 2", source="doc2.txt")
    ]
    mock_client_db.retrieve.return_value = mock_results
    
    request_data = {
        "query": "test query",
        "limit": 5
    }
    
    response = client.post("/rag/retrieve", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "count" in data
    assert data["query"] == "test query"
    assert data["count"] == 2
    
    # Verify database was called with correct parameters
    mock_client_db.retrieve.assert_called_once_with("test query", 5)


def test_retrieve_documents_custom_limit(client, mock_client_db):
    """Test retrieving documents with custom limit."""
    mock_client_db.retrieve.return_value = []
    
    request_data = {
        "query": "search term",
        "limit": 10
    }
    
    response = client.post("/rag/retrieve", json=request_data)
    
    assert response.status_code == 200
    mock_client_db.retrieve.assert_called_once_with("search term", 10)


def test_add_documents_db_not_initialized(client):
    """Test adding documents when database is not initialized."""
    with patch("ragprod.presentation.api.routes.rag.clientDB", None):
        request_data = {
            "documents": [
                {
                    "raw_text": "Test",
                    "source": "test.txt"
                }
            ]
        }
        
        response = client.post("/rag/add_documents", json=request_data)
        
        assert response.status_code == 500
        assert "Database client not initialized" in response.json()["detail"]


def test_retrieve_db_not_initialized(client):
    """Test retrieving documents when database is not initialized."""
    with patch("ragprod.presentation.api.routes.rag.clientDB", None):
        request_data = {
            "query": "test",
            "limit": 5
        }
        
        response = client.post("/rag/retrieve", json=request_data)
        
        assert response.status_code == 500
        assert "Database client not initialized" in response.json()["detail"]
