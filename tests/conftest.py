import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.vector_store import vector_store
from app.services.hybrid_search import hybrid_search_engine
from app.services.document_processor import document_processor


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def setup_test_index():
    # Seed a clean test knowledge document before tests
    test_content = """
# Test Knowledge Base

## Vector Database Scaling
Vector databases index high-dimensional embeddings using algorithms like HNSW (Hierarchical Navigable Small World).
HNSW offers logarithmic search complexity O(log N) compared to exhaustive flat scanning.

## Cache Invalidation
The Redis cache uses LRU (Least Recently Used) eviction and TTL-based invalidation.
"""
    chunks = document_processor.process_document(
        doc_id="test_doc_01",
        title="Test Knowledge Base",
        content=test_content
    )
    vector_store.add_chunks(chunks)
    hybrid_search_engine.sync_bm25()
