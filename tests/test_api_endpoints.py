def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store_chunks" in data


def test_document_ingest_and_query_flow(client):
    # 1. Ingest
    ingest_payload = {
        "doc_id": "api_test_doc",
        "title": "API Microservices Guide",
        "content": "FastAPI provides asynchronous concurrency and automatic OpenAPI schema generation."
    }
    ingest_resp = client.post("/api/v1/documents/ingest", json=ingest_payload)
    assert ingest_resp.status_code == 201
    assert ingest_resp.json()["status"] == "success"

    # 2. Query
    query_payload = {
        "query": "How does FastAPI handle concurrency and OpenAPI?",
        "top_k": 3
    }
    query_resp = client.post("/api/v1/query", json=query_payload)
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert "FastAPI" in data["answer"] or len(data["citations"]) > 0
    assert data["faithfulness_score"] > 0


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "average_latency_ms" in data
    assert "uptime_seconds" in data
