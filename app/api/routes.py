import asyncio
import json
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSummary,
    SystemMetricsResponse,
)
from app.services.agent_workflow import agentic_rag_workflow
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store
from app.services.hybrid_search import hybrid_search_engine
from app.services.metrics_service import metrics_service
from app.core.logging import logger

router = APIRouter(prefix="/api/v1", tags=["Nexus AI & RAG Backend"])


@router.post("/query", response_model=QueryResponse, summary="Execute Agentic RAG Query")
async def execute_query(req: QueryRequest):
    """
    Query the enterprise knowledge base using the Corrective Agentic RAG pipeline:
    Dense + BM25 Hybrid Retrieval, Reciprocal Rank Fusion, and Hallucination Verification.
    """
    try:
        response = await agentic_rag_workflow.execute(
            query=req.query,
            top_k=req.top_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
            enable_self_correction=req.enable_self_correction,
        )
        return response
    except Exception as e:
        logger.error(f"Error during query execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream", summary="Stream RAG Response via Server-Sent Events")
async def stream_query(req: QueryRequest):
    """
    Asynchronous token streaming via Server-Sent Events (SSE).
    """
    async def event_generator():
        yield f"event: status\ndata: {json.dumps({'message': 'Analyzing query & retrieving context...'})}\n\n"
        await asyncio.sleep(0.05)

        response = await agentic_rag_workflow.execute(
            query=req.query,
            top_k=req.top_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
            enable_self_correction=req.enable_self_correction,
        )

        words = response.answer.split(" ")
        for word in words:
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.02)

        yield f"event: citations\ndata: {json.dumps([c.model_dump() for c in response.citations])}\n\n"
        yield f"event: done\ndata: {json.dumps({'latency_ms': response.latency_ms, 'faithfulness': response.faithfulness_score})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/documents/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(req: DocumentIngestRequest):
    """
    Ingest and index a text/markdown document into dense vector store and sparse BM25 inverted index.
    """
    try:
        chunks = document_processor.process_document(
            doc_id=req.doc_id,
            title=req.title,
            content=req.content,
            metadata=req.metadata,
        )
        vector_store.add_chunks(chunks)
        hybrid_search_engine.sync_bm25()

        return DocumentIngestResponse(
            status="success",
            doc_id=req.doc_id,
            num_chunks=len(chunks),
            message=f"Successfully indexed {len(chunks)} chunks for document '{req.title}'."
        )
    except Exception as e:
        logger.error(f"Failed to ingest document '{req.doc_id}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/documents/upload", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    title: str = Form(...),
):
    """
    Upload and index text or markdown files directly from multipart form.
    """
    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8", errors="replace")

        return await ingest_document(DocumentIngestRequest(
            doc_id=doc_id,
            title=title,
            content=content_str,
            metadata={"filename": file.filename, "content_type": file.content_type}
        ))
    except Exception as e:
        logger.error(f"Failed file upload for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all indexed documents, chunk counts, and metadata.
    """
    all_chunks = vector_store.get_all_chunks()
    doc_map = {}

    for c in all_chunks:
        if c.doc_id not in doc_map:
            doc_map[c.doc_id] = {
                "doc_id": c.doc_id,
                "title": c.metadata.get("document_title", c.doc_id),
                "num_chunks": 0,
                "created_at": c.metadata.get("created_at", "N/A"),
                "metadata": c.metadata,
            }
        doc_map[c.doc_id]["num_chunks"] += 1

    summaries = [DocumentSummary(**info) for info in doc_map.values()]
    return DocumentListResponse(
        total_documents=len(summaries),
        total_chunks=len(all_chunks),
        documents=summaries
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Remove all chunks associated with a doc_id from the vector store and BM25 index.
    """
    deleted_count = vector_store.delete_document(doc_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    hybrid_search_engine.sync_bm25()
    return {"status": "success", "message": f"Deleted {deleted_count} chunks for document '{doc_id}'."}


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """
    Expose real-time telemetry metrics: query volume, latency percentiles, hit rate, and uptime.
    """
    return metrics_service.get_metrics()


@router.get("/health")
async def health_check():
    """
    Kubernetes / Docker container liveness probe.
    """
    return {
        "status": "healthy",
        "service": "Nexus Agentic RAG Backend",
        "vector_store_chunks": vector_store.count(),
        "bm25_synced": hybrid_search_engine.is_bm25_synced,
    }
