import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.routes import router as api_router
from app.core.logging import logger
from app.core.exceptions import NexusBaseException, nexus_exception_handler
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store
from app.services.hybrid_search import hybrid_search_engine


# Sample seed data loaded on initialization
DEFAULT_KNOWLEDGE_BASE = """
# Enterprise AI System Architecture

## Overview and Principles
Nexus is a production-grade Agentic Retrieval-Augmented Generation (RAG) platform.
It combines dense semantic embeddings with sparse BM25 keyword retrieval through Reciprocal Rank Fusion (RRF).
The core objective is eliminating LLM hallucinations while maintaining sub-100ms retrieval latency.

## Hybrid Search Engine Mechanics
1. Dense Retrieval: Encodes documents into high-dimensional vector spaces using cosine similarity to capture semantic nuances.
2. Sparse Retrieval: Okapi BM25 scores exact keyword matches, crucial for technical identifiers, IDs, and code snippets.
3. Reciprocal Rank Fusion (RRF): Merges dense and sparse ranks with formula RRF(d) = sum(w_m / (60 + rank_m(d))).
This guarantees resilience against vocabulary mismatch and ensures zero recall degradation.

## Self-Correction and Guardrails Loop
Nexus implements Corrective RAG (CRAG) patterns:
- Query Rewriter expands queries with domain synonyms.
- Document Grader filters out irrelevant noise before synthesis.
- Hallucination Verifier computes lexical and factual grounding scores against the retrieved evidence chunks.

## Microservices and API Architecture
The backend is built with asynchronous Python FastAPI, featuring streaming responses via Server-Sent Events (SSE),
in-memory vector caching with thread-safe atomic operations, and Prometheus telemetry endpoints.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Seed initial knowledge base
    logger.info("Initializing Nexus AI Knowledge Engine...")
    chunks = document_processor.process_document(
        doc_id="nexus_architecture_core",
        title="Nexus Architecture & Engineering Whitepaper",
        content=DEFAULT_KNOWLEDGE_BASE,
        metadata={"category": "architecture", "created_at": "2026-10-26"}
    )
    vector_store.add_chunks(chunks)
    hybrid_search_engine.sync_bm25()
    logger.info(f"System ready! Seeded {len(chunks)} chunks into Vector & BM25 stores.")
    yield
    logger.info("Shutting down Nexus AI Engine...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-Grade Agentic RAG & Enterprise Knowledge Retrieval Backend with Hybrid Search, Self-Correction, and Guardrails.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Exception handlers
app.add_exception_handler(NexusBaseException, nexus_exception_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.perf_counter()

    response = await call_next(request)

    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    return response


# Include API Routers
app.include_router(api_router)


@app.get("/", summary="Root Health & Discovery")
async def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "documentation": "/docs",
        "api_v1": "/api/v1",
        "health_endpoint": "/api/v1/health",
        "metrics_endpoint": "/api/v1/metrics"
    }
