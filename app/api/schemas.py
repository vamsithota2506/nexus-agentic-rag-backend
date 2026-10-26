from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunkSchema(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None


class DocumentIngestRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full text or markdown content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    status: str
    doc_id: str
    num_chunks: int
    message: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language user question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    enable_self_correction: bool = Field(default=True, description="Enable agent query rewrite and hallucination verification")


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    original_query: str
    rewritten_query: Optional[str] = None
    citations: List[Citation]
    retrieval_strategy: str
    faithfulness_score: float
    relevance_score: float
    latency_ms: float
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    num_chunks: int
    created_at: str
    metadata: Dict[str, Any]


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentSummary]


class SystemMetricsResponse(BaseModel):
    total_queries: int
    average_latency_ms: float
    total_documents_indexed: int
    total_chunks_indexed: int
    hybrid_retrieval_hit_rate: float
    average_faithfulness: float
    uptime_seconds: float
