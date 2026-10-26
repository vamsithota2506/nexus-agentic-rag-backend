import time
import threading
from typing import Dict, Any, List
from app.services.vector_store import vector_store


class MetricsService:
    """
    In-memory system telemetry and evaluation tracker for production observability.
    """
    def __init__(self):
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._total_queries = 0
        self._latencies: List[float] = []
        self._faithfulness_scores: List[float] = []
        self._retrieval_hits = 0

    def record_query(self, latency_ms: float, faithfulness: float, chunks_retrieved: int):
        with self._lock:
            self._total_queries += 1
            self._latencies.append(latency_ms)
            self._faithfulness_scores.append(faithfulness)
            if chunks_retrieved > 0:
                self._retrieval_hits += 1

            # Retain sliding window of 1000 queries
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]
                self._faithfulness_scores = self._faithfulness_scores[-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = (sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0
            avg_faithfulness = (sum(self._faithfulness_scores) / len(self._faithfulness_scores)) if self._faithfulness_scores else 1.0
            hit_rate = (self._retrieval_hits / self._total_queries) if self._total_queries > 0 else 1.0

            unique_docs = len({c.doc_id for c in vector_store.get_all_chunks()})
            total_chunks = vector_store.count()

            return {
                "total_queries": self._total_queries,
                "average_latency_ms": round(avg_latency, 2),
                "total_documents_indexed": unique_docs,
                "total_chunks_indexed": total_chunks,
                "hybrid_retrieval_hit_rate": round(hit_rate, 4),
                "average_faithfulness": round(avg_faithfulness, 4),
                "uptime_seconds": round(time.time() - self._start_time, 2)
            }


metrics_service = MetricsService()
