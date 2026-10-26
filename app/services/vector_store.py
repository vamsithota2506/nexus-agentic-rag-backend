import threading
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from app.services.embeddings import embedding_service
from app.core.logging import logger


class ChunkRecord:
    def __init__(self, chunk_id: str, doc_id: str, content: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}


class VectorStore:
    """
    High-performance in-memory Dense Vector Store with cosine similarity indexing
    and metadata filtering.
    """
    def __init__(self):
        self._chunks: Dict[str, ChunkRecord] = {}
        self._lock = threading.Lock()

    def add_chunks(self, chunks_data: List[Dict[str, Any]]):
        """
        Expects list of dicts: {'chunk_id': str, 'doc_id': str, 'content': str, 'metadata': dict}
        """
        contents = [c["content"] for c in chunks_data]
        embeddings = embedding_service.embed_batch(contents)

        with self._lock:
            for item, emb in zip(chunks_data, embeddings):
                record = ChunkRecord(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    content=item["content"],
                    embedding=emb,
                    metadata=item.get("metadata", {})
                )
                self._chunks[item["chunk_id"]] = record

        logger.info(f"Vector store indexed {len(chunks_data)} chunks. Total: {len(self._chunks)}")

    def search(self, query: str, top_k: int = 5, filter_doc_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Cosine similarity search. Returns list of (chunk_id, similarity_score).
        """
        with self._lock:
            if not self._chunks:
                return []

            filtered_records = [
                rec for rec in self._chunks.values()
                if filter_doc_id is None or rec.doc_id == filter_doc_id
            ]

            if not filtered_records:
                return []

            query_vector = embedding_service.embed_text(query)
            chunk_ids = [rec.chunk_id for rec in filtered_records]
            matrix = np.vstack([rec.embedding for rec in filtered_records])

            # Cosine similarity (vectors are L2 normalized)
            scores = np.dot(matrix, query_vector)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = [(chunk_ids[idx], float(scores[idx])) for idx in top_indices]
            return results

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        with self._lock:
            return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> List[ChunkRecord]:
        with self._lock:
            return list(self._chunks.values())

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            to_delete = [cid for cid, rec in self._chunks.items() if rec.doc_id == doc_id]
            for cid in to_delete:
                del self._chunks[cid]
            return len(to_delete)

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)


vector_store = VectorStore()
