from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.vector_store import vector_store
from app.services.bm25_retriever import BM25Retriever
from app.core.logging import logger


class HybridSearchEngine:
    """
    Hybrid Search combining Dense Vector Cosine Similarity and Sparse Okapi BM25
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.is_bm25_synced = False

    def sync_bm25(self):
        """Re-indexes all active chunks into BM25 inverted index."""
        all_chunks = vector_store.get_all_chunks()
        doc_tuples = [(c.chunk_id, c.content) for c in all_chunks]
        self.bm25_retriever.index_documents(doc_tuples)
        self.is_bm25_synced = True
        logger.info(f"BM25 index synchronized with {len(doc_tuples)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = None,
        sparse_weight: float = None,
        filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search and merges results via Reciprocal Rank Fusion.
        """
        w_dense = dense_weight if dense_weight is not None else settings.DENSE_WEIGHT
        w_sparse = sparse_weight if sparse_weight is not None else settings.SPARSE_WEIGHT
        rrf_k = settings.RRF_K_CONSTANT

        # 1. Dense Semantic Search
        dense_results = vector_store.search(query=query, top_k=top_k * 2, filter_doc_id=filter_doc_id)
        dense_rank_map = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(dense_results)}
        dense_score_map = {chunk_id: score for chunk_id, score in dense_results}

        # 2. Sparse BM25 Keyword Search
        if not self.is_bm25_synced:
            self.sync_bm25()

        sparse_results = self.bm25_retriever.search(query=query, top_k=top_k * 2)
        sparse_rank_map = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(sparse_results)}
        sparse_score_map = {chunk_id: score for chunk_id, score in sparse_results}

        # 3. Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_rank_map.keys()).union(set(sparse_rank_map.keys()))
        fused_scores: Dict[str, float] = {}

        for chunk_id in all_candidate_ids:
            score = 0.0
            if chunk_id in dense_rank_map:
                score += w_dense * (1.0 / (rrf_k + dense_rank_map[chunk_id]))
            if chunk_id in sparse_rank_map:
                score += w_sparse * (1.0 / (rrf_k + sparse_rank_map[chunk_id]))
            fused_scores[chunk_id] = score

        # Sort descending by RRF score
        sorted_candidates = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        enriched_results = []
        for chunk_id, rrf_score in sorted_candidates:
            chunk = vector_store.get_chunk(chunk_id)
            if not chunk:
                continue

            enriched_results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "dense_score": dense_score_map.get(chunk_id, 0.0),
                "sparse_score": sparse_score_map.get(chunk_id, 0.0),
                "rrf_score": round(rrf_score, 6),
            })

        return enriched_results


hybrid_search_engine = HybridSearchEngine()
