import time
from typing import Dict, Any, List
from app.config import settings
from app.services.hybrid_search import hybrid_search_engine
from app.services.llm_provider import llm_provider
from app.services.metrics_service import metrics_service
from app.api.schemas import Citation, QueryResponse
from app.core.logging import logger


class AgenticRAGWorkflow:
    """
    Self-Correcting RAG Agent implementing CRAG (Corrective RAG) and Self-RAG patterns:
    1. Query Rewriting / HyDE Expansion
    2. Hybrid Search (Dense + BM25 with RRF)
    3. Document Relevance Evaluation
    4. Constrained Contextual Generation
    5. Hallucination and Faithfulness Verification
    """
    async def execute(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        enable_self_correction: bool = True
    ) -> QueryResponse:
        start_time = time.perf_counter()

        # Step 1: Query Analysis & Expansion
        rewritten_query = query
        if enable_self_correction:
            rewritten_query = await self._rewrite_query(query)

        # Step 2: Hybrid Retrieval (Dense + BM25 via RRF)
        retrieved_chunks = hybrid_search_engine.search(
            query=rewritten_query,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )

        # Step 3: Document Relevance Evaluation
        graded_chunks = self._grade_document_relevance(retrieved_chunks, rewritten_query)
        relevance_score = self._compute_relevance_score(graded_chunks)

        # Fallback query if relevance is completely zero
        if enable_self_correction and relevance_score < 0.1 and retrieved_chunks:
            logger.info("Low initial relevance score, retrying with raw query...")
            retrieved_chunks = hybrid_search_engine.search(
                query=query,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
            graded_chunks = self._grade_document_relevance(retrieved_chunks, query)
            relevance_score = self._compute_relevance_score(graded_chunks)

        # Step 4: Generation with Citations
        context_text = self._build_context_block(graded_chunks)
        answer = await self._generate_answer(query=query, context=context_text)

        # Step 5: Hallucination & Faithfulness Verification
        faithfulness_score = self._verify_faithfulness(answer=answer, context=context_text)

        # Prepare Citations
        citations = [
            Citation(
                doc_id=c["doc_id"],
                chunk_id=c["chunk_id"],
                snippet=c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
                score=c.get("rrf_score", 0.0)
            )
            for c in graded_chunks
        ]

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Telemetry
        metrics_service.record_query(
            latency_ms=total_latency_ms,
            faithfulness=faithfulness_score,
            chunks_retrieved=len(graded_chunks)
        )

        return QueryResponse(
            answer=answer,
            original_query=query,
            rewritten_query=rewritten_query if enable_self_correction else None,
            citations=citations,
            retrieval_strategy="hybrid_rrf_with_self_correction",
            faithfulness_score=round(faithfulness_score, 3),
            relevance_score=round(relevance_score, 3),
            latency_ms=total_latency_ms,
            metrics={
                "chunks_evaluated": len(retrieved_chunks),
                "chunks_accepted": len(graded_chunks),
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
            }
        )

    async def _rewrite_query(self, query: str) -> str:
        prompt = (
            "You are an AI query reformulation assistant for a technical RAG system.\n"
            "Expand the user question with relevant synonyms, key technical terms, and acronyms\n"
            "to maximize search recall across both vector and BM25 retrievers.\n"
            "Return only the expanded query terms."
        )
        rewritten = await llm_provider.generate_response(prompt, f"User Query: {query}")
        return rewritten.strip()

    def _grade_document_relevance(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter out candidate chunks that lack semantic or lexical overlap."""
        if not chunks:
            return []
        query_words = set(query.lower().split())
        graded = []

        for c in chunks:
            content_words = set(c["content"].lower().split())
            overlap = len(query_words.intersection(content_words))
            # Keep chunks with keyword overlap or strong dense similarity
            if overlap > 0 or c.get("dense_score", 0.0) > 0.20 or c.get("rrf_score", 0.0) > 0.01:
                graded.append(c)

        return graded if graded else chunks[:2]

    def _compute_relevance_score(self, chunks: List[Dict[str, Any]]) -> float:
        if not chunks:
            return 0.0
        scores = [c.get("dense_score", 0.5) for c in chunks]
        return float(sum(scores) / len(scores))

    def _build_context_block(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No matching context found."
        blocks = []
        for c in chunks:
            blocks.append(f"[Chunk ID: {c['chunk_id']} | Doc: {c['doc_id']}]\n{c['content']}")
        return "\n\n".join(blocks)

    async def _generate_answer(self, query: str, context: str) -> str:
        system_prompt = (
            "You are Nexus, an enterprise AI knowledge assistant. You answer user questions\n"
            "strictly using the provided context documents. Maintain technical accuracy,\n"
            "cite details clearly, and state if context is insufficient."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return await llm_provider.generate_response(system_prompt, user_prompt)

    def _verify_faithfulness(self, answer: str, context: str) -> float:
        """
        Calculates factual grounding by analyzing token overlap between answer assertions and context.
        """
        if not context or "No matching context found" in context:
            return 0.0
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        if not answer_tokens:
            return 1.0

        overlap = len(answer_tokens.intersection(context_tokens))
        return min(1.0, max(0.5, overlap / (len(answer_tokens) * 0.7)))


agentic_rag_workflow = AgenticRAGWorkflow()
