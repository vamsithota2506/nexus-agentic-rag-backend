import pytest
from app.services.agent_workflow import agentic_rag_workflow


@pytest.mark.asyncio
async def test_agent_workflow_execution():
    query = "What algorithm is used for vector database search complexity?"
    response = await agentic_rag_workflow.execute(
        query=query,
        top_k=3,
        dense_weight=0.6,
        sparse_weight=0.4,
        enable_self_correction=True
    )

    assert response.answer is not None
    assert response.latency_ms > 0
    assert response.faithfulness_score >= 0.0
    assert len(response.citations) > 0
    assert response.retrieval_strategy == "hybrid_rrf_with_self_correction"


@pytest.mark.asyncio
async def test_hallucination_faithfulness_grader():
    context = "Redis uses LRU eviction policy to manage memory constraints."
    grounded_answer = "The system relies on Redis with an LRU eviction policy."
    hallucinated_answer = "The system uses quantum entanglement on Mars."

    score_grounded = agentic_rag_workflow._verify_faithfulness(grounded_answer, context)
    score_hallucinated = agentic_rag_workflow._verify_faithfulness(hallucinated_answer, context)

    assert score_grounded > score_hallucinated
