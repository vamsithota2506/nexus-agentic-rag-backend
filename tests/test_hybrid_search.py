from app.services.hybrid_search import hybrid_search_engine
from app.services.bm25_retriever import BM25Retriever
from app.services.vector_store import vector_store


def test_bm25_exact_keyword_matching():
    retriever = BM25Retriever()
    docs = [
        ("c1", "Python is an interpreted high-level programming language."),
        ("c2", "PostgreSQL is an advanced open-source relational database."),
        ("c3", "Kubernetes automates software deployment and scaling.")
    ]
    retriever.index_documents(docs)

    results = retriever.search("PostgreSQL relational", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "c2"


def test_hybrid_search_rrf_scoring():
    results = hybrid_search_engine.search("HNSW logarithmic search complexity", top_k=3)
    assert len(results) > 0
    top_chunk = results[0]
    assert "HNSW" in top_chunk["content"] or "Vector" in top_chunk["content"]
    assert top_chunk["rrf_score"] > 0
    assert "dense_score" in top_chunk
    assert "sparse_score" in top_chunk


def test_vector_store_deletion():
    initial_count = vector_store.count()
    deleted = vector_store.delete_document("non_existent_doc_id")
    assert deleted == 0
    assert vector_store.count() == initial_count
