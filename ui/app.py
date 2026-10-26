import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="Nexus Agentic RAG Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

# Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6; }
    .citation-card { background-color: #F8FAFC; padding: 0.8rem; border-radius: 6px; border: 1px solid #E2E8F0; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚡ Nexus Control Panel")
st.sidebar.markdown("**Production AI & Backend Platform**")

# Retrieval Hyperparameters
st.sidebar.subheader("Retrieval Hyperparameters")
dense_weight = st.sidebar.slider("Dense Semantic Weight (Vector)", 0.0, 1.0, 0.6, 0.05)
sparse_weight = st.sidebar.slider("Sparse Keyword Weight (BM25)", 0.0, 1.0, 0.4, 0.05)
top_k = st.sidebar.slider("Top K Retrieved Chunks", 1, 10, 4)
enable_self_correction = st.sidebar.checkbox("Enable Self-Correction & Verification", value=True)

# Health & Metrics Check
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
    metrics_resp = requests.get(f"{BACKEND_URL}/metrics", timeout=2).json()
    st.sidebar.success(f"● Backend Online ({health_resp.get('vector_store_chunks', 0)} chunks indexed)")
    st.sidebar.markdown(f"**Total Queries:** `{metrics_resp.get('total_queries', 0)}`")
    st.sidebar.markdown(f"**Avg Latency:** `{metrics_resp.get('average_latency_ms', 0)} ms`")
    st.sidebar.markdown(f"**Faithfulness:** `{metrics_resp.get('average_faithfulness', 0)}`")
except Exception:
    st.sidebar.warning("○ Backend offline or starting up at http://127.0.0.1:8000")

# Tabs
tab_query, tab_ingest, tab_metrics = st.tabs(["🔍 Intelligent Query", "📄 Ingest Documents", "📊 System Telemetry"])

with tab_query:
    st.markdown('<div class="main-title">Nexus Agentic RAG Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hybrid Retrieval (Dense + BM25 via Reciprocal Rank Fusion) with Hallucination Guardrails</div>', unsafe_allow_html=True)

    sample_questions = [
        "Explain how Hybrid Search combines dense vectors and BM25 using RRF.",
        "What are the self-correction and guardrails mechanisms in Nexus?",
        "How is the microservices and FastAPI architecture designed for low latency?",
    ]
    selected_sample = st.selectbox("Quick Sample Questions:", [""] + sample_questions)

    user_query = st.text_input("Enter your technical question:", value=selected_sample if selected_sample else "")

    if st.button("Execute Query", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Executing agentic retrieval and verification..."):
                payload = {
                    "query": user_query,
                    "top_k": top_k,
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight,
                    "enable_self_correction": enable_self_correction
                }
                try:
                    res = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Latency", f"{data['latency_ms']} ms")
                        col2.metric("Faithfulness Score", f"{data['faithfulness_score'] * 100:.1f}%")
                        col3.metric("Relevance Score", f"{data['relevance_score'] * 100:.1f}%")

                        if data.get("rewritten_query"):
                            st.info(f"**Query Reformulation (HyDE):** {data['rewritten_query']}")

                        st.subheader("Generated Answer")
                        st.markdown(data["answer"])

                        st.subheader("Grounded Evidence & Citations")
                        for idx, cit in enumerate(data.get("citations", [])):
                            with st.expander(f"Citation #{idx+1} — Chunk: {cit['chunk_id']} (RRF Score: {cit['score']:.4f})"):
                                st.write(cit["snippet"])
                    else:
                        st.error(f"Backend returned error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

with tab_ingest:
    st.subheader("Ingest New Knowledge Document")
    doc_id = st.text_input("Document Identifier (doc_id):", value="cloud_architecture_v1")
    doc_title = st.text_input("Document Title:", value="Cloud Microservices & Distributed Caching")
    doc_content = st.text_area("Document Content (Markdown or Raw Text):", height=200, value="""# Distributed Caching Architecture
Redis clusters serve as Tier-1 cache in front of PostgreSQL databases.
Cache invalidation uses the Cache-Aside pattern with TTL-based expiration.
High-throughput workloads achieve sub-5ms response times under 50,000 QPS.""")

    if st.button("Ingest & Index Document"):
        if not doc_id or not doc_content:
            st.warning("Please fill in Document ID and Content.")
        else:
            with st.spinner("Chunking, generating embeddings, and updating BM25 index..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/documents/ingest", json={
                        "doc_id": doc_id,
                        "title": doc_title,
                        "content": doc_content
                    }, timeout=10)
                    if res.status_code == 201:
                        data = res.json()
                        st.success(f"Success! Indexed {data['num_chunks']} chunks into Vector Store and BM25 index.")
                    else:
                        st.error(f"Ingestion failed: {res.text}")
                except Exception as e:
                    st.error(f"Failed to reach backend: {e}")

with tab_metrics:
    st.subheader("System Observability & Performance Metrics")
    try:
        metrics = requests.get(f"{BACKEND_URL}/metrics", timeout=2).json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", metrics.get("total_queries", 0))
        c2.metric("Avg Latency", f"{metrics.get('average_latency_ms', 0)} ms")
        c3.metric("Retrieval Hit Rate", f"{metrics.get('hybrid_retrieval_hit_rate', 0)*100:.1f}%")
        c4.metric("Total Chunks", metrics.get("total_chunks_indexed", 0))

        st.json(metrics)
    except Exception as e:
        st.info("Start the FastAPI backend to view live metrics.")
