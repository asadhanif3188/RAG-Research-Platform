"""Standalone Streamlit demo for Corrective RAG (CRAG).

Run with: streamlit run demo.py
Or via Docker: docker run -p 8501:8501 corrective-rag-demo

Requires environment variables:
  ANTHROPIC_API_KEY, TAVILY_API_KEY, OPENAI_API_KEY
  (plus vector store / database config if using real data)
"""

from __future__ import annotations

import asyncio
import os
import time

import streamlit as st

st.set_page_config(page_title="Corrective RAG Demo", page_icon="🔍", layout="wide")

st.title("Corrective RAG (CRAG) Demo")
st.markdown(
    "Enter a query to see the CRAG decision path in real time: "
    "**RELEVANT** (generate directly) / **AMBIGUOUS** (decompose) / **IRRELEVANT** (web search fallback)"
)

# ── Sidebar config ───────────────────────────────────────────────────────────

st.sidebar.header("Configuration")
anthropic_key = st.sidebar.text_input(
    "Anthropic API Key",
    value=os.getenv("ANTHROPIC_API_KEY", ""),
    type="password",
)
tavily_key = st.sidebar.text_input(
    "Tavily API Key",
    value=os.getenv("TAVILY_API_KEY", ""),
    type="password",
)
openai_key = st.sidebar.text_input(
    "OpenAI API Key (embeddings)",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)
top_k = st.sidebar.slider("Top-K retrieval", min_value=1, max_value=20, value=5)
generation_model = st.sidebar.selectbox(
    "Generation model",
    ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    index=0,
)

use_mock = st.sidebar.checkbox("Use mock mode (no API calls)", value=True)

# ── Mock mode for demo without API keys ──────────────────────────────────────


async def run_mock_crag(query: str) -> dict:
    """Simulate a CRAG execution for demo purposes."""
    await asyncio.sleep(0.5)

    # Simple heuristic for demo routing
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["rag", "retrieval", "augmented", "generation"]):
        return {
            "answer": (
                "Retrieval-Augmented Generation (RAG) combines information retrieval with "
                "text generation. It retrieves relevant documents from a knowledge base and "
                "uses them as context for the language model to generate accurate, grounded answers."
            ),
            "overall_grade": "RELEVANT",
            "decision_path": ["retrieve", "grade_documents:RELEVANT", "generate"],
            "node_timings": {"retrieve": 120.5, "grade_documents": 340.2, "generate": 890.1},
            "documents": [
                {"content": "RAG combines retrieval with generation...", "score": 0.95},
                {"content": "Introduced by Lewis et al. in 2020...", "score": 0.88},
            ],
        }
    elif any(kw in query_lower for kw in ["weather", "sports", "news", "today"]):
        return {
            "answer": (
                "Based on web search results, I found the latest information on your query. "
                "This answer was generated using web search fallback since the local knowledge "
                "base did not contain relevant information."
            ),
            "overall_grade": "IRRELEVANT",
            "decision_path": [
                "retrieve",
                "grade_documents:IRRELEVANT",
                "rewrite_query",
                "web_search",
                "generate",
            ],
            "node_timings": {
                "retrieve": 115.0,
                "grade_documents": 310.5,
                "rewrite_query": 180.3,
                "web_search": 1200.0,
                "generate": 920.4,
            },
            "rewritten_query": f"{query} latest information 2024",
            "web_results": [
                {"title": "Latest Info", "url": "https://example.com", "content": "Web result..."},
            ],
        }
    else:
        return {
            "answer": (
                "After extracting relevant sub-sections from partially matching documents, "
                "I can provide the following answer based on the decomposed content."
            ),
            "overall_grade": "AMBIGUOUS",
            "decision_path": [
                "retrieve",
                "grade_documents:AMBIGUOUS",
                "decompose_docs",
                "generate",
            ],
            "node_timings": {
                "retrieve": 118.0,
                "grade_documents": 325.8,
                "decompose_docs": 450.2,
                "generate": 880.5,
            },
            "decomposed_docs": ["Extracted relevant section about the topic..."],
        }


async def run_real_crag(query: str) -> dict:
    """Execute the real CRAG pipeline."""
    from corrective_rag.pipeline import CRAGPipeline
    from shared.config import get_settings
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import PgVectorClient

    settings = get_settings()

    embedding_service = EmbeddingService(
        api_key=openai_key or settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    embedding_service.connect()

    vector_store = PgVectorClient(database_url=settings.database_url)
    await vector_store.connect()

    pipeline = CRAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        anthropic_api_key=anthropic_key or settings.anthropic_api_key,
        tavily_api_key=tavily_key or settings.tavily_api_key,
        generation_model=generation_model,
    )
    pipeline.connect()

    from shared.models.query import QueryRequest

    request = QueryRequest(query=query, top_k=top_k, use_cache=False)
    response = await pipeline.run(request)
    return response.metadata


# ── Main UI ──────────────────────────────────────────────────────────────────

query = st.text_input("Enter your query:", placeholder="What is Retrieval-Augmented Generation?")

if st.button("Run CRAG", type="primary") and query:
    with st.spinner("Running Corrective RAG pipeline..."):
        start = time.perf_counter()
        if use_mock:
            result = asyncio.run(run_mock_crag(query))
        else:
            result = asyncio.run(run_real_crag(query))
        total_ms = (time.perf_counter() - start) * 1000

    # ── Decision path visualization ──────────────────────────────────────
    st.subheader("Decision Path")

    decision_path = result.get("decision_path", [])
    overall_grade = result.get("overall_grade", "UNKNOWN")

    grade_colors = {"RELEVANT": "green", "AMBIGUOUS": "orange", "IRRELEVANT": "red"}
    grade_color = grade_colors.get(overall_grade, "gray")

    cols = st.columns(len(decision_path))
    for i, step in enumerate(decision_path):
        with cols[i]:
            if "grade_documents" in step:
                st.markdown(
                    f"<div style='text-align:center; padding:10px; "
                    f"background-color:{grade_color}; color:white; "
                    f"border-radius:8px; font-weight:bold;'>{step}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center; padding:10px; "
                    f"background-color:#4a90d9; color:white; "
                    f"border-radius:8px;'>{step}</div>",
                    unsafe_allow_html=True,
                )

    # Arrows between steps
    arrow_str = " → ".join(decision_path)
    st.caption(f"Path: {arrow_str}")

    # ── Answer ───────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.write(result.get("answer", "(no answer)"))

    # ── Metrics ──────────────────────────────────────────────────────────
    st.subheader("Performance")
    timings = result.get("node_timings", {})

    metric_cols = st.columns(4)
    metric_cols[0].metric("Overall Grade", overall_grade)
    metric_cols[1].metric("Total Latency", f"{total_ms:.0f}ms")
    metric_cols[2].metric("Nodes Executed", len(decision_path))
    metric_cols[3].metric(
        "Retrieval Time",
        f"{timings.get('retrieve', 0):.0f}ms",
    )

    if timings:
        st.bar_chart(timings)

    # ── Details expander ─────────────────────────────────────────────────
    with st.expander("Full Result Details"):
        st.json(result)
