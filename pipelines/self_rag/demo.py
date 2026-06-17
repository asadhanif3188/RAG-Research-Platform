"""Standalone Streamlit demo for Self-RAG.

Run with: streamlit run demo.py
Or via Docker: docker run -p 8502:8502 self-rag-demo

Requires environment variables:
  ANTHROPIC_API_KEY, TAVILY_API_KEY, OPENAI_API_KEY
  (plus vector store / database config if using real data)
"""

from __future__ import annotations

import asyncio
import os
import time

import streamlit as st

st.set_page_config(page_title="Self-RAG Demo", page_icon="🧠", layout="wide")

st.title("Self-RAG Demo")
st.markdown(
    "Enter a query to see the Self-RAG decision graph in real time. "
    "Five decision points: **RetrieveOrNot** → **RelevanceGrade** → "
    "**HallucinationGrade** → **AnswerGrade** + **Adaptive Retry (HyDE)**"
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


async def run_mock_self_rag(query: str) -> dict:
    """Simulate a Self-RAG execution for demo purposes."""
    await asyncio.sleep(0.5)

    query_lower = query.lower()

    # Simple math/greeting → no retrieval
    if any(kw in query_lower for kw in ["2+2", "hello", "hi", "translate"]):
        return {
            "answer": "4" if "2+2" in query_lower else "Hello! How can I help you?",
            "decision_path": ["retrieve_or_not:NO", "direct_generate"],
            "node_timings": {"retrieve_or_not": 85.0, "direct_generate": 420.5},
            "grounding_grade": "",
            "answer_quality": "",
            "attempts": 0,
            "documents": [],
        }

    # Knowledge query → full path with grounding check
    if any(kw in query_lower for kw in ["rag", "retrieval", "augmented", "generation"]):
        return {
            "answer": (
                "Retrieval-Augmented Generation (RAG) combines information retrieval with "
                "text generation. It retrieves relevant documents from a knowledge base and "
                "uses them as context for the language model to generate accurate answers."
            ),
            "decision_path": [
                "retrieve_or_not:YES",
                "retrieve",
                "grade_relevance:RELEVANT",
                "generate",
                "grade_grounding:GROUNDED",
                "grade_answer:ADDRESSES_QUESTION",
            ],
            "node_timings": {
                "retrieve_or_not": 90.0,
                "retrieve": 125.5,
                "grade_relevance": 350.0,
                "generate": 920.0,
                "grade_grounding": 280.0,
                "grade_answer": 240.0,
            },
            "grounding_grade": "GROUNDED",
            "answer_quality": "ADDRESSES_QUESTION",
            "attempts": 0,
            "documents": [
                {"content": "RAG combines retrieval with generation...", "score": 0.95},
                {"content": "Introduced by Lewis et al. in 2020...", "score": 0.88},
            ],
        }

    # Hallucination scenario → HyDE retry
    return {
        "answer": (
            "After an initial hallucination detection and HyDE-based retry, "
            "the system produced a grounded answer using improved retrieval."
        ),
        "decision_path": [
            "retrieve_or_not:YES",
            "retrieve",
            "grade_relevance:RELEVANT",
            "generate",
            "grade_grounding:NOT_GROUNDED",
            "hyde_expand(attempt=1)",
            "retrieve",
            "grade_relevance:RELEVANT",
            "generate",
            "grade_grounding:GROUNDED",
            "grade_answer:ADDRESSES_QUESTION",
        ],
        "node_timings": {
            "retrieve_or_not": 88.0,
            "retrieve": 130.0,
            "grade_relevance": 340.0,
            "generate": 900.0,
            "grade_grounding": 275.0,
            "hyde_expand": 450.0,
            "retrieve_1": 120.0,
            "grade_relevance_1": 330.0,
            "generate_1": 880.0,
            "grade_grounding_1": 270.0,
            "grade_answer": 235.0,
        },
        "grounding_grade": "GROUNDED",
        "answer_quality": "ADDRESSES_QUESTION",
        "attempts": 1,
        "documents": [
            {"content": "Improved retrieval result via HyDE...", "score": 0.92},
        ],
    }


async def run_real_self_rag(query: str) -> dict:
    """Execute the real Self-RAG pipeline."""
    from self_rag.pipeline import SelfRAGPipeline
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

    pipeline = SelfRAGPipeline(
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

query = st.text_input(
    "Enter your query:",
    placeholder="What is Retrieval-Augmented Generation?",
)

if st.button("Run Self-RAG", type="primary") and query:
    with st.spinner("Running Self-RAG pipeline..."):
        start = time.perf_counter()
        if use_mock:
            result = asyncio.run(run_mock_self_rag(query))
        else:
            result = asyncio.run(run_real_self_rag(query))
        total_ms = (time.perf_counter() - start) * 1000

    # ── Decision path visualization (graph trace) ──────────────────────────
    st.subheader("Graph Trace Visualization")

    decision_path = result.get("decision_path", [])
    grounding = result.get("grounding_grade", "")
    answer_q = result.get("answer_quality", "")
    attempts = result.get("attempts", 0)

    # Color coding for decision nodes
    def get_node_color(step: str) -> str:
        if "retrieve_or_not:NO" in step:
            return "#e67e22"  # orange
        if "retrieve_or_not:YES" in step:
            return "#27ae60"  # green
        if "RELEVANT" in step:
            return "#27ae60"
        if "AMBIGUOUS" in step:
            return "#f39c12"
        if "IRRELEVANT" in step:
            return "#e74c3c"
        if "GROUNDED" in step:
            return "#27ae60"
        if "NOT_GROUNDED" in step:
            return "#e74c3c"
        if "ADDRESSES_QUESTION" in step:
            return "#27ae60"
        if "DOES_NOT_ADDRESS" in step:
            return "#e74c3c"
        if "hyde_expand" in step:
            return "#9b59b6"  # purple
        return "#4a90d9"  # blue

    # Display as horizontal flow with colored boxes
    cols = st.columns(min(len(decision_path), 6))
    for i, step in enumerate(decision_path):
        col_idx = i % min(len(decision_path), 6)
        with cols[col_idx]:
            color = get_node_color(step)
            st.markdown(
                f"<div style='text-align:center; padding:8px; margin:4px 0; "
                f"background-color:{color}; color:white; "
                f"border-radius:8px; font-size:0.8em; font-weight:bold;'>{step}</div>",
                unsafe_allow_html=True,
            )

    arrow_str = " → ".join(decision_path)
    st.caption(f"Path: {arrow_str}")

    # ── Answer ───────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.write(result.get("answer", "(no answer)"))

    # ── Metrics ──────────────────────────────────────────────────────────
    st.subheader("Performance & Grades")
    timings = result.get("node_timings", {})

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Latency", f"{total_ms:.0f}ms")
    metric_cols[1].metric("Nodes Executed", len(decision_path))
    metric_cols[2].metric("Grounding", grounding or "N/A")
    metric_cols[3].metric("Answer Quality", answer_q or "N/A")
    metric_cols[4].metric("Retry Attempts", attempts)

    if timings:
        st.bar_chart(timings)

    # ── Details expander ─────────────────────────────────────────────────
    with st.expander("Full Result Details"):
        st.json(result)
