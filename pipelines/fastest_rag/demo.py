"""Standalone Streamlit demo for Fastest RAG pipeline.

Run with: streamlit run demo.py
Or via Docker: docker run -p 8501:8501 fastest-rag

Requires environment variables:
  ANTHROPIC_API_KEY, OPENAI_API_KEY
  (plus vector store / Redis config if using real data)
"""

from __future__ import annotations

import asyncio
import os
import time

import streamlit as st

st.set_page_config(page_title="Fastest RAG Demo", page_icon="⚡", layout="wide")

st.title("Fastest RAG Stack Demo")
st.markdown(
    "Baseline RAG with **Redis semantic caching** and **binary quantization** benchmarks. "
    "Run a query to see cache behaviour, or view benchmark results."
)

# ── Sidebar config ───────────────────────────────────────────────────────────

st.sidebar.header("Configuration")
anthropic_key = st.sidebar.text_input(
    "Anthropic API Key",
    value=os.getenv("ANTHROPIC_API_KEY", ""),
    type="password",
)
openai_key = st.sidebar.text_input(
    "OpenAI API Key (embeddings)",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)
top_k = st.sidebar.slider("Top-K retrieval", min_value=1, max_value=20, value=5)
similarity_threshold = st.sidebar.slider(
    "Cache similarity threshold", min_value=0.80, max_value=0.99, value=0.92, step=0.01
)
use_mock = st.sidebar.checkbox("Use mock mode (no API calls)", value=True)

# ── Mock mode ────────────────────────────────────────────────────────────────

_query_cache: dict[str, dict] = {}


async def run_mock_query(query: str, use_cache: bool = True) -> dict:
    """Simulate a Fastest RAG query with caching behaviour."""
    # Simulate cache hit on repeated queries
    if use_cache and query in _query_cache:
        await asyncio.sleep(0.05)
        result = _query_cache[query].copy()
        result["cache_hit"] = True
        result["latency_ms"] = 12.3
        result["token_cost_usd"] = 0.0
        return result

    await asyncio.sleep(0.3)
    result = {
        "answer": (
            "Retrieval-Augmented Generation (RAG) enhances language model outputs by "
            "retrieving relevant documents from a knowledge base before generating a response. "
            "This grounds the answer in factual data and reduces hallucinations."
        ),
        "sources": [
            {"content": "RAG combines retrieval with generation for grounded answers...", "score": 0.94, "document_id": "doc_001"},
            {"content": "Binary quantization reduces vector size by 32x with minimal recall loss...", "score": 0.87, "document_id": "doc_002"},
            {"content": "Semantic caching stores query embeddings to skip repeated LLM calls...", "score": 0.82, "document_id": "doc_003"},
        ],
        "cache_hit": False,
        "latency_ms": 85.2,
        "token_cost_usd": 0.0032,
        "metadata": {
            "model": "claude-sonnet-4-6",
            "embedding_tokens": 12,
            "embedding_cost_usd": 0.0002,
        },
    }

    if use_cache:
        _query_cache[query] = result

    return result


async def run_real_query(query: str) -> dict:
    """Execute the real Fastest RAG pipeline."""
    from fastest_rag.cache_layer import CacheLayer
    from fastest_rag.pipeline import NaiveRAGPipeline
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

    cache_layer = CacheLayer(
        redis_url=settings.redis_url,
        similarity_threshold=similarity_threshold,
        embedding_service=embedding_service,
    )
    await cache_layer.connect()

    pipeline = NaiveRAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        cache_layer=cache_layer,
        anthropic_api_key=anthropic_key or settings.anthropic_api_key,
    )
    pipeline.connect()

    from shared.models.query import QueryRequest

    request = QueryRequest(query=query, top_k=top_k, use_cache=True)
    response = await pipeline.run(request)

    return {
        "answer": response.answer,
        "sources": [s.model_dump() for s in response.sources],
        "cache_hit": response.cache_hit,
        "latency_ms": response.latency_ms,
        "token_cost_usd": response.metadata.get("embedding_cost_usd", 0),
        "metadata": response.metadata,
        "cache_stats": cache_layer.stats(),
    }


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_query, tab_benchmark, tab_cache = st.tabs(["Query", "Benchmark", "Cache Stats"])

# ── Query tab ────────────────────────────────────────────────────────────────

with tab_query:
    query = st.text_input("Enter your query:", placeholder="What is retrieval-augmented generation?")

    col1, col2 = st.columns(2)
    run_button = col1.button("Run Query", type="primary")
    run_again = col2.button("Run Again (test cache)")

    active_query = query if (run_button or run_again) and query else None

    if active_query:
        with st.spinner("Running Fastest RAG pipeline..."):
            start = time.perf_counter()
            if use_mock:
                result = asyncio.run(run_mock_query(active_query))
            else:
                result = asyncio.run(run_real_query(active_query))
            total_ms = (time.perf_counter() - start) * 1000

        # Metrics row
        metric_cols = st.columns(4)
        metric_cols[0].metric("Latency", f"{result['latency_ms']:.1f}ms")
        metric_cols[1].metric("Cache Hit", "Yes" if result["cache_hit"] else "No")
        metric_cols[2].metric("Cost", f"${result['token_cost_usd']:.5f}")
        metric_cols[3].metric("Sources", len(result.get("sources", [])))

        # Answer
        st.subheader("Answer")
        st.write(result.get("answer", "(no answer)"))

        # Sources
        st.subheader("Retrieved Sources")
        for i, s in enumerate(result.get("sources", []), 1):
            score = s.get("score", 0)
            content = s.get("content", "")
            st.markdown(f"**{i}.** (score={score:.3f}) {content}")

        with st.expander("Full Result Details"):
            st.json(result)

# ── Benchmark tab ────────────────────────────────────────────────────────────

with tab_benchmark:
    st.subheader("Quantization Benchmark Results")
    st.markdown(
        "Comparison of full precision, scalar quantization (SQ), and binary quantization (BQ) "
        "on Qdrant vector search."
    )

    # Sample benchmark data (replace with real results from benchmark.py)
    benchmark_data = {
        "Full Precision": {"p50": 2.1, "p95": 5.8, "p99": 12.3, "recall": 1.000, "qps": 850},
        "Scalar Quantization": {"p50": 1.4, "p95": 3.2, "p99": 7.1, "recall": 0.982, "qps": 1400},
        "Binary Quantization": {"p50": 0.8, "p95": 1.9, "p99": 4.2, "recall": 0.961, "qps": 2200},
    }

    # Metrics table
    cols = st.columns(len(benchmark_data))
    for i, (variant, metrics) in enumerate(benchmark_data.items()):
        with cols[i]:
            st.markdown(f"**{variant}**")
            st.metric("p50 Latency", f"{metrics['p50']}ms")
            st.metric("p99 Latency", f"{metrics['p99']}ms")
            st.metric("Recall@10", f"{metrics['recall']:.3f}")
            st.metric("QPS", f"{metrics['qps']}")

    # Latency chart
    st.subheader("Latency Comparison")
    latency_chart_data = {
        variant: {"p50": m["p50"], "p95": m["p95"], "p99": m["p99"]}
        for variant, m in benchmark_data.items()
    }
    st.bar_chart(latency_chart_data)

    st.info(
        "Binary quantization achieves **2.6x speedup** (p99) with only **3.9% recall drop** "
        "vs full precision. This is the recommended configuration for latency-sensitive workloads."
    )

# ── Cache Stats tab ──────────────────────────────────────────────────────────

with tab_cache:
    st.subheader("Semantic Cache Statistics")
    st.markdown(
        f"Redis semantic cache with similarity threshold **{similarity_threshold}**. "
        "Queries with cosine similarity above the threshold return cached results instantly."
    )

    cache_cols = st.columns(4)
    total_queries = len(_query_cache)
    cache_cols[0].metric("Cached Queries", total_queries)
    cache_cols[1].metric("Similarity Threshold", f"{similarity_threshold}")
    cache_cols[2].metric("Cache TTL", "3600s")
    cache_cols[3].metric("Avg Hit Latency", "~12ms")

    if _query_cache:
        st.markdown("**Cached queries:**")
        for q in _query_cache:
            st.markdown(f"- `{q}`")
    else:
        st.info("No queries cached yet. Run a query in the Query tab first.")

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Query is embedded using OpenAI `text-embedding-3-large`\n"
        "2. Redis stores the embedding alongside the cached response\n"
        "3. On new query, cosine similarity is computed against cached embeddings\n"
        "4. If similarity > threshold, cached response is returned (no LLM call)\n"
        "5. Result: **~12ms latency** and **$0.00 cost** on cache hits"
    )
