"""E2E tests — verify all 5 pipelines via the FastAPI /query endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from shared.models.query import PipelineStrategy, QueryResponse
from shared.models.retrieval import RetrievalResult


def _make_response(
    query: str,
    pipeline: PipelineStrategy,
    metadata: dict[str, Any] | None = None,
) -> QueryResponse:
    """Build a realistic mock QueryResponse."""
    return QueryResponse(
        query=query,
        answer=f"Mock answer from {pipeline.value} pipeline.",
        pipeline=pipeline,
        sources=[
            RetrievalResult(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Sample retrieved content for testing.",
                score=0.92,
                metadata={"page_number": 1},
            ),
        ],
        latency_ms=42.5,
        cache_hit=False,
        metadata=metadata or {"total_cost_usd": 0.00035},
    )


def _mock_pipeline(pipeline_strategy: PipelineStrategy, extra_meta: dict[str, Any] | None = None):
    """Create a mock pipeline with a .run() async method."""
    mock = AsyncMock()
    mock.run = AsyncMock(
        side_effect=lambda req: _make_response(req.query, pipeline_strategy, extra_meta)
    )
    return mock


@pytest.fixture(autouse=True)
def _override_dependencies():
    """Override all pipeline dependencies with mocks."""
    from api.dependencies import (
        get_cache_layer,
        get_crag_pipeline,
        get_multimodal_pipeline,
        get_naive_pipeline,
        get_self_rag_pipeline,
        get_video_rag_pipeline,
    )

    mock_cache = AsyncMock()
    mock_cache.health_check = AsyncMock(return_value=True)
    mock_cache.stats = lambda: {"hits": 10, "misses": 5}

    app.dependency_overrides[get_naive_pipeline] = lambda: _mock_pipeline(
        PipelineStrategy.FASTEST_RAG,
    )
    app.dependency_overrides[get_multimodal_pipeline] = lambda: _mock_pipeline(
        PipelineStrategy.MULTIMODAL_RAG,
        {"total_cost_usd": 0.0012, "provenance": []},
    )
    app.dependency_overrides[get_crag_pipeline] = lambda: _mock_pipeline(
        PipelineStrategy.CORRECTIVE_RAG,
        {"total_cost_usd": 0.0008, "decision_path": ["retrieve", "grade", "web_search"]},
    )
    app.dependency_overrides[get_self_rag_pipeline] = lambda: _mock_pipeline(
        PipelineStrategy.SELF_RAG,
        {"total_cost_usd": 0.0015, "decision_path": ["retrieve", "grade", "generate"]},
    )
    app.dependency_overrides[get_video_rag_pipeline] = lambda: _mock_pipeline(
        PipelineStrategy.VIDEO_RAG,
        {
            "total_cost_usd": 0.0010,
            "segments_retrieved": 3,
            "video_ids": ["vid-001"],
            "start_ts": 12.5,
            "end_ts": 45.0,
        },
    )
    app.dependency_overrides[get_cache_layer] = lambda: mock_cache

    yield

    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_fastest_rag_query(client: AsyncClient) -> None:
    resp = await client.post("/query", json={"query": "What is RAG?", "pipeline": "fastest_rag"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"] == "fastest_rag"
    assert data["answer"]
    assert len(data["sources"]) >= 1
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_multimodal_rag_query(client: AsyncClient) -> None:
    resp = await client.post(
        "/query", json={"query": "Show me the architecture diagram", "pipeline": "multimodal_rag"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"] == "multimodal_rag"
    assert data["answer"]


@pytest.mark.asyncio
async def test_crag_query_with_web_search(client: AsyncClient) -> None:
    resp = await client.post(
        "/query", json={"query": "What is the latest news?", "pipeline": "corrective_rag"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"] == "corrective_rag"
    assert "decision_path" in data["metadata"]


@pytest.mark.asyncio
async def test_self_rag_query(client: AsyncClient) -> None:
    resp = await client.post(
        "/query", json={"query": "Explain transformers", "pipeline": "self_rag"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"] == "self_rag"


@pytest.mark.asyncio
async def test_video_rag_query_has_timestamps(client: AsyncClient) -> None:
    resp = await client.post(
        "/query", json={"query": "Find the section about embeddings", "pipeline": "video_rag"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"] == "video_rag"
    meta = data["metadata"]
    assert "start_ts" in meta or "segments_retrieved" in meta


@pytest.mark.asyncio
async def test_all_pipelines_return_consistent_schema(client: AsyncClient) -> None:
    """All pipelines must return the same top-level response keys."""
    required_keys = {
        "query",
        "answer",
        "pipeline",
        "sources",
        "latency_ms",
        "cache_hit",
        "metadata",
    }
    for pipeline in ["fastest_rag", "multimodal_rag", "corrective_rag", "self_rag", "video_rag"]:
        resp = await client.post("/query", json={"query": "test", "pipeline": pipeline})
        assert resp.status_code == 200
        assert required_keys.issubset(resp.json().keys()), f"{pipeline} missing keys"


@pytest.mark.asyncio
async def test_metrics_recorded_after_query(client: AsyncClient) -> None:
    """Verify the metrics collector records queries."""
    # Clear metrics first
    await client.delete("/metrics")

    # Run a query
    await client.post("/query", json={"query": "test metrics", "pipeline": "fastest_rag"})

    # Check metrics
    resp = await client.get("/metrics/summary")
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["total_queries"] >= 1


@pytest.mark.asyncio
async def test_metrics_history_endpoint(client: AsyncClient) -> None:
    await client.post("/query", json={"query": "history test", "pipeline": "self_rag"})
    resp = await client.get("/metrics/history", params={"pipeline": "self_rag", "limit": 10})
    assert resp.status_code == 200
    history = resp.json()
    assert isinstance(history, list)


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
