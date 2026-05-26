"""Integration test — end-to-end POST /query through FastAPI.

Uses httpx.AsyncClient and mocks all external services (Anthropic, OpenAI, pgvector, Redis)
so the test can run without live infrastructure.
"""

from __future__ import annotations

import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport

from shared.models.query import PipelineStrategy, QueryRequest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fastest_rag_e2e():
    """POST /query with pipeline=fastest_rag returns a valid QueryResponse."""
    from api.main import app
    from api import dependencies

    # Mock all external services
    fake_embedding = [1.0 / math.sqrt(3072)] * 3072
    mock_emb_svc = AsyncMock()
    mock_emb_svc.embed = AsyncMock(return_value=fake_embedding)
    mock_emb_svc.total_tokens_used = 10
    mock_emb_svc.total_cost_usd = 0.000001

    from shared.models.retrieval import RetrievalResult
    mock_vector_store = AsyncMock()
    mock_vector_store.search = AsyncMock(return_value=[
        RetrievalResult(
            chunk_id="c1",
            document_id="doc-1",
            content="RAG combines retrieval with language model generation.",
            score=0.95,
        )
    ])

    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_cache.health_check = AsyncMock(return_value=True)
    mock_cache.stats = MagicMock(return_value={"hits": 0, "misses": 1, "hit_rate": 0.0})

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="RAG stands for Retrieval-Augmented Generation.")]
    mock_anthropic_client = AsyncMock()
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_message)

    # Override dependencies
    app.dependency_overrides[dependencies.get_embedding_service] = lambda: mock_emb_svc
    app.dependency_overrides[dependencies.get_pgvector_client] = lambda: mock_vector_store
    app.dependency_overrides[dependencies.get_cache_layer] = lambda: mock_cache

    from fastest_rag.pipeline import NaiveRAGPipeline
    pipeline = NaiveRAGPipeline(
        vector_store=mock_vector_store,
        embedding_service=mock_emb_svc,
        cache_layer=mock_cache,
        anthropic_api_key="sk-test",
    )
    pipeline._client = mock_anthropic_client
    app.dependency_overrides[dependencies.get_naive_pipeline] = lambda: pipeline

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={
                    "query": "What is RAG?",
                    "pipeline": "fastest_rag",
                    "top_k": 3,
                    "use_cache": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline"] == "fastest_rag"
        assert data["answer"] == "RAG stands for Retrieval-Augmented Generation."
        assert len(data["sources"]) == 1
        assert data["cache_hit"] is False
        assert data["latency_ms"] is not None
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_health_endpoint():
    """GET /health returns ok."""
    from api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_unimplemented_pipeline_returns_501():
    """POST /query with an unimplemented pipeline returns 501."""
    from api.main import app
    from api import dependencies

    mock_emb_svc = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.health_check = AsyncMock(return_value=True)
    mock_cache.stats = MagicMock(return_value={})

    from fastest_rag.pipeline import NaiveRAGPipeline
    pipeline = NaiveRAGPipeline(
        vector_store=mock_vector_store,
        embedding_service=mock_emb_svc,
        cache_layer=mock_cache,
        anthropic_api_key="sk-test",
    )

    app.dependency_overrides[dependencies.get_embedding_service] = lambda: mock_emb_svc
    app.dependency_overrides[dependencies.get_pgvector_client] = lambda: mock_vector_store
    app.dependency_overrides[dependencies.get_cache_layer] = lambda: mock_cache
    app.dependency_overrides[dependencies.get_naive_pipeline] = lambda: pipeline

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"query": "test", "pipeline": "self_rag"},
            )
        assert response.status_code == 501
    finally:
        app.dependency_overrides.clear()
