"""Integration test — MultimodalRAGPipeline end-to-end via FastAPI POST /query.

Mocks all external services (Anthropic, OpenAI, pgvector, Redis) so no live
infrastructure is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from shared.models.retrieval import RetrievalResult


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multimodal_rag_e2e():
    """POST /query with pipeline=multimodal_rag returns a valid QueryResponse with provenance."""
    from api import dependencies
    from api.main import app

    # Build mocks
    fake_embedding = [0.1] * 3072
    mock_emb_svc = AsyncMock()
    mock_emb_svc.embed = AsyncMock(return_value=fake_embedding)
    mock_emb_svc.total_tokens_used = 15
    mock_emb_svc.total_cost_usd = 0.000002

    mock_vector_store = AsyncMock()
    mock_vector_store.search = AsyncMock(
        return_value=[
            RetrievalResult(
                chunk_id="c1",
                document_id="paper-1",
                content="Multimodal RAG retrieves both text and image descriptions for richer context.",
                chunk_type="text",
                score=0.92,
                metadata={"page_number": 3},
            ),
            RetrievalResult(
                chunk_id="c2",
                document_id="paper-1",
                content="Figure 2 shows a bar chart of retrieval recall across three chunk types.",
                chunk_type="image_description",
                score=0.85,
                metadata={"page_number": 4},
            ),
        ]
    )

    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_cache.health_check = AsyncMock(return_value=True)
    mock_cache.stats = MagicMock(return_value={"hits": 0, "misses": 1, "hit_rate": 0.0})

    mock_message = MagicMock()
    mock_message.content = [
        MagicMock(
            text=(
                "Multimodal RAG retrieves both text and image descriptions for richer context. "
                "Figure 2 shows retrieval recall across chunk types."
            )
        )
    ]
    mock_anthropic = AsyncMock()
    mock_anthropic.messages.create = AsyncMock(return_value=mock_message)

    from multimodal_rag.pipeline import MultimodalRAGPipeline

    pipeline = MultimodalRAGPipeline(
        vector_store=mock_vector_store,
        embedding_service=mock_emb_svc,
        cache_layer=mock_cache,
        anthropic_api_key="sk-test",
        track_provenance=True,
    )
    pipeline._client = mock_anthropic

    app.dependency_overrides[dependencies.get_embedding_service] = lambda: mock_emb_svc
    app.dependency_overrides[dependencies.get_pgvector_client] = lambda: mock_vector_store
    app.dependency_overrides[dependencies.get_cache_layer] = lambda: mock_cache
    app.dependency_overrides[dependencies.get_multimodal_pipeline] = lambda: pipeline

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={
                    "query": "How does multimodal RAG improve retrieval?",
                    "pipeline": "multimodal_rag",
                    "top_k": 5,
                    "use_cache": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline"] == "multimodal_rag"
        assert data["answer"]
        assert len(data["sources"]) == 2
        assert data["cache_hit"] is False
        assert data["latency_ms"] is not None
        assert "provenance" in data["metadata"]
        assert "chunk_type_distribution" in data["metadata"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multimodal_cache_hit():
    """Cached multimodal response is returned immediately."""
    from api import dependencies
    from api.main import app

    mock_emb_svc = AsyncMock()
    mock_emb_svc.embed = AsyncMock(return_value=[0.1] * 3072)
    mock_emb_svc.total_tokens_used = 0
    mock_emb_svc.total_cost_usd = 0.0

    mock_vector_store = AsyncMock()
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(
        return_value={
            "answer": "Cached multimodal answer.",
            "sources": [],
            "metadata": {"provenance": [], "chunk_type_distribution": {}},
        }
    )

    from multimodal_rag.pipeline import MultimodalRAGPipeline

    pipeline = MultimodalRAGPipeline(
        vector_store=mock_vector_store,
        embedding_service=mock_emb_svc,
        cache_layer=mock_cache,
        anthropic_api_key="sk-test",
    )

    app.dependency_overrides[dependencies.get_embedding_service] = lambda: mock_emb_svc
    app.dependency_overrides[dependencies.get_pgvector_client] = lambda: mock_vector_store
    app.dependency_overrides[dependencies.get_cache_layer] = lambda: mock_cache
    app.dependency_overrides[dependencies.get_multimodal_pipeline] = lambda: pipeline

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"query": "test query", "pipeline": "multimodal_rag", "use_cache": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["cache_hit"] is True
        assert data["answer"] == "Cached multimodal answer."
    finally:
        app.dependency_overrides.clear()
