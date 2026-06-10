"""Unit tests for NaiveRAGPipeline — all external dependencies mocked."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastest_rag.pipeline import NaiveRAGPipeline
from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse
from shared.models.retrieval import RetrievalResult


def make_fake_results(n: int = 3) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"chunk-{i}",
            document_id="doc-001",
            content=f"Context chunk {i}: RAG combines retrieval with generation.",
            score=0.95 - i * 0.05,
        )
        for i in range(n)
    ]


def make_pipeline(cache_layer=None) -> tuple[NaiveRAGPipeline, MagicMock, MagicMock]:
    mock_vector_store = AsyncMock()
    mock_embedding_service = AsyncMock()
    mock_embedding_service.total_tokens_used = 10
    mock_embedding_service.total_cost_usd = 0.000001

    pipeline = NaiveRAGPipeline(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        cache_layer=cache_layer,
        anthropic_api_key="sk-test",
        model="claude-sonnet-4-6",
    )

    # Inject mock Anthropic client
    mock_client = AsyncMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="RAG stands for Retrieval-Augmented Generation.")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    pipeline._client = mock_client

    return pipeline, mock_vector_store, mock_embedding_service


class TestNaiveRAGPipeline:
    @pytest.mark.asyncio
    async def test_run_returns_query_response(self):
        pipeline, mock_vs, mock_emb = make_pipeline()
        fake_embedding = [1.0 / math.sqrt(3072)] * 3072
        mock_emb.embed = AsyncMock(return_value=fake_embedding)
        mock_vs.search = AsyncMock(return_value=make_fake_results(3))

        request = QueryRequest(query="What is RAG?")
        response = await pipeline.run(request)

        assert isinstance(response, QueryResponse)
        assert response.pipeline == PipelineStrategy.FASTEST_RAG
        assert response.answer == "RAG stands for Retrieval-Augmented Generation."
        assert len(response.sources) == 3
        assert response.cache_hit is False
        assert response.latency_ms is not None and response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_run_calls_embed_and_search(self):
        pipeline, mock_vs, mock_emb = make_pipeline()
        fake_embedding = [0.1] * 3072
        mock_emb.embed = AsyncMock(return_value=fake_embedding)
        mock_vs.search = AsyncMock(return_value=make_fake_results(2))

        request = QueryRequest(query="test query", top_k=2)
        await pipeline.run(request)

        mock_emb.embed.assert_called_once_with("test query")
        mock_vs.search.assert_called_once()
        call_kwargs = mock_vs.search.call_args
        assert call_kwargs.kwargs["query_embedding"] == fake_embedding
        assert call_kwargs.kwargs["top_k"] == 2

    @pytest.mark.asyncio
    async def test_cache_hit_skips_vector_search(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value={
            "answer": "Cached answer",
            "sources": [
                {
                    "chunk_id": "c1", "document_id": "d1", "content": "ctx",
                    "chunk_type": "text", "score": 0.9, "metadata": {}
                }
            ],
            "metadata": {},
        })

        pipeline, mock_vs, mock_emb = make_pipeline(cache_layer=mock_cache)
        request = QueryRequest(query="What is RAG?", use_cache=True)
        response = await pipeline.run(request)

        assert response.cache_hit is True
        assert response.answer == "Cached answer"
        mock_vs.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_stores_result(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        pipeline, mock_vs, mock_emb = make_pipeline(cache_layer=mock_cache)
        fake_embedding = [0.1] * 3072
        mock_emb.embed = AsyncMock(return_value=fake_embedding)
        mock_vs.search = AsyncMock(return_value=make_fake_results(1))

        request = QueryRequest(query="cache miss query", use_cache=True)
        await pipeline.run(request)

        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_use_cache_false_skips_cache(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value={"answer": "should not return this"})

        pipeline, mock_vs, mock_emb = make_pipeline(cache_layer=mock_cache)
        mock_emb.embed = AsyncMock(return_value=[0.1] * 3072)
        mock_vs.search = AsyncMock(return_value=make_fake_results(1))

        request = QueryRequest(query="bypass cache", use_cache=False)
        response = await pipeline.run(request)

        assert response.cache_hit is False
        mock_cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_is_built_from_results(self):
        pipeline, mock_vs, mock_emb = make_pipeline()
        mock_emb.embed = AsyncMock(return_value=[0.1] * 3072)
        results = make_fake_results(3)
        mock_vs.search = AsyncMock(return_value=results)

        context = pipeline._build_context(results)
        assert "[Chunk 1]" in context
        assert "[Chunk 2]" in context
        assert "[Chunk 3]" in context
        assert "Context chunk 0" in context

    @pytest.mark.asyncio
    async def test_no_sources_returns_answer(self):
        pipeline, mock_vs, mock_emb = make_pipeline()
        mock_emb.embed = AsyncMock(return_value=[0.1] * 3072)
        mock_vs.search = AsyncMock(return_value=[])

        request = QueryRequest(query="empty retrieval")
        response = await pipeline.run(request)

        assert response.answer  # still generates with empty context
        assert response.sources == []
