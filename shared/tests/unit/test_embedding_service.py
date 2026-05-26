"""Unit tests for EmbeddingService — batching logic and cache integration."""

from __future__ import annotations

import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.embeddings.service import EmbeddingService


def make_fake_response(texts: list[str], dims: int = 3072):
    """Build a mock OpenAI embeddings response."""
    response = MagicMock()
    response.data = [
        MagicMock(index=i, embedding=[1.0 / math.sqrt(dims)] * dims)
        for i in range(len(texts))
    ]
    response.usage = MagicMock(total_tokens=len(texts) * 10)
    return response


class TestEmbeddingService:
    @pytest.fixture
    def service(self):
        svc = EmbeddingService(api_key="sk-test", batch_size=3, dimensions=3072)
        mock_client = AsyncMock()
        svc._client = mock_client
        return svc, mock_client

    @pytest.mark.asyncio
    async def test_embed_single_text(self, service):
        svc, mock_client = service
        mock_client.embeddings.create = AsyncMock(return_value=make_fake_response(["hello"]))
        result = await svc.embed("hello")
        assert len(result) == 3072

    @pytest.mark.asyncio
    async def test_embed_batch_respects_batch_size(self, service):
        svc, mock_client = service
        # 7 texts with batch_size=3 → 3 API calls
        texts = [f"text {i}" for i in range(7)]

        def side_effect(*args, **kwargs):
            n = len(kwargs.get("input", []))
            return make_fake_response(list(range(n)))

        mock_client.embeddings.create = AsyncMock(side_effect=side_effect)
        results = await svc.embed_batch(texts)
        assert len(results) == 7
        assert mock_client.embeddings.create.call_count == 3  # ceil(7/3)

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, service):
        svc, _ = service
        results = await svc.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_cost_tracking_accumulates(self, service):
        svc, mock_client = service
        mock_client.embeddings.create = AsyncMock(return_value=make_fake_response(["a", "b"]))
        await svc.embed_batch(["a", "b"])
        assert svc.total_tokens_used == 20  # 2 texts × 10 tokens each
        assert svc.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api_call(self, service):
        svc, mock_client = service
        mock_cache = AsyncMock()
        embedding = [0.5] * 3072
        mock_cache.get_cached_embedding = AsyncMock(return_value=embedding)
        svc._cache = mock_cache

        results = await svc.embed_batch(["cached text"])
        assert results == [embedding]
        mock_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_api_and_stores(self, service):
        svc, mock_client = service
        mock_cache = AsyncMock()
        mock_cache.get_cached_embedding = AsyncMock(return_value=None)
        mock_cache.cache_embedding = AsyncMock()
        svc._cache = mock_cache

        mock_client.embeddings.create = AsyncMock(
            return_value=make_fake_response(["new text"])
        )
        results = await svc.embed_batch(["new text"])
        assert len(results) == 1
        mock_cache.cache_embedding.assert_called_once()
