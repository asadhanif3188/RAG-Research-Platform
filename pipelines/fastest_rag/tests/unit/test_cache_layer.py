"""Unit tests for CacheLayer — hit/miss tracking and stats."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest

from fastest_rag.cache_layer import CacheLayer


def make_unit_vec(dims: int = 16) -> list[float]:
    val = 1.0 / math.sqrt(dims)
    return [val] * dims


class TestCacheLayer:
    @pytest.fixture
    def layer_with_mock_cache(self):
        layer = CacheLayer(redis_url="redis://localhost:6379/0")
        mock_cache = AsyncMock()
        layer._cache = mock_cache
        mock_emb = AsyncMock()
        mock_emb.embed = AsyncMock(return_value=make_unit_vec())
        layer._embedding_service = mock_emb
        return layer, mock_cache

    @pytest.mark.asyncio
    async def test_initial_stats_are_zero(self, layer_with_mock_cache):
        layer, _ = layer_with_mock_cache
        assert layer.hit_count == 0
        assert layer.miss_count == 0
        assert layer.hit_rate == 0.0

    @pytest.mark.asyncio
    async def test_hit_increments_counter(self, layer_with_mock_cache):
        layer, mock_cache = layer_with_mock_cache
        mock_cache.get = AsyncMock(return_value={"answer": "cached"})

        result = await layer.get("what is RAG?")
        assert result == {"answer": "cached"}
        assert layer.hit_count == 1
        assert layer.miss_count == 0

    @pytest.mark.asyncio
    async def test_miss_increments_counter(self, layer_with_mock_cache):
        layer, mock_cache = layer_with_mock_cache
        mock_cache.get = AsyncMock(return_value=None)

        result = await layer.get("unknown query")
        assert result is None
        assert layer.miss_count == 1
        assert layer.hit_count == 0

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self, layer_with_mock_cache):
        layer, mock_cache = layer_with_mock_cache
        # 2 hits, 1 miss
        mock_cache.get = AsyncMock(side_effect=[
            {"answer": "hit 1"},
            None,
            {"answer": "hit 2"},
        ])

        for q in ["q1", "q2", "q3"]:
            await layer.get(q)

        assert layer.total_requests == 3
        assert layer.hit_rate == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_no_embedding_service_returns_none(self):
        layer = CacheLayer(redis_url="redis://localhost:6379/0")
        layer._embedding_service = None
        result = await layer.get("any query")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_embedding_bypasses_embedding(self, layer_with_mock_cache):
        layer, mock_cache = layer_with_mock_cache
        mock_cache.get = AsyncMock(return_value={"answer": "direct"})

        vec = make_unit_vec()
        result = await layer.get_with_embedding("test", vec)
        assert result == {"answer": "direct"}

    @pytest.mark.asyncio
    async def test_reset_stats_clears_counters(self, layer_with_mock_cache):
        layer, mock_cache = layer_with_mock_cache
        mock_cache.get = AsyncMock(return_value={"answer": "x"})
        await layer.get("query 1")
        assert layer.hit_count == 1

        layer.reset_stats()
        assert layer.hit_count == 0
        assert layer.miss_count == 0
        assert layer.hit_rate == 0.0

    @pytest.mark.asyncio
    async def test_stats_dict_structure(self, layer_with_mock_cache):
        layer, _ = layer_with_mock_cache
        stats = layer.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "avg_hit_latency_ms" in stats
        assert "avg_miss_latency_ms" in stats

    @pytest.mark.asyncio
    async def test_flush_resets_stats(self, layer_with_mock_cache):
        layer, mock_cache = layer_with_mock_cache
        mock_cache.get = AsyncMock(return_value={"answer": "x"})
        await layer.get("q1")
        mock_cache.flush = AsyncMock(return_value=1)

        deleted = await layer.flush()
        assert deleted == 1
        assert layer.hit_count == 0
