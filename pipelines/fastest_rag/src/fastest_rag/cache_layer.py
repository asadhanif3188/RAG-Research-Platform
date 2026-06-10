"""CacheLayer — wraps RedisSemanticCache with hit-rate and latency tracking.

This sits between NaiveRAGPipeline and the shared RedisSemanticCache,
adding observability metrics useful for the benchmark dashboard.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from shared.storage.cache import RedisSemanticCache

logger = logging.getLogger(__name__)


class CacheLayer:
    """Thin wrapper around RedisSemanticCache that tracks hit/miss stats.

    Args:
        redis_url: Redis connection URL.
        similarity_threshold: Cosine similarity required for a cache hit.
        ttl: Entry TTL in seconds.
        embedding_service: EmbeddingService used to embed queries for lookup.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        similarity_threshold: float = 0.92,
        ttl: int = 3600,
        embedding_service: Any | None = None,
    ) -> None:
        self._cache = RedisSemanticCache(
            redis_url=redis_url,
            similarity_threshold=similarity_threshold,
            ttl=ttl,
        )
        self._embedding_service = embedding_service
        self._hits: int = 0
        self._misses: int = 0
        self._total_hit_latency_ms: float = 0.0
        self._total_miss_latency_ms: float = 0.0

    async def connect(self) -> None:
        await self._cache.connect()

    async def close(self) -> None:
        await self._cache.close()

    async def get(self, query: str) -> dict[str, Any] | None:
        """Return cached response for query, or None on miss.

        Embeds the query internally if an embedding_service is provided,
        otherwise returns None (caller must use get_with_embedding).
        """
        if self._embedding_service is None:
            return None

        start = time.perf_counter()
        embedding = await self._embedding_service.embed(query)
        result = await self._cache.get(query, embedding)
        elapsed = (time.perf_counter() - start) * 1000

        if result is not None:
            self._hits += 1
            self._total_hit_latency_ms += elapsed
            logger.debug("Cache HIT (%.1fms, hits=%d)", elapsed, self._hits)
        else:
            self._misses += 1
            self._total_miss_latency_ms += elapsed
            logger.debug("Cache MISS (%.1fms, misses=%d)", elapsed, self._misses)

        return result

    async def get_with_embedding(self, query: str, embedding: list[float]) -> dict[str, Any] | None:
        """Return cached response using a pre-computed embedding."""
        start = time.perf_counter()
        result = await self._cache.get(query, embedding)
        elapsed = (time.perf_counter() - start) * 1000

        if result is not None:
            self._hits += 1
            self._total_hit_latency_ms += elapsed
        else:
            self._misses += 1
            self._total_miss_latency_ms += elapsed

        return result

    async def set(
        self,
        query: str,
        embedding: list[float],
        response: dict[str, Any],
    ) -> None:
        await self._cache.set(query, embedding, response)

    async def flush(self) -> int:
        """Clear all cache entries. Returns count deleted."""
        count = await self._cache.flush()
        self.reset_stats()
        return count

    async def health_check(self) -> bool:
        return await self._cache.health_check()

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    @property
    def total_requests(self) -> int:
        return self._hits + self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0–1.0)."""
        total = self.total_requests
        return self._hits / total if total > 0 else 0.0

    @property
    def avg_hit_latency_ms(self) -> float:
        return self._total_hit_latency_ms / self._hits if self._hits > 0 else 0.0

    @property
    def avg_miss_latency_ms(self) -> float:
        return self._total_miss_latency_ms / self._misses if self._misses > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self.total_requests,
            "hit_rate": self.hit_rate,
            "avg_hit_latency_ms": self.avg_hit_latency_ms,
            "avg_miss_latency_ms": self.avg_miss_latency_ms,
        }

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_hit_latency_ms = 0.0
        self._total_miss_latency_ms = 0.0
