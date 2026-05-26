"""Redis semantic cache — stores query embeddings and cached answers."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "rag:semantic_cache:"
_EMBEDDING_KEY_PREFIX = "rag:embedding_cache:"


class RedisSemanticCache:
    """Cache RAG responses by semantic similarity of query embeddings.

    Flow:
    1. Embed incoming query.
    2. Scan recent cached embeddings; if cosine similarity > threshold → return cached answer.
    3. Otherwise compute answer, then store (embedding, answer) in Redis with TTL.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        similarity_threshold: float = 0.92,
        ttl: int = 3600,
        max_scan: int = 500,
    ) -> None:
        self._redis_url = redis_url
        self._threshold = similarity_threshold
        self._ttl = ttl
        self._max_scan = max_scan
        self._redis: Any = None

    async def connect(self) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(self._redis_url, decode_responses=False)
        logger.info("RedisSemanticCache connected to %s", self._redis_url)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(
        self, query: str, query_embedding: list[float]
    ) -> dict[str, Any] | None:
        """Return cached response dict if a semantically similar query exists, else None."""
        keys = await self._redis.keys(f"{_CACHE_KEY_PREFIX}*")
        if not keys:
            return None

        query_vec = np.array(query_embedding, dtype=np.float32)
        best_score = 0.0
        best_key: bytes | None = None

        for key in keys[: self._max_scan]:
            raw = await self._redis.hget(key, "embedding")
            if raw is None:
                continue
            cached_vec = np.frombuffer(raw, dtype=np.float32)
            score = float(np.dot(query_vec, cached_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cached_vec) + 1e-9
            ))
            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= self._threshold and best_key is not None:
            raw_response = await self._redis.hget(best_key, "response")
            if raw_response:
                logger.debug(
                    "Semantic cache HIT (score=%.4f, threshold=%.4f)", best_score, self._threshold
                )
                return json.loads(raw_response)

        logger.debug("Semantic cache MISS (best_score=%.4f)", best_score)
        return None

    async def set(
        self,
        query: str,
        query_embedding: list[float],
        response: dict[str, Any],
    ) -> None:
        """Store a query embedding + response in Redis."""
        cache_key = _CACHE_KEY_PREFIX + hashlib.sha256(query.encode()).hexdigest()
        vec_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        async with self._redis.pipeline() as pipe:
            pipe.hset(cache_key, mapping={
                "query": query,
                "embedding": vec_bytes,
                "response": json.dumps(response),
            })
            pipe.expire(cache_key, self._ttl)
            await pipe.execute()

        logger.debug("Stored in semantic cache: key=%s", cache_key[-8:])

    async def invalidate(self, query: str) -> bool:
        """Delete a specific cache entry by query text. Returns True if deleted."""
        cache_key = _CACHE_KEY_PREFIX + hashlib.sha256(query.encode()).hexdigest()
        deleted = await self._redis.delete(cache_key)
        return deleted > 0

    async def flush(self) -> int:
        """Delete all semantic cache entries. Returns count deleted."""
        keys = await self._redis.keys(f"{_CACHE_KEY_PREFIX}*")
        if keys:
            return await self._redis.delete(*keys)
        return 0

    async def cache_embedding(self, text: str, embedding: list[float]) -> None:
        """Cache a raw text→embedding mapping to avoid redundant API calls."""
        key = _EMBEDDING_KEY_PREFIX + hashlib.sha256(text.encode()).hexdigest()
        await self._redis.set(
            key,
            np.array(embedding, dtype=np.float32).tobytes(),
            ex=self._ttl * 24,  # embeddings stay longer
        )

    async def get_cached_embedding(self, text: str) -> list[float] | None:
        """Retrieve a previously cached embedding or None."""
        key = _EMBEDDING_KEY_PREFIX + hashlib.sha256(text.encode()).hexdigest()
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).tolist()

    async def health_check(self) -> bool:
        try:
            await self._redis.ping()
            return True
        except Exception:
            logger.exception("Redis health check failed")
            return False
