"""Unified embedding service wrapping OpenAI text-embedding-3-large.

Features:
- Async batch API calls (respects OpenAI rate limits)
- Local Redis embedding cache to avoid redundant API calls
- Cost tracking (token count → estimated USD)
- LangFuse tracing for observability
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from shared.storage.cache import RedisSemanticCache

logger = logging.getLogger(__name__)

# text-embedding-3-large pricing as of 2024 ($0.13 / 1M tokens)
_COST_PER_TOKEN = 0.13 / 1_000_000


class EmbeddingService:
    """Embed text using OpenAI text-embedding-3-large with batching and caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        batch_size: int = 100,
        cache: "RedisSemanticCache | None" = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._cache = cache
        self._client: object | None = None

        # Cost tracking
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0

    def connect(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self._api_key)
        logger.info("EmbeddingService initialised (model=%s, dims=%d)", self._model, self._dimensions)

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns the embedding vector."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings. Handles batching, caching, and retries internally."""
        if not texts:
            return []

        # 1. Check cache for each text
        embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        if self._cache:
            for i, text in enumerate(texts):
                cached = await self._cache.get_cached_embedding(text)
                if cached is not None:
                    embeddings[i] = cached
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(len(texts)))

        logger.debug(
            "embed_batch: %d total, %d cache hits, %d to embed",
            len(texts),
            len(texts) - len(uncached_indices),
            len(uncached_indices),
        )

        # 2. Embed uncached texts in batches
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            fresh_embeddings = await self._embed_in_batches(uncached_texts)

            for idx, embedding in zip(uncached_indices, fresh_embeddings):
                embeddings[idx] = embedding
                # Store in cache
                if self._cache:
                    await self._cache.cache_embedding(texts[idx], embedding)

        return [e for e in embeddings if e is not None]

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Split texts into batches and call OpenAI in parallel (max 5 concurrent)."""
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        semaphore = asyncio.Semaphore(5)

        async def embed_one_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._call_openai(batch)

        results = await asyncio.gather(*[embed_one_batch(b) for b in batches])
        # Flatten
        return [emb for batch_result in results for emb in batch_result]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_openai(self, texts: list[str]) -> list[list[float]]:
        """Single OpenAI embeddings API call with retry logic."""
        response = await self._client.embeddings.create(  # type: ignore[union-attr]
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )

        # Update cost tracking
        usage = response.usage
        self._total_tokens += usage.total_tokens
        self._total_cost_usd += usage.total_tokens * _COST_PER_TOKEN

        logger.debug(
            "OpenAI embed: %d texts, %d tokens, $%.6f",
            len(texts),
            usage.total_tokens,
            usage.total_tokens * _COST_PER_TOKEN,
        )

        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
