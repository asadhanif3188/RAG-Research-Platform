"""Integration tests for RedisSemanticCache using testcontainers."""

from __future__ import annotations

import math

import pytest
import pytest_asyncio

from shared.storage.cache import RedisSemanticCache


@pytest.fixture(scope="session")
def redis_url():
    pytest.importorskip("testcontainers")
    from testcontainers.redis import RedisContainer

    with RedisContainer("redis:7.2-alpine") as redis:
        yield f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}/0"


@pytest_asyncio.fixture(scope="function")
async def cache(redis_url):
    c = RedisSemanticCache(
        redis_url=redis_url,
        similarity_threshold=0.90,
        ttl=300,
    )
    await c.connect()
    yield c
    await c.flush()
    await c.close()


def unit_vec(dims: int = 16, offset: float = 0.0) -> list[float]:
    """Small unit vector (16-dim for test speed)."""
    val = 1.0 / math.sqrt(dims) + offset
    vec = [val] * dims
    # Re-normalise
    norm = math.sqrt(sum(x**2 for x in vec))
    return [x / norm for x in vec]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_health_check(cache):
    assert await cache.health_check() is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cache_miss_returns_none(cache):
    result = await cache.get("unknown query", unit_vec())
    assert result is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_set_and_get_exact_match(cache):
    query = "What is retrieval-augmented generation?"
    embedding = unit_vec()
    response = {"answer": "RAG combines retrieval with generation."}

    await cache.set(query, embedding, response)
    hit = await cache.get(query, embedding)
    assert hit is not None
    assert hit["answer"] == response["answer"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_semantic_similarity_hit(cache):
    """Two very similar embeddings (cosine ~1.0) should hit the cache."""
    query = "semantic hit test query"
    embedding = unit_vec(offset=0.0)
    similar_embedding = unit_vec(offset=0.001)  # nearly identical

    response = {"answer": "Cached answer"}
    await cache.set(query, embedding, response)

    hit = await cache.get("similar semantic query", similar_embedding)
    assert hit is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_low_similarity_is_miss(cache):
    """An orthogonal embedding should not match the cache."""
    query = "orthogonal miss test"
    embedding_a = [1.0] + [0.0] * 15
    embedding_b = [0.0, 1.0] + [0.0] * 14

    await cache.set(query, embedding_a, {"answer": "something"})
    hit = await cache.get("different query", embedding_b)
    # With threshold=0.90, orthogonal vectors (cosine~0) should miss
    assert hit is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_invalidate(cache):
    query = "invalidation test"
    embedding = unit_vec()
    await cache.set(query, embedding, {"answer": "temp"})
    deleted = await cache.invalidate(query)
    assert deleted is True
    hit = await cache.get(query, embedding)
    assert hit is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_embedding_cache_roundtrip(cache):
    text = "hello world"
    embedding = [0.1] * 3072
    await cache.cache_embedding(text, embedding)
    retrieved = await cache.get_cached_embedding(text)
    assert retrieved is not None
    assert len(retrieved) == 3072
    assert abs(retrieved[0] - 0.1) < 1e-5


@pytest.mark.asyncio
@pytest.mark.integration
async def test_flush_clears_all(cache):
    for i in range(3):
        await cache.set(f"query {i}", unit_vec(offset=float(i) * 0.1), {"a": i})
    deleted = await cache.flush()
    assert deleted >= 3
