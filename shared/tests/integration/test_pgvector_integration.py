"""Integration tests for PgVectorClient using testcontainers.

Run with: pytest shared/tests/integration/ -v
Requires Docker to be running locally.
"""

from __future__ import annotations

import math

import pytest
import pytest_asyncio

from shared.models.document import ChunkType, DocumentChunk
from shared.storage.vector_store import PgVectorClient


@pytest.fixture(scope="module")
def postgres_url():
    """Spin up a temporary PostgreSQL container with pgvector."""
    pytest.importorskip("testcontainers")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        # Replace psycopg2 driver with asyncpg
        sync_url = pg.get_connection_url()
        async_url = sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
        yield async_url, sync_url


@pytest.fixture(scope="module")
def init_schema(postgres_url):
    """Run the pgvector schema init SQL synchronously."""
    import psycopg2

    _, sync_url = postgres_url

    conn = psycopg2.connect(sync_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_type TEXT NOT NULL DEFAULT 'text',
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding vector(3072),
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
    """)
    conn.close()


@pytest_asyncio.fixture(scope="module")
async def pgvector_client(postgres_url, init_schema):
    async_url, _ = postgres_url
    client = PgVectorClient(database_url=async_url)
    await client.connect()
    yield client
    await client.close()


def make_chunks(n: int = 3) -> list[DocumentChunk]:
    dim = 3072
    return [
        DocumentChunk(
            id=f"integ-chunk-{i:03d}",
            document_id="integ-doc-001",
            chunk_index=i,
            chunk_type=ChunkType.TEXT,
            content=f"Integration test chunk number {i}. " * 5,
            embedding=[1.0 / math.sqrt(dim) * (i + 1)] * dim,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_health_check(pgvector_client):
    assert await pgvector_client.health_check() is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upsert_and_count(pgvector_client):
    chunks = make_chunks(3)
    ids = await pgvector_client.upsert(chunks)
    assert len(ids) == 3
    count = await pgvector_client.count()
    assert count >= 3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_chunk(pgvector_client):
    chunks = make_chunks(1)
    await pgvector_client.upsert(chunks)
    fetched = await pgvector_client.get(chunks[0].id)
    assert fetched is not None
    assert fetched.content == chunks[0].content


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_nonexistent_returns_none(pgvector_client):
    result = await pgvector_client.get("nonexistent-id-xyz")
    assert result is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upsert_updates_existing(pgvector_client):
    chunk = make_chunks(1)[0]
    await pgvector_client.upsert([chunk])
    chunk.content = "Updated content for this chunk"
    await pgvector_client.upsert([chunk])
    fetched = await pgvector_client.get(chunk.id)
    assert fetched.content == "Updated content for this chunk"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_delete_chunks(pgvector_client):
    chunks = make_chunks(2)
    # Give them unique IDs to avoid collisions
    for c in chunks:
        c.id = c.id + "-del"
    await pgvector_client.upsert(chunks)
    deleted = await pgvector_client.delete([c.id for c in chunks])
    assert deleted == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_delete_document(pgvector_client):
    chunks = [
        DocumentChunk(
            id=f"del-doc-chunk-{i}",
            document_id="doc-to-delete",
            chunk_index=i,
            content=f"chunk {i} of document to delete",
        )
        for i in range(3)
    ]
    await pgvector_client.upsert(chunks)
    deleted = await pgvector_client.delete_document("doc-to-delete")
    assert deleted == 3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_returns_results(pgvector_client):
    dim = 3072
    query_vec = [1.0 / math.sqrt(dim)] * dim
    results = await pgvector_client.search(query_vec, top_k=3)
    assert isinstance(results, list)
    # Scores should be between 0 and 1
    for r in results:
        assert 0.0 <= r.score <= 1.0
