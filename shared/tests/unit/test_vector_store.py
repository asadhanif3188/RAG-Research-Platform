"""Unit tests for VectorStoreClient with mocked DB responses."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.models.document import ChunkType, DocumentChunk
from shared.models.retrieval import RetrievalResult
from shared.storage.vector_store import PgVectorClient, QdrantVectorClient


# ── PgVectorClient unit tests ────────────────────────────────────────────────

class TestPgVectorClientUnit:
    @pytest.fixture
    def client(self):
        c = PgVectorClient(database_url="postgresql+asyncpg://user:pass@localhost/db")
        # Inject mock session factory
        mock_session = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_ctx)
        c._session_factory = mock_factory
        return c, mock_session

    @pytest.mark.asyncio
    async def test_upsert_returns_ids(self, client, sample_chunks):
        c, mock_session = client
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(ch.id,) for ch in sample_chunks]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        ids = await c.upsert(sample_chunks)
        assert len(ids) == len(sample_chunks)

    @pytest.mark.asyncio
    async def test_upsert_empty_list_returns_empty(self, client):
        c, _ = client
        ids = await c.upsert([])
        assert ids == []

    @pytest.mark.asyncio
    async def test_delete_returns_rowcount(self, client):
        c, mock_session = client
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        count = await c.delete(["id1", "id2", "id3"])
        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_empty_list_returns_zero(self, client):
        c, _ = client
        count = await c.delete([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        c, mock_session = client
        mock_session.execute = AsyncMock()
        result = await c.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        c, mock_session = client
        mock_session.execute = AsyncMock(side_effect=Exception("connection refused"))
        result = await c.health_check()
        assert result is False


# ── QdrantVectorClient unit tests ────────────────────────────────────────────

class TestQdrantVectorClientUnit:
    @pytest.fixture
    def client(self):
        c = QdrantVectorClient(collection_name="test_collection")
        c._client = AsyncMock()
        return c

    @pytest.mark.asyncio
    async def test_upsert_skips_chunks_without_embedding(self, client):
        c = client
        chunks = [
            DocumentChunk(document_id="d", chunk_index=0, content="hello"),  # no embedding
        ]
        ids = await c.upsert(chunks)
        assert ids == []

    @pytest.mark.asyncio
    async def test_upsert_with_embedding(self, client, sample_chunks):
        c = client
        c._client.upsert = AsyncMock()
        ids = await c.upsert(sample_chunks)
        assert len(ids) == len(sample_chunks)

    @pytest.mark.asyncio
    async def test_search_returns_retrieval_results(self, client, fake_embedding):
        c = client
        mock_hit = MagicMock()
        mock_hit.id = "abc-123"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "document_id": "doc-1",
            "content": "some text",
            "chunk_type": "text",
            "chunk_index": 0,
        }
        c._client.search = AsyncMock(return_value=[mock_hit])

        results = await c.search(fake_embedding, top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_delete_empty_returns_zero(self, client):
        c = client
        count = await c.delete([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_count(self, client):
        c = client
        mock_info = MagicMock()
        mock_info.points_count = 42
        c._client.get_collection = AsyncMock(return_value=mock_info)
        assert await c.count() == 42

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        c = client
        c._client.get_collections = AsyncMock()
        assert await c.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        c = client
        c._client.get_collections = AsyncMock(side_effect=Exception("timeout"))
        assert await c.health_check() is False
