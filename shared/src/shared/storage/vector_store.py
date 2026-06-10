"""Vector store abstraction — pgvector and Qdrant backends behind a common interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from shared.models.document import DocumentChunk
from shared.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class VectorStoreClient(ABC):
    """Common interface that all vector store backends must implement."""

    @abstractmethod
    async def upsert(self, chunks: list[DocumentChunk]) -> list[str]:
        """Insert or update chunks. Returns list of stored IDs."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Nearest-neighbour search. Returns ranked results."""

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID. Returns count deleted."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""

    @abstractmethod
    async def get(self, chunk_id: str) -> DocumentChunk | None:
        """Fetch a single chunk by ID."""

    @abstractmethod
    async def count(self) -> int:
        """Total number of stored chunks."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable."""


# ── pgvector backend ──────────────────────────────────────────────────────────


class PgVectorClient(VectorStoreClient):
    """PostgreSQL + pgvector backend using SQLAlchemy async."""

    def __init__(self, database_url: str, table: str = "document_chunks") -> None:
        self._database_url = database_url
        self._table = table
        self._engine: Any = None
        self._session_factory: Any = None

    async def connect(self) -> None:
        """Initialise async engine and session factory (call once at startup)."""
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        self._engine = create_async_engine(self._database_url, pool_pre_ping=True)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info("PgVectorClient connected to %s", self._database_url.split("@")[-1])

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def upsert(self, chunks: list[DocumentChunk]) -> list[str]:
        from sqlalchemy import text

        if not chunks:
            return []

        sql = text(
            f"""
            INSERT INTO {self._table}
                (id, document_id, chunk_index, chunk_type, content, metadata, embedding)
            VALUES
                (:id, :document_id, :chunk_index, :chunk_type, :content, :metadata::jsonb, :embedding)
            ON CONFLICT (id) DO UPDATE SET
                content   = EXCLUDED.content,
                metadata  = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            RETURNING id
            """
        )

        import json

        params = [
            {
                "id": c.id,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type.value,
                "content": c.content,
                "metadata": json.dumps(c.metadata),
                "embedding": str(c.embedding) if c.embedding else None,
            }
            for c in chunks
        ]

        async with self._session_factory() as session:
            result = await session.execute(sql, params)
            await session.commit()
            ids = [str(row[0]) for row in result.fetchall()]
            logger.debug("Upserted %d chunks into pgvector", len(ids))
            return ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        from sqlalchemy import text

        where_clause = ""
        extra_params: dict[str, Any] = {}
        if filters:
            conditions = []
            for i, (key, val) in enumerate(filters.items()):
                param_name = f"filter_{i}"
                conditions.append(f"metadata->>'${key}' = :{param_name}")
                extra_params[param_name] = str(val)
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = text(
            f"""
            SELECT id, document_id, content, chunk_type, metadata,
                   1 - (embedding <=> :query_vec::vector) AS score
            FROM {self._table}
            {where_clause}
            ORDER BY embedding <=> :query_vec::vector
            LIMIT :top_k
            """
        )

        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"query_vec": embedding_str, "top_k": top_k, **extra_params},
            )
            rows = result.fetchall()

        return [
            RetrievalResult(
                chunk_id=str(row.id),
                document_id=row.document_id,
                content=row.content,
                chunk_type=row.chunk_type,
                score=float(row.score),
                metadata=row.metadata or {},
            )
            for row in rows
        ]

    async def delete(self, chunk_ids: list[str]) -> int:
        from sqlalchemy import text

        if not chunk_ids:
            return 0
        sql = text(f"DELETE FROM {self._table} WHERE id = ANY(:ids)")
        async with self._session_factory() as session:
            result = await session.execute(sql, {"ids": chunk_ids})
            await session.commit()
            return result.rowcount

    async def delete_document(self, document_id: str) -> int:
        from sqlalchemy import text

        sql = text(f"DELETE FROM {self._table} WHERE document_id = :doc_id")
        async with self._session_factory() as session:
            result = await session.execute(sql, {"doc_id": document_id})
            await session.commit()
            return result.rowcount

    async def get(self, chunk_id: str) -> DocumentChunk | None:
        from sqlalchemy import text

        sql = text(f"SELECT * FROM {self._table} WHERE id = :id")
        async with self._session_factory() as session:
            result = await session.execute(sql, {"id": chunk_id})
            row = result.fetchone()

        if row is None:
            return None

        from shared.models.document import ChunkType

        return DocumentChunk(
            id=str(row.id),
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            chunk_type=ChunkType(row.chunk_type),
            content=row.content,
            metadata=row.metadata or {},
        )

    async def count(self) -> int:
        from sqlalchemy import text

        sql = text(f"SELECT COUNT(*) FROM {self._table}")
        async with self._session_factory() as session:
            result = await session.execute(sql)
            return result.scalar() or 0

    async def health_check(self) -> bool:
        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.exception("PgVector health check failed")
            return False


# ── Qdrant backend ────────────────────────────────────────────────────────────


class QdrantVectorClient(VectorStoreClient):
    """Qdrant vector database backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        collection_name: str = "document_chunks",
        vector_size: int = 3072,
    ) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._collection = collection_name
        self._vector_size = vector_size
        self._client: Any = None

    async def connect(self) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = AsyncQdrantClient(
            host=self._host, port=self._port, api_key=self._api_key or None
        )
        # Ensure collection exists
        existing = {c.name for c in (await self._client.get_collections()).collections}
        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'", self._collection)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def upsert(self, chunks: list[DocumentChunk]) -> list[str]:
        from qdrant_client.models import PointStruct

        if not chunks:
            return []

        points = [
            PointStruct(
                id=c.id,
                vector=c.embedding or [],
                payload={
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "chunk_type": c.chunk_type.value,
                    "content": c.content,
                    **c.metadata,
                },
            )
            for c in chunks
            if c.embedding
        ]

        await self._client.upsert(collection_name=self._collection, points=points)
        ids = [p.id for p in points]
        logger.debug("Upserted %d points to Qdrant", len(ids))
        return ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            RetrievalResult(
                chunk_id=str(hit.id),
                document_id=hit.payload.get("document_id", ""),
                content=hit.payload.get("content", ""),
                chunk_type=hit.payload.get("chunk_type", "text"),
                score=float(hit.score),
                metadata={
                    k: v
                    for k, v in hit.payload.items()
                    if k not in {"document_id", "content", "chunk_type", "chunk_index"}
                },
            )
            for hit in hits
        ]

    async def delete(self, chunk_ids: list[str]) -> int:
        from qdrant_client.models import PointIdsList

        if not chunk_ids:
            return 0
        await self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=chunk_ids),
        )
        return len(chunk_ids)

    async def delete_document(self, document_id: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        result = await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        return getattr(result, "deleted_count", 0)

    async def get(self, chunk_id: str) -> DocumentChunk | None:
        results = await self._client.retrieve(
            collection_name=self._collection,
            ids=[chunk_id],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None

        p = results[0]
        from shared.models.document import ChunkType

        return DocumentChunk(
            id=str(p.id),
            document_id=p.payload.get("document_id", ""),
            chunk_index=p.payload.get("chunk_index", 0),
            chunk_type=ChunkType(p.payload.get("chunk_type", "text")),
            content=p.payload.get("content", ""),
            embedding=p.vector,
        )

    async def count(self) -> int:
        info = await self._client.get_collection(self._collection)
        return info.points_count or 0

    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            logger.exception("Qdrant health check failed")
            return False
