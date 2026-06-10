"""Neo4j client for Video RAG knowledge graph operations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Async-friendly Neo4j client wrapping the official neo4j driver.

    Graph schema (Video RAG):
        (:Video {id, title, url, duration_s})
        (:Topic {name})
        (:Segment {id, video_id, start_s, end_s, transcript, embedding_id})

        (:Video)-[:HAS_SEGMENT]->(:Segment)
        (:Segment)-[:COVERS]->(:Topic)
        (:Topic)-[:RELATED_TO {weight}]->(:Topic)
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Any = None

    async def connect(self) -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
        logger.info("Neo4jClient connected to %s", self._uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    # ── Video node CRUD ───────────────────────────────────────────────────────

    async def upsert_video(self, video_id: str, title: str, url: str, duration_s: float) -> None:
        query = """
        MERGE (v:Video {id: $video_id})
        SET v.title = $title, v.url = $url, v.duration_s = $duration_s
        """
        await self._run(query, video_id=video_id, title=title, url=url, duration_s=duration_s)

    async def upsert_segment(
        self,
        segment_id: str,
        video_id: str,
        start_s: float,
        end_s: float,
        transcript: str,
        embedding_id: str,
    ) -> None:
        query = """
        MERGE (s:Segment {id: $segment_id})
        SET s.video_id = $video_id,
            s.start_s  = $start_s,
            s.end_s    = $end_s,
            s.transcript = $transcript,
            s.embedding_id = $embedding_id
        WITH s
        MATCH (v:Video {id: $video_id})
        MERGE (v)-[:HAS_SEGMENT]->(s)
        """
        await self._run(
            query,
            segment_id=segment_id,
            video_id=video_id,
            start_s=start_s,
            end_s=end_s,
            transcript=transcript,
            embedding_id=embedding_id,
        )

    async def link_segment_to_topic(self, segment_id: str, topic_name: str) -> None:
        query = """
        MERGE (t:Topic {name: $topic_name})
        WITH t
        MATCH (s:Segment {id: $segment_id})
        MERGE (s)-[:COVERS]->(t)
        """
        await self._run(query, segment_id=segment_id, topic_name=topic_name)

    async def link_topics(self, topic_a: str, topic_b: str, weight: float = 1.0) -> None:
        query = """
        MERGE (a:Topic {name: $topic_a})
        MERGE (b:Topic {name: $topic_b})
        MERGE (a)-[r:RELATED_TO]-(b)
        SET r.weight = $weight
        """
        await self._run(query, topic_a=topic_a, topic_b=topic_b, weight=weight)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def get_segments_by_topic(self, topic_name: str) -> list[dict[str, Any]]:
        """Return all segments covering a given topic."""
        query = """
        MATCH (s:Segment)-[:COVERS]->(t:Topic {name: $topic_name})
        RETURN s.id AS segment_id,
               s.video_id AS video_id,
               s.start_s AS start_s,
               s.end_s AS end_s,
               s.transcript AS transcript,
               s.embedding_id AS embedding_id
        ORDER BY s.start_s
        """
        return await self._query(query, topic_name=topic_name)

    async def get_related_topics(self, topic_name: str, depth: int = 2) -> list[str]:
        """BFS-expand related topics up to `depth` hops."""
        query = """
        MATCH path = (t:Topic {name: $topic_name})-[:RELATED_TO*1..$depth]-(related:Topic)
        RETURN DISTINCT related.name AS topic
        """
        rows = await self._query(query, topic_name=topic_name, depth=depth)
        return [r["topic"] for r in rows]

    async def get_video_segments(self, video_id: str) -> list[dict[str, Any]]:
        query = """
        MATCH (v:Video {id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
        RETURN s.id AS segment_id, s.start_s AS start_s, s.end_s AS end_s,
               s.transcript AS transcript, s.embedding_id AS embedding_id
        ORDER BY s.start_s
        """
        return await self._query(query, video_id=video_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run(self, query: str, **params: Any) -> None:
        async with self._driver.session() as session:
            await session.run(query, **params)

    async def _query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            return [dict(record) async for record in result]

    async def health_check(self) -> bool:
        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            logger.exception("Neo4j health check failed")
            return False
