"""KnowledgeGraph — Neo4j topic graph for multi-hop video segment queries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """High-level wrapper around Neo4jClient for video topic graph operations.

    Provides multi-hop queries like "find segments about machine_learning
    in videos tagged ai" by traversing:
        Video -[:HAS_SEGMENT]-> Segment -[:COVERS]-> Topic

    Args:
        neo4j_client: Connected Neo4jClient instance.
        anthropic_api_key: Key for topic extraction via Claude.
        model: Claude model for topic extraction.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        anthropic_api_key: str = "",
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._neo4j = neo4j_client
        self._anthropic_api_key = anthropic_api_key
        self._model = model
        self._client: Any = None

    def connect(self) -> None:
        """Initialize the Anthropic client for topic extraction."""
        import anthropic

        if self._anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=self._anthropic_api_key)
        logger.info("KnowledgeGraph connected")

    async def index_video(
        self,
        video_id: str,
        title: str,
        url: str,
        duration_s: float,
    ) -> None:
        """Create or update a Video node in the graph."""
        await self._neo4j.upsert_video(
            video_id=video_id, title=title, url=url, duration_s=duration_s,
        )

    async def index_segment(
        self,
        segment_id: str,
        video_id: str,
        start_s: float,
        end_s: float,
        transcript: str,
        embedding_id: str,
    ) -> None:
        """Create a Segment node and link it to its Video."""
        await self._neo4j.upsert_segment(
            segment_id=segment_id,
            video_id=video_id,
            start_s=start_s,
            end_s=end_s,
            transcript=transcript,
            embedding_id=embedding_id,
        )

    async def extract_and_link_topics(
        self,
        segment_id: str,
        transcript: str,
    ) -> list[str]:
        """Extract topics from transcript text and link them to the segment.

        Uses Claude to extract key topics, then creates Topic nodes and
        COVERS relationships in the graph.

        Returns:
            List of extracted topic names.
        """
        topics = self._extract_topics(transcript)
        for topic in topics:
            await self._neo4j.link_segment_to_topic(segment_id, topic)
        return topics

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topic keywords from text using Claude."""
        if not self._client:
            return self._simple_topic_extraction(text)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "Extract 1-5 key topics from this transcript segment. "
                    "Return ONLY a comma-separated list of lowercase topic names, "
                    "no explanation.\n\n"
                    f"Text: {text[:500]}"
                ),
            }],
        )
        raw = response.content[0].text.strip()
        topics = [t.strip().lower().replace(" ", "_") for t in raw.split(",") if t.strip()]
        return topics[:5]

    @staticmethod
    def _simple_topic_extraction(text: str) -> list[str]:
        """Fallback topic extraction without LLM — extracts long capitalised words."""
        words = text.split()
        candidates: list[str] = []
        for word in words:
            clean = word.strip(".,!?;:\"'()[]{}").lower()
            if len(clean) >= 5 and clean.isalpha():
                candidates.append(clean)

        # Return unique topics by frequency
        freq: dict[str, int] = {}
        for w in candidates:
            freq[w] = freq.get(w, 0) + 1
        sorted_topics = sorted(freq, key=lambda k: freq[k], reverse=True)
        return sorted_topics[:5]

    async def query_segments_by_topic(
        self,
        topic: str,
        expand_related: bool = True,
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Find segments covering a topic, optionally expanding to related topics.

        Args:
            topic: Topic name to search for.
            expand_related: Whether to include segments from related topics.
            depth: How many hops to traverse for related topics.

        Returns:
            List of segment dicts with video_id, start_s, end_s, transcript.
        """
        segments = await self._neo4j.get_segments_by_topic(topic)

        if expand_related:
            related = await self._neo4j.get_related_topics(topic, depth=depth)
            for related_topic in related:
                related_segments = await self._neo4j.get_segments_by_topic(related_topic)
                # Deduplicate by segment_id
                existing_ids = {s["segment_id"] for s in segments}
                for seg in related_segments:
                    if seg["segment_id"] not in existing_ids:
                        segments.append(seg)
                        existing_ids.add(seg["segment_id"])

        return segments

    async def get_video_topics(self, video_id: str) -> list[str]:
        """Get all topics associated with a video's segments."""
        segments = await self._neo4j.get_video_segments(video_id)
        all_topics: set[str] = set()
        for seg in segments:
            seg_topics = await self._neo4j.get_segments_by_topic(seg.get("segment_id", ""))
            for t in seg_topics:
                all_topics.add(t.get("topic", ""))
        return [t for t in all_topics if t]
