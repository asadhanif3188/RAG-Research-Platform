"""Unit tests for KnowledgeGraph."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from video_rag.knowledge_graph import KnowledgeGraph


class TestKnowledgeGraph:
    def _make_kg(self) -> tuple[KnowledgeGraph, AsyncMock]:
        mock_neo4j = AsyncMock()
        kg = KnowledgeGraph(neo4j_client=mock_neo4j)
        return kg, mock_neo4j

    @pytest.mark.asyncio
    async def test_index_video(self) -> None:
        kg, mock_neo4j = self._make_kg()
        await kg.index_video("v1", "Test", "https://example.com", 120.0)
        mock_neo4j.upsert_video.assert_called_once_with(
            video_id="v1", title="Test", url="https://example.com", duration_s=120.0,
        )

    @pytest.mark.asyncio
    async def test_index_segment(self) -> None:
        kg, mock_neo4j = self._make_kg()
        await kg.index_segment("s1", "v1", 0.0, 5.0, "Hello", "emb1")
        mock_neo4j.upsert_segment.assert_called_once()

    def test_simple_topic_extraction(self) -> None:
        text = "Machine learning and artificial intelligence are transforming healthcare"
        topics = KnowledgeGraph._simple_topic_extraction(text)
        assert isinstance(topics, list)
        assert len(topics) <= 5
        assert all(isinstance(t, str) for t in topics)

    @pytest.mark.asyncio
    async def test_extract_and_link_topics_fallback(self) -> None:
        """Without anthropic key, falls back to simple extraction."""
        kg, mock_neo4j = self._make_kg()
        topics = await kg.extract_and_link_topics("s1", "machine learning is great")

        assert isinstance(topics, list)
        # link_segment_to_topic called for each topic
        assert mock_neo4j.link_segment_to_topic.call_count == len(topics)

    @pytest.mark.asyncio
    async def test_query_segments_by_topic(self) -> None:
        kg, mock_neo4j = self._make_kg()
        mock_neo4j.get_segments_by_topic.return_value = [
            {"segment_id": "s1", "video_id": "v1", "start_s": 0, "end_s": 5,
             "transcript": "Hello", "embedding_id": "e1"},
        ]
        mock_neo4j.get_related_topics.return_value = []

        results = await kg.query_segments_by_topic("machine_learning")
        assert len(results) == 1
        assert results[0]["segment_id"] == "s1"

    @pytest.mark.asyncio
    async def test_query_segments_expands_related(self) -> None:
        kg, mock_neo4j = self._make_kg()
        mock_neo4j.get_segments_by_topic.side_effect = [
            [{"segment_id": "s1", "video_id": "v1", "start_s": 0, "end_s": 5,
              "transcript": "ML", "embedding_id": "e1"}],
            [{"segment_id": "s2", "video_id": "v1", "start_s": 10, "end_s": 15,
              "transcript": "DL", "embedding_id": "e2"}],
        ]
        mock_neo4j.get_related_topics.return_value = ["deep_learning"]

        results = await kg.query_segments_by_topic("machine_learning", expand_related=True)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_segments_deduplicates(self) -> None:
        kg, mock_neo4j = self._make_kg()
        mock_neo4j.get_segments_by_topic.side_effect = [
            [{"segment_id": "s1", "video_id": "v1", "start_s": 0, "end_s": 5,
              "transcript": "AI", "embedding_id": "e1"}],
            [{"segment_id": "s1", "video_id": "v1", "start_s": 0, "end_s": 5,
              "transcript": "AI", "embedding_id": "e1"}],  # duplicate
        ]
        mock_neo4j.get_related_topics.return_value = ["ai"]

        results = await kg.query_segments_by_topic("artificial_intelligence")
        assert len(results) == 1  # deduplicated
