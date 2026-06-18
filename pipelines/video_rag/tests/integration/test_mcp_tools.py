"""Integration test: MCP server responds to tool calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from video_rag import mcp_server
from video_rag.mcp_server import (
    create_mcp_server,
    get_segment,
    get_transcript,
    list_videos,
    search_video,
)
from video_rag.segment_retriever import VideoSegmentResult


@pytest.mark.integration
class TestMCPToolIntegration:
    """Verify MCP tool calls work end-to-end with mock backends."""

    def setup_method(self) -> None:
        """Wire up mock retriever and knowledge graph."""
        self.mock_retriever = AsyncMock()
        self.mock_neo4j = AsyncMock()
        self.mock_kg = MagicMock()
        self.mock_kg._neo4j = self.mock_neo4j

        self.registry = {
            "vid-001": {
                "title": "AI Lecture 1",
                "url": "https://youtube.com/watch?v=abc",
                "duration_s": 600.0,
                "segment_count": 25,
            },
            "vid-002": {
                "title": "ML Workshop",
                "url": "https://youtube.com/watch?v=def",
                "duration_s": 1200.0,
                "segment_count": 50,
            },
        }

        create_mcp_server(self.mock_retriever, self.mock_kg, self.registry)

    def teardown_method(self) -> None:
        mcp_server._retriever = None
        mcp_server._knowledge_graph = None
        mcp_server._video_registry = {}

    @pytest.mark.asyncio
    async def test_search_video_tool(self) -> None:
        self.mock_retriever.retrieve.return_value = [
            VideoSegmentResult(
                segment_id="s1", video_id="vid-001",
                start_ts=120.0, end_ts=130.0,
                transcript="Neural networks learn representations",
                text_score=0.93, visual_score=0.78, fused_score=0.87,
            ),
            VideoSegmentResult(
                segment_id="s2", video_id="vid-002",
                start_ts=450.0, end_ts=460.0,
                transcript="Backpropagation computes gradients",
                text_score=0.85, visual_score=0.65, fused_score=0.77,
            ),
        ]

        results = await search_video("how do neural networks learn?", top_k=5)

        assert len(results) == 2
        assert results[0]["video_id"] == "vid-001"
        assert results[0]["start_ts"] == 120.0
        assert results[0]["fused_score"] == 0.87

    @pytest.mark.asyncio
    async def test_list_videos_tool(self) -> None:
        results = await list_videos()
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "AI Lecture 1" in titles
        assert "ML Workshop" in titles

    @pytest.mark.asyncio
    async def test_get_transcript_tool(self) -> None:
        self.mock_neo4j.get_video_segments.return_value = [
            {"start_s": 0.0, "end_s": 10.0, "transcript": "Intro to AI"},
            {"start_s": 10.0, "end_s": 20.0, "transcript": "What is ML?"},
            {"start_s": 20.0, "end_s": 30.0, "transcript": "Supervised learning"},
        ]

        result = await get_transcript("vid-001")

        assert result["video_id"] == "vid-001"
        assert result["title"] == "AI Lecture 1"
        assert len(result["segments"]) == 3
        assert result["segments"][0]["transcript"] == "Intro to AI"

    @pytest.mark.asyncio
    async def test_get_segment_tool(self) -> None:
        self.mock_neo4j.get_video_segments.return_value = [
            {"segment_id": "s5", "start_s": 100.0, "end_s": 110.0,
             "transcript": "Gradient descent optimizes the loss function"},
        ]

        result = await get_segment("s5", "vid-001")
        assert result["transcript"] == "Gradient descent optimizes the loss function"
        assert result["start_ts"] == 100.0
