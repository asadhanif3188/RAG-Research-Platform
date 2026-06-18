"""Unit tests for MCP server tools."""

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


class TestMCPServer:
    def setup_method(self) -> None:
        """Reset module-level state before each test."""
        mcp_server._retriever = None
        mcp_server._knowledge_graph = None
        mcp_server._video_registry = {}

    def test_create_mcp_server_sets_globals(self) -> None:
        mock_retriever = MagicMock()
        mock_kg = MagicMock()
        registry = {"v1": {"title": "Test", "url": "http://test.com"}}

        create_mcp_server(mock_retriever, mock_kg, registry)

        assert mcp_server._retriever is mock_retriever
        assert mcp_server._knowledge_graph is mock_kg
        assert mcp_server._video_registry == registry

    @pytest.mark.asyncio
    async def test_search_video_returns_results(self) -> None:
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = [
            VideoSegmentResult(
                segment_id="s1",
                video_id="v1",
                start_ts=0.0,
                end_ts=5.0,
                transcript="Hello",
                text_score=0.9,
                visual_score=0.7,
                fused_score=0.82,
            ),
        ]
        mcp_server._retriever = mock_retriever

        results = await search_video("test query", top_k=3)
        assert len(results) == 1
        assert results[0]["segment_id"] == "s1"
        assert results[0]["fused_score"] == 0.82

    @pytest.mark.asyncio
    async def test_search_video_no_retriever(self) -> None:
        mcp_server._retriever = None
        results = await search_video("test")
        assert results[0]["error"] == "Retriever not initialised"

    @pytest.mark.asyncio
    async def test_list_videos(self) -> None:
        mcp_server._video_registry = {
            "v1": {
                "title": "Video 1",
                "url": "http://v1.com",
                "duration_s": 120,
                "segment_count": 5,
            },
            "v2": {
                "title": "Video 2",
                "url": "http://v2.com",
                "duration_s": 300,
                "segment_count": 20,
            },
        }
        results = await list_videos()
        assert len(results) == 2
        assert results[0]["video_id"] == "v1"

    @pytest.mark.asyncio
    async def test_get_segment_found(self) -> None:
        mock_kg = MagicMock()
        mock_kg._neo4j = AsyncMock()
        mock_kg._neo4j.get_video_segments.return_value = [
            {"segment_id": "s1", "start_s": 10.0, "end_s": 15.0, "transcript": "Found"},
        ]
        mcp_server._knowledge_graph = mock_kg

        result = await get_segment("s1", "v1")
        assert result["transcript"] == "Found"

    @pytest.mark.asyncio
    async def test_get_segment_not_found(self) -> None:
        mock_kg = MagicMock()
        mock_kg._neo4j = AsyncMock()
        mock_kg._neo4j.get_video_segments.return_value = []
        mcp_server._knowledge_graph = mock_kg

        result = await get_segment("missing", "v1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_transcript(self) -> None:
        mock_kg = MagicMock()
        mock_kg._neo4j = AsyncMock()
        mock_kg._neo4j.get_video_segments.return_value = [
            {"start_s": 0, "end_s": 5, "transcript": "Hello"},
            {"start_s": 5, "end_s": 10, "transcript": "World"},
        ]
        mcp_server._knowledge_graph = mock_kg
        mcp_server._video_registry = {"v1": {"title": "Test Video"}}

        result = await get_transcript("v1")
        assert len(result["segments"]) == 2
        assert result["title"] == "Test Video"
