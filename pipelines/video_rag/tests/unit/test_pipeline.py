"""Unit tests for VideoRAGPipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse
from video_rag.pipeline import VideoRAGPipeline
from video_rag.segment_retriever import VideoSegmentResult


class TestVideoRAGPipeline:
    def _make_pipeline(self) -> VideoRAGPipeline:
        return VideoRAGPipeline(
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
            neo4j_client=None,
            anthropic_api_key="sk-test",
            clip_model="ViT-B-32",
            clip_device="cpu",
        )

    def test_init(self) -> None:
        pipeline = self._make_pipeline()
        assert pipeline._generation_model == "claude-sonnet-4-6"
        assert pipeline._text_weight == 0.6
        assert pipeline._visual_weight == 0.4

    def test_format_ts(self) -> None:
        assert VideoRAGPipeline._format_ts(0.0) == "00:00"
        assert VideoRAGPipeline._format_ts(65.0) == "01:05"
        assert VideoRAGPipeline._format_ts(3661.0) == "61:01"

    @patch("video_rag.pipeline.anthropic")
    def test_connect(self, mock_anthropic: MagicMock) -> None:
        pipeline = self._make_pipeline()
        pipeline._clip_embedder = MagicMock()

        pipeline.connect()

        assert pipeline._retriever is not None
        assert pipeline._llm_client is not None

    @pytest.mark.asyncio
    async def test_run_returns_query_response(self) -> None:
        pipeline = self._make_pipeline()

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = [
            VideoSegmentResult(
                segment_id="s1",
                video_id="v1",
                start_ts=10.0,
                end_ts=15.0,
                transcript="Machine learning is a subfield of AI",
                text_score=0.9,
                visual_score=0.7,
                fused_score=0.82,
            ),
        ]
        pipeline._retriever = mock_retriever

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ML is a branch of artificial intelligence.")]
        mock_llm.messages.create.return_value = mock_response
        pipeline._llm_client = mock_llm

        request = QueryRequest(query="What is machine learning?", pipeline="video_rag")
        response = await pipeline.run(request)

        assert isinstance(response, QueryResponse)
        assert response.pipeline == PipelineStrategy.VIDEO_RAG
        assert response.answer == "ML is a branch of artificial intelligence."
        assert len(response.sources) == 1
        assert response.sources[0].metadata["start_ts"] == 10.0
        assert response.latency_ms is not None
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_run_raises_without_connect(self) -> None:
        pipeline = self._make_pipeline()
        request = QueryRequest(query="test", pipeline="video_rag")
        with pytest.raises(RuntimeError, match="Call connect"):
            await pipeline.run(request)

    @pytest.mark.asyncio
    async def test_run_with_video_filter(self) -> None:
        pipeline = self._make_pipeline()
        pipeline._retriever = AsyncMock()
        pipeline._retriever.retrieve.return_value = []

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="No results found.")]
        mock_llm.messages.create.return_value = mock_response
        pipeline._llm_client = mock_llm

        request = QueryRequest(
            query="test",
            pipeline="video_rag",
            filters={"video_id": "v1"},
        )
        await pipeline.run(request)

        pipeline._retriever.retrieve.assert_called_once_with(
            query="test",
            top_k=5,
            video_id="v1",
        )
