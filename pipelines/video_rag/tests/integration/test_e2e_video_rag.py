"""E2E integration test: full pipeline query with mock services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse

from video_rag.pipeline import VideoRAGPipeline
from video_rag.segment_retriever import VideoSegmentResult


@pytest.mark.integration
class TestE2EVideoRAG:
    """End-to-end test: query via VideoRAGPipeline, verify response structure."""

    @pytest.mark.asyncio
    async def test_full_query_pipeline(self) -> None:
        """Simulate a complete query through the pipeline."""
        pipeline = VideoRAGPipeline(
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
            neo4j_client=None,
            anthropic_api_key="sk-test",
        )

        # Mock CLIP embedder
        mock_clip = MagicMock()
        mock_clip.embed_text.return_value = [0.1] * 512
        pipeline._clip_embedder = mock_clip

        # Mock retriever
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = [
            VideoSegmentResult(
                segment_id="s1", video_id="lecture-ai-101",
                start_ts=120.5, end_ts=135.0,
                transcript="Machine learning is a subset of artificial intelligence that enables systems to learn from data",
                text_score=0.94, visual_score=0.72, fused_score=0.85,
            ),
            VideoSegmentResult(
                segment_id="s2", video_id="lecture-ai-101",
                start_ts=200.0, end_ts=215.0,
                transcript="Supervised learning uses labeled data to train models",
                text_score=0.88, visual_score=0.65, fused_score=0.79,
            ),
            VideoSegmentResult(
                segment_id="s3", video_id="workshop-ml-basics",
                start_ts=45.0, end_ts=60.0,
                transcript="In this workshop we will build a classifier from scratch",
                text_score=0.82, visual_score=0.60, fused_score=0.73,
            ),
        ]
        pipeline._retriever = mock_retriever

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text="Machine learning is a branch of AI that allows systems to learn from data. "
            "At [02:00] in the AI lecture, the instructor explains that ML enables systems "
            "to improve through experience."
        )]
        mock_llm.messages.create.return_value = mock_response
        pipeline._llm_client = mock_llm

        # Execute query
        request = QueryRequest(
            query="What is machine learning?",
            pipeline="video_rag",
            top_k=5,
        )
        response = await pipeline.run(request)

        # Verify response structure
        assert isinstance(response, QueryResponse)
        assert response.pipeline == PipelineStrategy.VIDEO_RAG
        assert "machine learning" in response.answer.lower() or "ML" in response.answer
        assert response.latency_ms > 0
        assert not response.cache_hit

        # Verify sources
        assert len(response.sources) == 3
        assert response.sources[0].chunk_id == "s1"
        assert response.sources[0].document_id == "lecture-ai-101"
        assert response.sources[0].score == 0.85
        assert response.sources[0].metadata["start_ts"] == 120.5
        assert response.sources[0].metadata["end_ts"] == 135.0

        # Verify metadata
        assert response.metadata["generation_model"] == "claude-sonnet-4-6"
        assert response.metadata["segments_retrieved"] == 3
        assert "lecture-ai-101" in response.metadata["video_ids"]
        assert "workshop-ml-basics" in response.metadata["video_ids"]

    @pytest.mark.asyncio
    async def test_query_with_video_filter(self) -> None:
        """Verify that video_id filter is passed through correctly."""
        pipeline = VideoRAGPipeline(
            vector_store=AsyncMock(),
            embedding_service=AsyncMock(),
            anthropic_api_key="sk-test",
        )

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = [
            VideoSegmentResult(
                segment_id="s1", video_id="specific-video",
                start_ts=0.0, end_ts=10.0,
                transcript="Only from this video",
                text_score=0.9, visual_score=0.7, fused_score=0.82,
            ),
        ]
        pipeline._retriever = mock_retriever

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Answer from specific video.")]
        mock_llm.messages.create.return_value = mock_response
        pipeline._llm_client = mock_llm

        request = QueryRequest(
            query="test", pipeline="video_rag", filters={"video_id": "specific-video"},
        )
        response = await pipeline.run(request)

        mock_retriever.retrieve.assert_called_once_with(
            query="test", top_k=5, video_id="specific-video",
        )
        assert response.metadata["video_ids"] == ["specific-video"]
