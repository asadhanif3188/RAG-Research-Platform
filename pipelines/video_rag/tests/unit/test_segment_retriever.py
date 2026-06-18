"""Unit tests for SegmentRetriever — fused ranking logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from video_rag.segment_retriever import SegmentRetriever, VideoSegmentResult


class TestVideoSegmentResult:
    def test_frozen_dataclass(self) -> None:
        result = VideoSegmentResult(
            segment_id="s1",
            video_id="v1",
            start_ts=0.0,
            end_ts=5.0,
            transcript="Hello",
            text_score=0.9,
            visual_score=0.7,
            fused_score=0.82,
        )
        assert result.fused_score == 0.82


class TestSegmentRetriever:
    def _make_retriever(
        self,
        text_weight: float = 0.6,
        visual_weight: float = 0.4,
    ) -> tuple[SegmentRetriever, AsyncMock, AsyncMock, MagicMock]:
        mock_vs = AsyncMock()
        mock_emb = AsyncMock()
        mock_clip = MagicMock()

        retriever = SegmentRetriever(
            vector_store=mock_vs,
            embedding_service=mock_emb,
            clip_embedder=mock_clip,
            text_weight=text_weight,
            visual_weight=visual_weight,
        )
        return retriever, mock_vs, mock_emb, mock_clip

    @pytest.mark.asyncio
    async def test_retrieve_fuses_scores(self) -> None:
        retriever, mock_vs, mock_emb, mock_clip = self._make_retriever()

        mock_emb.embed.return_value = [0.1] * 3072
        mock_clip.embed_text.return_value = [0.1] * 512

        # Simulate vector store results
        mock_vs.search.return_value = [
            {
                "chunk_id": "s1",
                "content": "Machine learning basics",
                "score": 0.95,
                "metadata": {
                    "video_id": "v1",
                    "start_ts": 10.0,
                    "end_ts": 15.0,
                    "frame_embedding": [0.2] * 512,
                },
            },
            {
                "chunk_id": "s2",
                "content": "Deep learning intro",
                "score": 0.80,
                "metadata": {
                    "video_id": "v1",
                    "start_ts": 30.0,
                    "end_ts": 35.0,
                    "frame_embedding": [0.3] * 512,
                },
            },
        ]

        results = await retriever.retrieve("what is machine learning?", top_k=2)

        assert len(results) == 2
        # Both should have fused scores
        for r in results:
            assert r.fused_score > 0.0
            assert r.text_score > 0.0
        # Results should be sorted by fused_score descending
        assert results[0].fused_score >= results[1].fused_score

    @pytest.mark.asyncio
    async def test_retrieve_with_video_filter(self) -> None:
        retriever, mock_vs, mock_emb, mock_clip = self._make_retriever()

        mock_emb.embed.return_value = [0.1] * 3072
        mock_clip.embed_text.return_value = [0.1] * 512
        mock_vs.search.return_value = []

        await retriever.retrieve("query", video_id="v1")

        # Verify filters were passed
        call_args = mock_vs.search.call_args
        filters = call_args.kwargs.get("filters", {})
        assert filters.get("video_id") == "v1"

    @pytest.mark.asyncio
    async def test_retrieve_no_frame_embedding(self) -> None:
        """Segments without frame embeddings should get visual_score=0."""
        retriever, mock_vs, mock_emb, mock_clip = self._make_retriever()

        mock_emb.embed.return_value = [0.1] * 3072
        mock_clip.embed_text.return_value = [0.1] * 512
        mock_vs.search.return_value = [
            {
                "chunk_id": "s1",
                "content": "No visual",
                "score": 0.9,
                "metadata": {"video_id": "v1", "start_ts": 0.0, "end_ts": 5.0},
            },
        ]

        results = await retriever.retrieve("query", top_k=1)
        assert len(results) == 1
        assert results[0].visual_score == 0.0
        assert results[0].fused_score == 0.6 * 0.9  # text_weight * text_score

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self) -> None:
        retriever, mock_vs, mock_emb, mock_clip = self._make_retriever()

        mock_emb.embed.return_value = [0.1] * 3072
        mock_clip.embed_text.return_value = [0.1] * 512
        mock_vs.search.return_value = []

        results = await retriever.retrieve("obscure query")
        assert results == []

    def test_cosine_similarity(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert SegmentRetriever._cosine_similarity(a, b) == pytest.approx(1.0)

        c = [0.0, 1.0, 0.0]
        assert SegmentRetriever._cosine_similarity(a, c) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert SegmentRetriever._cosine_similarity(a, b) == 0.0

    @pytest.mark.asyncio
    async def test_fused_ranking_weights(self) -> None:
        """Verify that text_weight and visual_weight correctly influence ranking."""
        # High text weight
        retriever, mock_vs, mock_emb, mock_clip = self._make_retriever(
            text_weight=0.9, visual_weight=0.1
        )
        mock_emb.embed.return_value = [0.1] * 3072
        mock_clip.embed_text.return_value = [0.1] * 512

        # s1: high text, low visual. s2: low text, high visual
        frame_emb_1 = list(np.random.randn(512).astype(float))
        frame_emb_2 = list(np.random.randn(512).astype(float))
        mock_vs.search.return_value = [
            {
                "chunk_id": "s1",
                "content": "text heavy",
                "score": 0.95,
                "metadata": {
                    "video_id": "v1",
                    "start_ts": 0,
                    "end_ts": 5,
                    "frame_embedding": frame_emb_1,
                },
            },
            {
                "chunk_id": "s2",
                "content": "visual heavy",
                "score": 0.50,
                "metadata": {
                    "video_id": "v1",
                    "start_ts": 10,
                    "end_ts": 15,
                    "frame_embedding": frame_emb_2,
                },
            },
        ]

        results = await retriever.retrieve("query", top_k=2)
        # With 90% text weight, s1 (high text score) should rank first
        assert results[0].segment_id == "s1"
