"""Integration test: ingest a sample video, retrieve segments, verify timestamps."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from shared.models.document import ChunkType
from video_rag.scene_detector import Keyframe
from video_rag.segment_retriever import SegmentRetriever
from video_rag.timestamp_chunker import TimestampChunker
from video_rag.video_indexer import TranscriptSegment


@pytest.mark.integration
class TestVideoIngestPipeline:
    """Full ingestion pipeline: transcribe → detect scenes → embed → chunk → retrieve."""

    @pytest.mark.asyncio
    async def test_ingest_and_retrieve(self) -> None:
        """Verify end-to-end: segments are chunked and retrievable with correct timestamps."""
        # 1. Simulate transcription output
        segments = [
            TranscriptSegment(start_ts=0.0, end_ts=5.0, text="Welcome to our AI lecture"),
            TranscriptSegment(start_ts=5.0, end_ts=10.0, text="Today we discuss neural networks"),
            TranscriptSegment(start_ts=10.0, end_ts=15.0, text="Deep learning is transformative"),
            TranscriptSegment(start_ts=15.0, end_ts=20.0, text="Let us look at convolutional nets"),
            TranscriptSegment(start_ts=20.0, end_ts=25.0, text="CNNs process visual data"),
        ]

        # 2. Simulate keyframes at scene changes
        keyframes = [
            Keyframe(timestamp_s=0.0, frame_index=0, frame=np.zeros((100, 100, 3), dtype=np.uint8)),
            Keyframe(
                timestamp_s=10.0,
                frame_index=300,
                frame=np.ones((100, 100, 3), dtype=np.uint8) * 128,
            ),
            Keyframe(
                timestamp_s=20.0,
                frame_index=600,
                frame=np.ones((100, 100, 3), dtype=np.uint8) * 255,
            ),
        ]

        # 3. Create fake CLIP embeddings for keyframes
        kf_embeddings = [list(np.random.randn(512).astype(float)) for _ in keyframes]

        # 4. Map segments to nearest keyframe embeddings
        chunker = TimestampChunker()
        frame_mapping = chunker.assign_frame_embeddings(
            segments=segments,
            keyframe_timestamps=[kf.timestamp_s for kf in keyframes],
            keyframe_embeddings=kf_embeddings,
        )

        # 5. Create document chunks
        chunks = chunker.chunks_from_segments(
            video_id="test-video-001",
            segments=segments,
            frame_embeddings=frame_mapping,
        )

        assert len(chunks) == 5
        assert all(c.chunk_type == ChunkType.VIDEO_TRANSCRIPT for c in chunks)
        assert all("start_ts" in c.metadata for c in chunks)
        assert all("end_ts" in c.metadata for c in chunks)

        # Verify timestamps are within 1 second of originals
        for chunk, seg in zip(chunks, segments, strict=True):
            assert abs(chunk.metadata["start_ts"] - seg.start_ts) < 1.0
            assert abs(chunk.metadata["end_ts"] - seg.end_ts) < 1.0

        # Verify frame embeddings were assigned
        assert all("frame_embedding" in c.metadata for c in chunks)

        # 6. Mock vector store to return our chunks
        mock_vs = AsyncMock()
        mock_vs.search.return_value = [
            {
                "chunk_id": chunks[2].id,
                "content": chunks[2].content,
                "score": 0.92,
                "metadata": chunks[2].metadata,
            },
            {
                "chunk_id": chunks[3].id,
                "content": chunks[3].content,
                "score": 0.85,
                "metadata": chunks[3].metadata,
            },
        ]

        mock_emb = AsyncMock()
        mock_emb.embed.return_value = [0.1] * 3072

        mock_clip = MagicMock()
        mock_clip.embed_text.return_value = [0.1] * 512

        retriever = SegmentRetriever(
            vector_store=mock_vs,
            embedding_service=mock_emb,
            clip_embedder=mock_clip,
        )

        results = await retriever.retrieve("deep learning transformative", top_k=2)

        assert len(results) == 2
        # Timestamps should match the original segments
        assert results[0].start_ts == 10.0 or results[0].start_ts == 15.0
