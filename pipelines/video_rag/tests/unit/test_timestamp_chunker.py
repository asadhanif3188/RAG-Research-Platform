"""Unit tests for TimestampChunker."""

from __future__ import annotations

from shared.models.document import ChunkType

from video_rag.timestamp_chunker import TimestampChunker
from video_rag.video_indexer import TranscriptSegment


class TestTimestampChunker:
    def _make_segments(self) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(start_ts=0.0, end_ts=3.0, text="Welcome to the lecture"),
            TranscriptSegment(start_ts=3.0, end_ts=6.0, text="Today we cover AI"),
            TranscriptSegment(start_ts=6.0, end_ts=10.0, text="Let us begin"),
        ]

    def test_chunks_from_segments_basic(self) -> None:
        chunker = TimestampChunker()
        segments = self._make_segments()

        chunks = chunker.chunks_from_segments(video_id="v1", segments=segments)

        assert len(chunks) == 3
        assert chunks[0].chunk_type == ChunkType.VIDEO_TRANSCRIPT
        assert chunks[0].document_id == "v1"
        assert chunks[0].content == "Welcome to the lecture"
        assert chunks[0].metadata["start_ts"] == 0.0
        assert chunks[0].metadata["end_ts"] == 3.0

    def test_chunks_include_frame_embeddings(self) -> None:
        chunker = TimestampChunker()
        segments = self._make_segments()
        frame_embeddings = {0: [0.1] * 512, 2: [0.3] * 512}

        chunks = chunker.chunks_from_segments(
            video_id="v1", segments=segments, frame_embeddings=frame_embeddings,
        )

        assert "frame_embedding" in chunks[0].metadata
        assert len(chunks[0].metadata["frame_embedding"]) == 512
        assert "frame_embedding" not in chunks[1].metadata
        assert "frame_embedding" in chunks[2].metadata

    def test_chunks_have_unique_ids(self) -> None:
        chunker = TimestampChunker()
        segments = self._make_segments()
        chunks = chunker.chunks_from_segments(video_id="v1", segments=segments)

        ids = [c.id for c in chunks]
        assert len(set(ids)) == len(ids)

    def test_assign_frame_embeddings_nearest(self) -> None:
        chunker = TimestampChunker()
        segments = [
            TranscriptSegment(start_ts=0.0, end_ts=5.0, text="seg1"),   # mid=2.5
            TranscriptSegment(start_ts=5.0, end_ts=10.0, text="seg2"),  # mid=7.5
            TranscriptSegment(start_ts=10.0, end_ts=15.0, text="seg3"), # mid=12.5
        ]
        kf_timestamps = [1.0, 8.0, 14.0]
        kf_embeddings = [[0.1] * 512, [0.2] * 512, [0.3] * 512]

        mapping = chunker.assign_frame_embeddings(segments, kf_timestamps, kf_embeddings)

        assert mapping[0] == [0.1] * 512  # seg1 mid=2.5 closest to kf@1.0
        assert mapping[1] == [0.2] * 512  # seg2 mid=7.5 closest to kf@8.0
        assert mapping[2] == [0.3] * 512  # seg3 mid=12.5 closest to kf@14.0

    def test_assign_frame_embeddings_empty(self) -> None:
        chunker = TimestampChunker()
        segments = [TranscriptSegment(start_ts=0, end_ts=1, text="test")]
        mapping = chunker.assign_frame_embeddings(segments, [], [])
        assert mapping == {}
