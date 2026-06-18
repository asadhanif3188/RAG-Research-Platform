"""TimestampChunker — extend DocumentChunk with video timestamp metadata."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from shared.models.document import ChunkType, DocumentChunk

if TYPE_CHECKING:
    from video_rag.video_indexer import TranscriptSegment


class TimestampChunker:
    """Convert transcript segments into DocumentChunks with timestamp metadata.

    Each chunk carries start_ts, end_ts, and optionally a CLIP frame embedding
    in its metadata, enabling timestamp-level retrieval.
    """

    def chunks_from_segments(
        self,
        video_id: str,
        segments: list[TranscriptSegment],
        frame_embeddings: dict[int, list[float]] | None = None,
    ) -> list[DocumentChunk]:
        """Convert transcript segments into DocumentChunks.

        Args:
            video_id: Unique identifier for the source video.
            segments: Ordered transcript segments with timestamps.
            frame_embeddings: Optional mapping of segment_index → CLIP embedding
                for the nearest keyframe.

        Returns:
            List of DocumentChunks with VIDEO_TRANSCRIPT chunk_type.
        """
        chunks: list[DocumentChunk] = []
        frame_embeddings = frame_embeddings or {}

        for idx, seg in enumerate(segments):
            metadata: dict[str, object] = {
                "start_ts": seg.start_ts,
                "end_ts": seg.end_ts,
                "video_id": video_id,
                "segment_index": idx,
            }
            if idx in frame_embeddings:
                metadata["frame_embedding"] = frame_embeddings[idx]

            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=video_id,
                chunk_index=idx,
                chunk_type=ChunkType.VIDEO_TRANSCRIPT,
                content=seg.text,
                metadata=metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            chunks.append(chunk)

        return chunks

    def assign_frame_embeddings(
        self,
        segments: list[TranscriptSegment],
        keyframe_timestamps: list[float],
        keyframe_embeddings: list[list[float]],
    ) -> dict[int, list[float]]:
        """Map each segment to the nearest keyframe's CLIP embedding.

        For each segment, find the keyframe whose timestamp is closest to the
        segment's midpoint and assign that keyframe's embedding.

        Args:
            segments: Transcript segments.
            keyframe_timestamps: Timestamps of each keyframe.
            keyframe_embeddings: CLIP embeddings corresponding to each keyframe.

        Returns:
            Mapping of segment_index → CLIP embedding.
        """
        if not keyframe_timestamps or not keyframe_embeddings:
            return {}

        mapping: dict[int, list[float]] = {}
        for idx, seg in enumerate(segments):
            midpoint = (seg.start_ts + seg.end_ts) / 2.0
            closest_kf = min(
                range(len(keyframe_timestamps)),
                key=lambda k: abs(keyframe_timestamps[k] - midpoint),
            )
            mapping[idx] = keyframe_embeddings[closest_kf]

        return mapping
