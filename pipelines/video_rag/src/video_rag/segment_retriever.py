"""SegmentRetriever — dual text+visual retrieval with fused ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import VectorStoreClient

    from video_rag.clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoSegmentResult:
    """A retrieved video segment with fused score."""

    segment_id: str
    video_id: str
    start_ts: float
    end_ts: float
    transcript: str
    text_score: float
    visual_score: float
    fused_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SegmentRetriever:
    """Retrieve video segments using fused text + visual similarity.

    Performs two parallel retrievals:
    1. Text similarity: query embedding vs transcript embeddings (OpenAI).
    2. Visual similarity: query CLIP embedding vs keyframe CLIP embeddings.

    Results are fused using a weighted linear combination.

    Args:
        vector_store: PgVector client for transcript embeddings.
        embedding_service: OpenAI embedding service for text queries.
        clip_embedder: CLIP model for visual query embeddings.
        text_weight: Weight for text similarity score (default 0.6).
        visual_weight: Weight for visual similarity score (default 0.4).
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        clip_embedder: CLIPEmbedder,
        text_weight: float = 0.6,
        visual_weight: float = 0.4,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._clip_embedder = clip_embedder
        self._text_weight = text_weight
        self._visual_weight = visual_weight

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        video_id: str | None = None,
    ) -> list[VideoSegmentResult]:
        """Retrieve top-k video segments for a query using fused ranking.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            video_id: Optional filter to restrict to a single video.

        Returns:
            List of VideoSegmentResult sorted by fused_score descending.
        """
        # 1. Get text embedding for the query
        text_embedding = await self._embedding_service.embed(query)

        # 2. Get CLIP text embedding for visual matching
        clip_embedding = self._clip_embedder.embed_text(query)

        # 3. Text retrieval: search transcript embeddings
        filters = {"chunk_type": "video_transcript"}
        if video_id:
            filters["video_id"] = video_id

        text_results = await self._vector_store.search(
            embedding=text_embedding,
            top_k=top_k * 3,  # over-fetch for fusion
            filters=filters,
        )

        # 4. Normalise results to dicts (PgVectorClient returns RetrievalResult objects)
        text_scores: dict[str, float] = {}
        segment_data: dict[str, dict[str, Any]] = {}
        for result in text_results:
            if isinstance(result, dict):
                row = result
            else:
                # Pydantic model (RetrievalResult)
                row = {
                    "chunk_id": getattr(result, "chunk_id", ""),
                    "content": getattr(result, "content", ""),
                    "score": getattr(result, "score", 0.0),
                    "metadata": getattr(result, "metadata", {}),
                }
            sid = row.get("chunk_id", row.get("id", ""))
            text_scores[sid] = row.get("score", 0.0)
            segment_data[sid] = row

        # 5. Compute visual scores for each candidate
        visual_scores: dict[str, float] = {}
        for sid, data in segment_data.items():
            frame_emb = data.get("metadata", {}).get("frame_embedding")
            if frame_emb:
                visual_scores[sid] = self._cosine_similarity(clip_embedding, frame_emb)
            else:
                visual_scores[sid] = 0.0

        # 6. Fused ranking
        fused: list[VideoSegmentResult] = []
        for sid in segment_data:
            ts = text_scores.get(sid, 0.0)
            vs = visual_scores.get(sid, 0.0)
            fs = self._text_weight * ts + self._visual_weight * vs
            meta = segment_data[sid].get("metadata", {})

            fused.append(
                VideoSegmentResult(
                    segment_id=sid,
                    video_id=meta.get("video_id", ""),
                    start_ts=meta.get("start_ts", 0.0),
                    end_ts=meta.get("end_ts", 0.0),
                    transcript=segment_data[sid].get("content", ""),
                    text_score=ts,
                    visual_score=vs,
                    fused_score=fs,
                    metadata=meta,
                )
            )

        fused.sort(key=lambda r: r.fused_score, reverse=True)
        return fused[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm == 0:
            return 0.0
        return float(dot / norm)
