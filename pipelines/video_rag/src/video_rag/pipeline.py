"""VideoRAGPipeline — wraps video retrieval components to match the pipeline interface."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import anthropic

from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse
from shared.models.retrieval import RetrievalResult

from video_rag.clip_embedder import CLIPEmbedder
from video_rag.knowledge_graph import KnowledgeGraph
from video_rag.segment_retriever import SegmentRetriever

if TYPE_CHECKING:
    from shared.embeddings.service import EmbeddingService
    from shared.storage.neo4j_client import Neo4jClient
    from shared.storage.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)


class VideoRAGPipeline:
    """MCP-powered RAG pipeline for video search with hybrid text+visual retrieval.

    Combines transcript text similarity (OpenAI embeddings) and visual frame
    similarity (CLIP embeddings) to find relevant video segments, then generates
    an answer using Claude.

    Args:
        vector_store: Connected VectorStoreClient for transcript embeddings.
        embedding_service: Connected EmbeddingService for text queries.
        neo4j_client: Connected Neo4jClient for topic graph.
        anthropic_api_key: Key for Claude API.
        clip_model: CLIP model name (default ViT-B-32).
        clip_device: Device for CLIP inference.
        generation_model: Claude model for answer generation.
        text_weight: Weight for text similarity in fused ranking.
        visual_weight: Weight for visual similarity in fused ranking.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        neo4j_client: Neo4jClient | None = None,
        anthropic_api_key: str = "",
        clip_model: str = "ViT-B-32",
        clip_device: str = "cpu",
        generation_model: str = "claude-sonnet-4-6",
        text_weight: float = 0.6,
        visual_weight: float = 0.4,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._neo4j_client = neo4j_client
        self._anthropic_api_key = anthropic_api_key
        self._generation_model = generation_model

        self._clip_embedder = CLIPEmbedder(
            model_name=clip_model, device=clip_device,
        )
        self._retriever: SegmentRetriever | None = None
        self._knowledge_graph: KnowledgeGraph | None = None
        self._llm_client: Any = None
        self._text_weight = text_weight
        self._visual_weight = visual_weight

    def connect(self) -> None:
        """Initialize all sub-components."""
        self._clip_embedder.connect()

        self._retriever = SegmentRetriever(
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            clip_embedder=self._clip_embedder,
            text_weight=self._text_weight,
            visual_weight=self._visual_weight,
        )

        if self._neo4j_client:
            self._knowledge_graph = KnowledgeGraph(
                neo4j_client=self._neo4j_client,
                anthropic_api_key=self._anthropic_api_key,
            )
            self._knowledge_graph.connect()

        self._llm_client = anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)
        logger.info("VideoRAGPipeline fully connected")

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the video RAG pipeline for a single query."""
        if self._retriever is None or self._llm_client is None:
            raise RuntimeError("Call connect() before run()")

        start = time.perf_counter()

        # 1. Retrieve relevant video segments
        video_id_filter = request.filters.get("video_id") if request.filters else None
        segments = await self._retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            video_id=video_id_filter,
        )

        # 2. Build context from retrieved segments
        context_parts: list[str] = []
        for i, seg in enumerate(segments, 1):
            ts_range = f"[{self._format_ts(seg.start_ts)} - {self._format_ts(seg.end_ts)}]"
            context_parts.append(
                f"Segment {i} (Video: {seg.video_id}) {ts_range}:\n{seg.transcript}"
            )
        context = "\n\n".join(context_parts)

        # 3. Generate answer with Claude
        answer = await self._generate_answer(request.query, context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 4. Build response
        sources = [
            RetrievalResult(
                chunk_id=seg.segment_id,
                document_id=seg.video_id,
                content=seg.transcript,
                score=seg.fused_score,
                metadata={
                    "start_ts": seg.start_ts,
                    "end_ts": seg.end_ts,
                    "text_score": seg.text_score,
                    "visual_score": seg.visual_score,
                },
            )
            for seg in segments
        ]

        return QueryResponse(
            query=request.query,
            answer=answer,
            pipeline=PipelineStrategy.VIDEO_RAG,
            sources=sources,
            latency_ms=elapsed_ms,
            cache_hit=False,
            metadata={
                "generation_model": self._generation_model,
                "text_weight": self._text_weight,
                "visual_weight": self._visual_weight,
                "segments_retrieved": len(segments),
                "video_ids": list({s.video_id for s in segments}),
            },
        )

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using Claude with retrieved video context."""
        system_prompt = (
            "You are a video research assistant. Answer questions using the provided "
            "video transcript segments. Reference specific timestamps when relevant. "
            "If the context doesn't contain enough information, say so."
        )

        response = await self._llm_client.messages.create(
            model=self._generation_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Video transcript segments:\n{context}"
                ),
            }],
        )
        return response.content[0].text

    @staticmethod
    def _format_ts(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
