"""MultimodalRAGPipeline — retrieves across text, image descriptions, and tables.

Flow:
1. Check Redis semantic cache (if use_cache=True).
2. Embed query.
3. Retrieve top-k chunks across TEXT, IMAGE_DESCRIPTION, and TABLE types.
   Uses per-type quotas (50% text / 30% image / 20% table) then merges by score.
4. Build type-annotated context prompt.
5. Generate answer using claude-sonnet-4-6.
6. Run ProvenanceTracker to attribute sentences to source chunks.
7. Store result in cache and return QueryResponse with provenance in metadata.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from multimodal_rag.provenance import ProvenanceTracker
from shared.models.document import ChunkType
from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse
from shared.models.retrieval import RetrievalResult

if TYPE_CHECKING:
    from fastest_rag.cache_layer import CacheLayer
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise research assistant with access to text passages, image descriptions, \
and table data extracted from documents. Answer the user's question using ONLY the \
provided context chunks. If the context does not contain enough information to answer, \
say so clearly. Be concise and factual. Cite chunk(s) supporting each claim as [Chunk N] \
and indicate the content type (Text/Image/Table) in parentheses.\
"""

_CONTEXT_TEMPLATE = """\
<context>
{chunks}
</context>

Question: {question}

Answer based only on the context above, citing [Chunk N] (Type) for each claim:"""

_CHUNK_TYPE_LABELS: dict[str, str] = {
    ChunkType.TEXT.value: "Text",
    ChunkType.IMAGE_DESCRIPTION.value: "Image",
    ChunkType.TABLE.value: "Table",
    ChunkType.VIDEO_TRANSCRIPT.value: "Video",
}


class MultimodalRAGPipeline:
    """RAG pipeline that retrieves from text, image description, and table chunks.

    Args:
        vector_store: Connected VectorStoreClient.
        embedding_service: Connected EmbeddingService.
        cache_layer: Optional semantic cache layer for response caching.
        anthropic_api_key: Anthropic API key for Claude generation.
        model: Claude model ID.
        max_context_chunks: Maximum total chunks included in the prompt.
        track_provenance: Whether to run ProvenanceTracker on generated answers.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        cache_layer: CacheLayer | None = None,
        anthropic_api_key: str = "",
        model: str = "claude-sonnet-4-6",
        max_context_chunks: int = 8,
        track_provenance: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._cache_layer = cache_layer
        self._anthropic_api_key = anthropic_api_key
        self._model = model
        self._max_context_chunks = max_context_chunks
        self._track_provenance = track_provenance
        self._provenance_tracker = ProvenanceTracker()
        self._client: Any = None

    def connect(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)
        logger.info("MultimodalRAGPipeline ready (model=%s)", self._model)

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute multimodal RAG: embed → retrieve all types → generate → provenance."""
        start = time.perf_counter()

        # 1. Cache check
        if request.use_cache and self._cache_layer:
            cached = await self._cache_layer.get(request.query)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.debug("Cache HIT for multimodal query (%.1fms)", elapsed_ms)
                return QueryResponse(
                    query=request.query,
                    answer=cached["answer"],
                    pipeline=PipelineStrategy.MULTIMODAL_RAG,
                    sources=[RetrievalResult(**s) for s in cached.get("sources", [])],
                    latency_ms=elapsed_ms,
                    cache_hit=True,
                    metadata=cached.get("metadata", {}),
                )

        # 2. Embed query
        query_embedding = await self._embedding_service.embed(request.query)

        # 3. Multi-type retrieval
        results = await self._retrieve_multimodal(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters or None,
        )

        # 4. Build type-annotated context
        context_str = self._build_context(results)

        # 5. Generate answer
        answer = await self._generate(request.query, context_str)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 6. Provenance tracking
        provenance: list[dict[str, Any]] = []
        if self._track_provenance and results:
            records = self._provenance_tracker.track(answer, results)
            provenance = [
                {
                    "sentence": r.sentence,
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "page_number": r.page_number,
                    "chunk_type": r.chunk_type,
                    "confidence": r.confidence,
                }
                for r in records
            ]

        metadata: dict[str, Any] = {
            "model": self._model,
            "provenance": provenance,
            "chunk_type_distribution": _type_distribution(results),
            "embedding_tokens": self._embedding_service.total_tokens_used,
            "embedding_cost_usd": self._embedding_service.total_cost_usd,
        }

        response = QueryResponse(
            query=request.query,
            answer=answer,
            pipeline=PipelineStrategy.MULTIMODAL_RAG,
            sources=results,
            latency_ms=elapsed_ms,
            cache_hit=False,
            metadata=metadata,
        )

        # 7. Store in cache
        if request.use_cache and self._cache_layer:
            await self._cache_layer.set(
                query=request.query,
                embedding=query_embedding,
                response={
                    "answer": answer,
                    "sources": [s.model_dump() for s in results],
                    "metadata": metadata,
                },
            )

        logger.info(
            "MultimodalRAG completed (%.1fms, %d sources, %d provenance records)",
            elapsed_ms,
            len(results),
            len(provenance),
        )
        return response

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _retrieve_multimodal(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Retrieve from each chunk type with per-type quotas, merge by score."""
        text_k = max(1, int(top_k * 0.5))
        image_k = max(1, int(top_k * 0.3))
        table_k = max(1, top_k - text_k - image_k)

        text_results = await self._search_by_type(query_embedding, text_k, ChunkType.TEXT, filters)
        image_results = await self._search_by_type(
            query_embedding, image_k, ChunkType.IMAGE_DESCRIPTION, filters
        )
        table_results = await self._search_by_type(
            query_embedding, table_k, ChunkType.TABLE, filters
        )

        # If type-filtered search returns nothing, fall back to unfiltered
        total = len(text_results) + len(image_results) + len(table_results)
        if total == 0:
            logger.debug("Type-filtered search returned no results, falling back to unfiltered")
            fallback = await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            return fallback[: self._max_context_chunks]

        # Deduplicate by chunk_id and sort by score
        seen: set[str] = set()
        merged: list[RetrievalResult] = []
        for r in text_results + image_results + table_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                merged.append(r)

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[: self._max_context_chunks]

    async def _search_by_type(
        self,
        query_embedding: list[float],
        top_k: int,
        chunk_type: ChunkType,
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Search with a chunk_type filter; returns [] on any error."""
        type_filter: dict[str, Any] = {"chunk_type": chunk_type.value}
        if filters:
            type_filter.update(filters)
        try:
            return await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=type_filter,
            )
        except Exception as exc:
            logger.warning("Type-filtered search failed (type=%s): %s", chunk_type.value, exc)
            return []

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            label = _CHUNK_TYPE_LABELS.get(r.chunk_type, "Text")
            page_info = f", page={r.metadata['page_number']}" if "page_number" in r.metadata else ""
            header = f"[Chunk {i}] ({label}, score={r.score:.3f}, doc={r.document_id}{page_info})"
            parts.append(f"{header}\n{r.content}")
        return "\n\n---\n\n".join(parts)

    async def _generate(self, question: str, context: str) -> str:
        user_message = _CONTEXT_TEMPLATE.format(chunks=context, question=question)
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text


def _type_distribution(results: list[RetrievalResult]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in results:
        dist[r.chunk_type] = dist.get(r.chunk_type, 0) + 1
    return dist
