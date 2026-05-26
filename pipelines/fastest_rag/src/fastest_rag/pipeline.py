"""NaiveRAGPipeline — baseline RAG: embed query → retrieve → generate.

Flow:
1. Check Redis semantic cache (if use_cache=True).
2. If miss: embed query via EmbeddingService.
3. Retrieve top-k chunks from vector store (pgvector by default).
4. Build context prompt with retrieved chunks.
5. Generate answer using claude-sonnet-4-6.
6. Store result in cache.
7. Return QueryResponse.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from shared.embeddings.service import EmbeddingService
from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse
from shared.models.retrieval import RetrievalResult
from shared.storage.vector_store import VectorStoreClient

from fastest_rag.cache_layer import CacheLayer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question using ONLY the provided context \
chunks. If the context does not contain enough information to answer, say so clearly. \
Be concise and factual. Cite which chunk(s) support each claim by referencing [Chunk N].\
"""

_CONTEXT_TEMPLATE = """\
<context>
{chunks}
</context>

Question: {question}

Answer based only on the context above:"""


class NaiveRAGPipeline:
    """Baseline RAG pipeline: embed → retrieve → generate.

    Args:
        vector_store: Connected VectorStoreClient (pgvector or Qdrant).
        embedding_service: Connected EmbeddingService.
        cache_layer: Optional CacheLayer for semantic response caching.
        anthropic_api_key: Key for Claude generation.
        model: Claude model ID for generation.
        max_context_chunks: Cap on how many chunks to include in prompt.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        cache_layer: CacheLayer | None = None,
        anthropic_api_key: str = "",
        model: str = "claude-sonnet-4-6",
        max_context_chunks: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._cache_layer = cache_layer
        self._anthropic_api_key = anthropic_api_key
        self._model = model
        self._max_context_chunks = max_context_chunks
        self._client: Any = None

    def connect(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)
        logger.info("NaiveRAGPipeline ready (model=%s)", self._model)

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the full RAG pipeline for a single query."""
        start = time.perf_counter()

        # 1. Cache check
        if request.use_cache and self._cache_layer:
            cached = await self._cache_layer.get(request.query)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.debug("Cache HIT for query (%.1fms)", elapsed_ms)
                return QueryResponse(
                    query=request.query,
                    answer=cached["answer"],
                    pipeline=PipelineStrategy.FASTEST_RAG,
                    sources=[RetrievalResult(**s) for s in cached.get("sources", [])],
                    latency_ms=elapsed_ms,
                    cache_hit=True,
                    metadata=cached.get("metadata", {}),
                )

        # 2. Embed query
        query_embedding = await self._embedding_service.embed(request.query)

        # 3. Retrieve top-k
        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=min(request.top_k, self._max_context_chunks),
            filters=request.filters or None,
        )

        # 4. Build context
        context_str = self._build_context(results)

        # 5. Generate answer
        answer = await self._generate(request.query, context_str)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("NaiveRAG query completed (%.1fms, %d sources)", elapsed_ms, len(results))

        response = QueryResponse(
            query=request.query,
            answer=answer,
            pipeline=PipelineStrategy.FASTEST_RAG,
            sources=results,
            latency_ms=elapsed_ms,
            cache_hit=False,
            metadata={
                "model": self._model,
                "embedding_tokens": self._embedding_service.total_tokens_used,
                "embedding_cost_usd": self._embedding_service.total_cost_usd,
            },
        )

        # 6. Store in cache
        if request.use_cache and self._cache_layer:
            await self._cache_layer.set(
                query=request.query,
                embedding=query_embedding,
                response={
                    "answer": answer,
                    "sources": [s.model_dump() for s in results],
                    "metadata": response.metadata,
                },
            )

        return response

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Chunk {i}] (score={r.score:.3f}, doc={r.document_id})\n{r.content}")
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
