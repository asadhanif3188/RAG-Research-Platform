"""FastAPI dependency injection for shared services.

Services are initialised once at startup and shared across requests
via module-level singletons injected through FastAPI's Depends mechanism.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends

from corrective_rag.pipeline import CRAGPipeline
from fastest_rag.cache_layer import CacheLayer
from fastest_rag.pipeline import NaiveRAGPipeline
from multimodal_rag.pipeline import MultimodalRAGPipeline
from self_rag.pipeline import SelfRAGPipeline
from shared.config import get_settings
from shared.embeddings.service import EmbeddingService
from shared.storage.neo4j_client import Neo4jClient
from shared.storage.vector_store import PgVectorClient
from video_rag.pipeline import VideoRAGPipeline

logger = logging.getLogger(__name__)

# ── Module-level singletons (initialised on first use) ────────────────────────

_embedding_service: EmbeddingService | None = None
_pgvector_client: PgVectorClient | None = None
_cache_layer: CacheLayer | None = None
_naive_pipeline: NaiveRAGPipeline | None = None
_multimodal_pipeline: MultimodalRAGPipeline | None = None
_crag_pipeline: CRAGPipeline | None = None
_self_rag_pipeline: SelfRAGPipeline | None = None
_video_rag_pipeline: VideoRAGPipeline | None = None
_neo4j_client: Neo4jClient | None = None


async def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        )
        _embedding_service.connect()
    return _embedding_service


async def get_pgvector_client() -> PgVectorClient:
    global _pgvector_client
    if _pgvector_client is None:
        settings = get_settings()
        _pgvector_client = PgVectorClient(database_url=settings.database_url)
        await _pgvector_client.connect()
    return _pgvector_client


async def get_cache_layer(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> CacheLayer:
    global _cache_layer
    if _cache_layer is None:
        settings = get_settings()
        _cache_layer = CacheLayer(
            redis_url=settings.redis_url,
            similarity_threshold=settings.semantic_cache_threshold,
            ttl=settings.redis_cache_ttl,
            embedding_service=embedding_service,
        )
        await _cache_layer.connect()
    return _cache_layer


async def get_naive_pipeline(
    vector_store: PgVectorClient = Depends(get_pgvector_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    cache_layer: CacheLayer = Depends(get_cache_layer),
) -> NaiveRAGPipeline:
    global _naive_pipeline
    if _naive_pipeline is None:
        settings = get_settings()
        _naive_pipeline = NaiveRAGPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            cache_layer=cache_layer,
            anthropic_api_key=settings.anthropic_api_key,
        )
        _naive_pipeline.connect()
    return _naive_pipeline


async def get_multimodal_pipeline(
    vector_store: PgVectorClient = Depends(get_pgvector_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    cache_layer: CacheLayer = Depends(get_cache_layer),
) -> MultimodalRAGPipeline:
    global _multimodal_pipeline
    if _multimodal_pipeline is None:
        settings = get_settings()
        _multimodal_pipeline = MultimodalRAGPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            cache_layer=cache_layer,
            anthropic_api_key=settings.anthropic_api_key,
        )
        _multimodal_pipeline.connect()
    return _multimodal_pipeline


async def get_crag_pipeline(
    vector_store: PgVectorClient = Depends(get_pgvector_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> CRAGPipeline:
    global _crag_pipeline
    if _crag_pipeline is None:
        settings = get_settings()
        _crag_pipeline = CRAGPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            anthropic_api_key=settings.anthropic_api_key,
            tavily_api_key=settings.tavily_api_key,
        )
        _crag_pipeline.connect()
    return _crag_pipeline


async def get_self_rag_pipeline(
    vector_store: PgVectorClient = Depends(get_pgvector_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> SelfRAGPipeline:
    global _self_rag_pipeline
    if _self_rag_pipeline is None:
        settings = get_settings()
        _self_rag_pipeline = SelfRAGPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            anthropic_api_key=settings.anthropic_api_key,
            tavily_api_key=settings.tavily_api_key,
        )
        _self_rag_pipeline.connect()
    return _self_rag_pipeline


async def get_neo4j_client() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        settings = get_settings()
        _neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        await _neo4j_client.connect()
    return _neo4j_client


async def get_video_rag_pipeline(
    vector_store: PgVectorClient = Depends(get_pgvector_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
) -> VideoRAGPipeline:
    global _video_rag_pipeline
    if _video_rag_pipeline is None:
        settings = get_settings()
        _video_rag_pipeline = VideoRAGPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            neo4j_client=neo4j_client,
            anthropic_api_key=settings.anthropic_api_key,
        )
        _video_rag_pipeline.connect()
    return _video_rag_pipeline


# Type aliases for cleaner route signatures
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
PgVectorDep = Annotated[PgVectorClient, Depends(get_pgvector_client)]
CacheLayerDep = Annotated[CacheLayer, Depends(get_cache_layer)]
NaivePipelineDep = Annotated[NaiveRAGPipeline, Depends(get_naive_pipeline)]
MultimodalPipelineDep = Annotated[MultimodalRAGPipeline, Depends(get_multimodal_pipeline)]
CRAGPipelineDep = Annotated[CRAGPipeline, Depends(get_crag_pipeline)]
SelfRAGPipelineDep = Annotated[SelfRAGPipeline, Depends(get_self_rag_pipeline)]
VideoRAGPipelineDep = Annotated[VideoRAGPipeline, Depends(get_video_rag_pipeline)]
