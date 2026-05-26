"""Query router — POST /query routes to the selected RAG pipeline strategy."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status

from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse

from api.dependencies import CacheLayerDep, NaivePipelineDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Run a RAG query",
    description=(
        "Routes the query to the selected pipeline strategy. "
        "Returns a structured answer with source chunks and latency."
    ),
)
async def run_query(
    request: QueryRequest,
    naive_pipeline: NaivePipelineDep,
) -> QueryResponse:
    """Execute a RAG query through the selected pipeline."""
    logger.info(
        "Query received: pipeline=%s, query=%s...",
        request.pipeline,
        request.query[:60],
    )

    try:
        match request.pipeline:
            case PipelineStrategy.FASTEST_RAG:
                return await naive_pipeline.run(request)
            case _:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail=f"Pipeline '{request.pipeline}' is not yet implemented. "
                    "Available: fastest_rag",
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline error for query: %s", request.query[:60])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc


@router.get("/health", summary="Pipeline health check")
async def health_check(
    naive_pipeline: NaivePipelineDep,
    cache_layer: CacheLayerDep,
) -> dict[str, object]:
    cache_ok = await cache_layer.health_check()
    return {
        "status": "ok",
        "cache_healthy": cache_ok,
        "cache_stats": cache_layer.stats(),
    }
