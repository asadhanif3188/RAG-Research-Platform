"""Metrics router — GET /metrics/summary and /metrics/history."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from api.middleware.observability import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "/summary",
    summary="Aggregated pipeline metrics",
    description="Returns per-pipeline RAGAS scores, latency percentiles, cost, and cache stats.",
)
async def metrics_summary() -> dict[str, Any]:
    """Return aggregated metrics for all pipelines."""
    collector = get_metrics_collector()
    return collector.summary()


@router.get(
    "/history",
    summary="Recent metrics history",
    description="Returns time-series metrics, optionally filtered by pipeline.",
)
async def metrics_history(
    pipeline: str | None = Query(default=None, description="Filter by pipeline name"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max records to return"),
) -> list[dict[str, Any]]:
    """Return recent query metrics as a list."""
    collector = get_metrics_collector()
    return collector.history(pipeline=pipeline, limit=limit)


@router.delete(
    "",
    summary="Clear all metrics",
    description="Reset the in-memory metrics store.",
)
async def clear_metrics() -> dict[str, str]:
    """Clear all stored metrics."""
    collector = get_metrics_collector()
    collector.clear()
    return {"status": "cleared"}
