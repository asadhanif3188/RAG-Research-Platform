"""Benchmark router — GET /benchmark runs perf tests and returns results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

RESULTS_DIR = Path("benchmark_results")


class BenchmarkRequest(BaseModel):
    n_queries: int = Query(default=100, ge=10, le=10_000)
    top_k: int = Query(default=10, ge=1, le=50)
    vector_size: int = Query(default=3072)


@router.post(
    "/run",
    summary="Run benchmark in background",
    description="Starts an async benchmark across all Qdrant collection variants.",
)
async def run_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Queue a benchmark run in the background."""
    background_tasks.add_task(
        _run_benchmark_task,
        n_queries=request.n_queries,
        top_k=request.top_k,
        vector_size=request.vector_size,
    )
    return {"status": "started", "message": f"Benchmarking {request.n_queries} queries in background"}


@router.get(
    "/results",
    summary="List available benchmark results",
)
async def list_results() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return [p.name for p in sorted(RESULTS_DIR.glob("*.json"), reverse=True)]


@router.get(
    "/results/{filename}",
    summary="Get a specific benchmark result",
)
async def get_result(filename: str) -> dict[str, Any]:
    path = RESULTS_DIR / filename
    if not path.exists() or not filename.endswith(".json"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")
    import json
    return json.loads(path.read_text())


async def _run_benchmark_task(
    n_queries: int, top_k: int, vector_size: int
) -> None:
    """Background task that runs the benchmark and saves results."""
    import datetime
    from shared.config import get_settings
    from fastest_rag.benchmark import BenchmarkRunner

    logger.info("Background benchmark started: %d queries, top_k=%d", n_queries, top_k)

    try:
        settings = get_settings()
        runner = BenchmarkRunner(
            qdrant_host=settings.qdrant_host,
            qdrant_port=settings.qdrant_port,
            qdrant_api_key=settings.qdrant_api_key,
        )
        await runner.connect()
        report = await runner.run(
            n_queries=n_queries, top_k=top_k, vector_size=vector_size
        )
        await runner.close()

        RESULTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
        BenchmarkRunner.save(report, output_path)
        logger.info("Benchmark complete, saved to %s", output_path)
    except Exception:
        logger.exception("Background benchmark failed")
