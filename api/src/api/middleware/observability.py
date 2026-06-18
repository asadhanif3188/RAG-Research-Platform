"""Observability middleware — cost tracking and metrics collection.

Tracks tokens used per LLM call, calculates USD cost, and logs to LangFuse.
Metrics are stored in-memory and exposed via the /metrics API.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Model pricing (USD per 1K tokens) ────────────────────────────────────────
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate USD cost for a single LLM/embedding call."""
    pricing = MODEL_PRICING.get(model, {"input": 0.003, "output": 0.015})
    cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
    return round(cost, 8)


@dataclass
class QueryMetric:
    """Metrics recorded for a single query execution."""

    pipeline: str
    query: str
    latency_ms: float
    total_cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)
    ragas_scores: dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """In-memory metrics store aggregated per pipeline.

    Thread-safe for single-process async usage (FastAPI with uvicorn).
    """

    def __init__(self) -> None:
        self._history: list[QueryMetric] = []
        self._pipeline_stats: dict[str, list[QueryMetric]] = defaultdict(list)

    def record(self, metric: QueryMetric) -> None:
        """Record a query metric."""
        self._history.append(metric)
        self._pipeline_stats[metric.pipeline].append(metric)
        logger.info(
            "Metric recorded: pipeline=%s latency=%.0fms cost=$%.6f",
            metric.pipeline,
            metric.latency_ms,
            metric.total_cost_usd,
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate metrics per pipeline."""
        result: dict[str, Any] = {"pipelines": {}, "total_queries": len(self._history)}

        for pipeline, metrics in self._pipeline_stats.items():
            latencies = sorted(m.latency_ms for m in metrics)
            costs = [m.total_cost_usd for m in metrics]
            cache_hits = sum(1 for m in metrics if m.cache_hit)

            n = len(latencies)
            p50 = latencies[n // 2] if n else None
            p95 = latencies[int(n * 0.95)] if n else None
            p99 = latencies[int(n * 0.99)] if n else None

            # Aggregate RAGAS scores
            ragas_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            ragas_agg: dict[str, float | None] = {}
            for key in ragas_keys:
                values = [m.ragas_scores[key] for m in metrics if key in m.ragas_scores]
                ragas_agg[key] = sum(values) / len(values) if values else None

            total_seconds = (
                (metrics[-1].timestamp - metrics[0].timestamp) if len(metrics) > 1 else 1.0
            )
            qps = len(metrics) / max(total_seconds, 0.001)

            result["pipelines"][pipeline] = {
                "ragas": ragas_agg,
                "performance": {
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "qps": round(qps, 2),
                },
                "cost": {
                    "avg_usd_per_query": sum(costs) / len(costs) if costs else None,
                    "total_usd": sum(costs),
                },
                "cache": {
                    "hit_rate": cache_hits / len(metrics) if metrics else None,
                },
                "query_count": len(metrics),
            }

        return result

    def history(
        self,
        pipeline: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent metrics as dicts, optionally filtered by pipeline."""
        source = self._pipeline_stats.get(pipeline, []) if pipeline else self._history
        items = source[-limit:]
        return [
            {
                "pipeline": m.pipeline,
                "latency_ms": m.latency_ms,
                "total_cost_usd": m.total_cost_usd,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "cache_hit": m.cache_hit,
                "timestamp": m.timestamp,
                "ragas_scores": m.ragas_scores,
            }
            for m in items
        ]

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._history.clear()
        self._pipeline_stats.clear()


# Module-level singleton
_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Return the singleton MetricsCollector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


async def log_to_langfuse(
    metric: QueryMetric,
    langfuse_host: str = "",
    public_key: str = "",
    secret_key: str = "",
) -> None:
    """Log a query metric to LangFuse (best-effort, non-blocking)."""
    if not (langfuse_host and public_key and secret_key):
        return

    try:
        import httpx

        payload = {
            "name": f"rag-query-{metric.pipeline}",
            "metadata": {
                "pipeline": metric.pipeline,
                "latency_ms": metric.latency_ms,
                "total_cost_usd": metric.total_cost_usd,
                "input_tokens": metric.input_tokens,
                "output_tokens": metric.output_tokens,
                "cache_hit": metric.cache_hit,
            },
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{langfuse_host}/api/public/events",
                json=payload,
                auth=(public_key, secret_key),
            )
    except Exception as exc:
        logger.debug("LangFuse logging failed (non-critical): %s", exc)
