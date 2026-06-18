"""Metrics Dashboard — Real-time pipeline performance and quality metrics."""

from __future__ import annotations

import csv
import html
import io
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:8000"

# Pipeline display names
_PIPELINE_NAMES = {
    "fastest_rag": "Naive RAG (Fastest)",
    "multimodal_rag": "Multimodal RAG",
    "corrective_rag": "CRAG",
    "self_rag": "Self-RAG",
    "video_rag": "Video RAG",
}


async def fetch_metrics_summary(
    api_base: str = DEFAULT_API_BASE,
) -> dict[str, Any]:
    """Fetch aggregated metrics from the API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{api_base}/metrics/summary")
        resp.raise_for_status()
    return resp.json()


async def fetch_metrics_history(
    api_base: str = DEFAULT_API_BASE,
    pipeline: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch time-series metrics history."""
    params: dict[str, Any] = {"limit": limit}
    if pipeline:
        params["pipeline"] = pipeline
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{api_base}/metrics/history", params=params)
        resp.raise_for_status()
    return resp.json()


def _render_metric_cell(value: float | None, fmt: str = ".3f", suffix: str = "") -> str:
    """Render a single metric value with colour coding."""
    if value is None:
        return '<span style="color: #cbd5e1;">—</span>'
    colour = "#16a34a" if value >= 0.7 else "#d97706" if value >= 0.4 else "#dc2626"
    return f'<span style="color: {colour}; font-weight: 600;">{value:{fmt}}{suffix}</span>'


def _render_latency_cell(p50: float | None, p95: float | None, p99: float | None) -> str:
    """Render latency percentiles."""
    parts = []
    for label, val in [("p50", p50), ("p95", p95), ("p99", p99)]:
        if val is not None:
            colour = "#16a34a" if val < 500 else "#d97706" if val < 2000 else "#dc2626"
            parts.append(f'<span style="color: {colour};">{label}: {val:.0f}ms</span>')
    return " / ".join(parts) if parts else '<span style="color: #cbd5e1;">—</span>'


def render_dashboard(metrics: dict[str, Any]) -> str:
    """Render the full metrics dashboard as HTML.

    Expected metrics format:
    {
        "pipelines": {
            "fastest_rag": {
                "ragas": {"faithfulness": 0.85, "answer_relevancy": 0.90, ...},
                "performance": {"p50_ms": 120, "p95_ms": 450, "p99_ms": 800, "qps": 5.2},
                "cost": {"avg_usd_per_query": 0.003},
                "cache": {"hit_rate": 0.45},  # only for fastest_rag
                "query_count": 150,
            },
            ...
        },
        "total_queries": 500,
    }
    """
    pipelines = metrics.get("pipelines", {})
    total = metrics.get("total_queries", 0)

    rows = ""
    for strategy, data in pipelines.items():
        name = html.escape(_PIPELINE_NAMES.get(strategy, strategy))
        ragas = data.get("ragas", {})
        perf = data.get("performance", {})
        cost_data = data.get("cost", {})
        cache = data.get("cache", {})
        count = data.get("query_count", 0)

        faithfulness = _render_metric_cell(ragas.get("faithfulness"))
        relevancy = _render_metric_cell(ragas.get("answer_relevancy"))
        precision = _render_metric_cell(ragas.get("context_precision"))
        recall = _render_metric_cell(ragas.get("context_recall"))
        latency = _render_latency_cell(
            perf.get("p50_ms"),
            perf.get("p95_ms"),
            perf.get("p99_ms"),
        )
        qps = perf.get("qps")
        qps_str = f"{qps:.1f}" if qps is not None else "—"
        avg_cost = cost_data.get("avg_usd_per_query")
        cost_str = f"${avg_cost:.5f}" if avg_cost is not None else "—"
        hit_rate = cache.get("hit_rate")
        cache_str = f"{hit_rate:.0%}" if hit_rate is not None else "—"

        rows += f"""
      <tr style="border-bottom: 1px solid #e2e8f0;">
        <td style="padding: 10px; font-weight: 600; color: #1e293b;">{name}</td>
        <td style="padding: 10px; text-align: center;">{faithfulness}</td>
        <td style="padding: 10px; text-align: center;">{relevancy}</td>
        <td style="padding: 10px; text-align: center;">{precision}</td>
        <td style="padding: 10px; text-align: center;">{recall}</td>
        <td style="padding: 10px; font-size: 12px;">{latency}</td>
        <td style="padding: 10px; text-align: center;">{qps_str}</td>
        <td style="padding: 10px; text-align: center; color: #475569;">{cost_str}</td>
        <td style="padding: 10px; text-align: center; color: #475569;">{cache_str}</td>
        <td style="padding: 10px; text-align: center; color: #64748b;">{count}</td>
      </tr>"""

    return f"""
<div style="font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto;">
  <h3 style="color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">
    Pipeline Metrics Dashboard
  </h3>
  <p style="color: #64748b; font-size: 13px; margin-bottom: 12px;">
    Total queries: <strong>{total}</strong>
  </p>

  <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; font-size: 13px;
                  border: 1px solid #e2e8f0; border-radius: 8px;">
      <thead>
        <tr style="background: #f1f5f9; text-align: left;">
          <th style="padding: 10px; color: #475569;">Pipeline</th>
          <th style="padding: 10px; color: #475569; text-align: center;">Faithfulness</th>
          <th style="padding: 10px; color: #475569; text-align: center;">Relevancy</th>
          <th style="padding: 10px; color: #475569; text-align: center;">Precision</th>
          <th style="padding: 10px; color: #475569; text-align: center;">Recall</th>
          <th style="padding: 10px; color: #475569;">Latency</th>
          <th style="padding: 10px; color: #475569; text-align: center;">QPS</th>
          <th style="padding: 10px; color: #475569; text-align: center;">Avg Cost</th>
          <th style="padding: 10px; color: #475569; text-align: center;">Cache Hit</th>
          <th style="padding: 10px; color: #475569; text-align: center;">Queries</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</div>
"""


def export_metrics_csv(metrics: dict[str, Any]) -> str:
    """Export pipeline metrics as CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "pipeline",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "qps",
            "avg_cost_usd",
            "cache_hit_rate",
            "query_count",
        ]
    )

    for strategy, data in metrics.get("pipelines", {}).items():
        ragas = data.get("ragas", {})
        perf = data.get("performance", {})
        cost_data = data.get("cost", {})
        cache = data.get("cache", {})
        writer.writerow(
            [
                strategy,
                ragas.get("faithfulness", ""),
                ragas.get("answer_relevancy", ""),
                ragas.get("context_precision", ""),
                ragas.get("context_recall", ""),
                perf.get("p50_ms", ""),
                perf.get("p95_ms", ""),
                perf.get("p99_ms", ""),
                perf.get("qps", ""),
                cost_data.get("avg_usd_per_query", ""),
                cache.get("hit_rate", ""),
                data.get("query_count", 0),
            ]
        )

    return output.getvalue()


async def send_dashboard(api_base: str = DEFAULT_API_BASE) -> None:
    """Fetch metrics and send dashboard as a Chainlit message."""
    try:
        import chainlit as cl

        metrics = await fetch_metrics_summary(api_base)
        content = render_dashboard(metrics)
        await cl.Message(content=content, author="Metrics Dashboard").send()

    except ImportError:
        logger.warning("chainlit not installed — cannot send dashboard")
    except httpx.HTTPError as exc:
        logger.error("Failed to fetch metrics: %s", exc)
        try:
            import chainlit as cl

            await cl.Message(content=f"Failed to load metrics: {exc}").send()
        except ImportError:
            pass
