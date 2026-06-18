"""A/B Comparison — Run a query through two pipelines side-by-side."""

from __future__ import annotations

import asyncio
import html
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:8000"


async def run_pipeline_query(
    api_base: str,
    query: str,
    pipeline: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Call the /query endpoint for a single pipeline and return the response dict."""
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{api_base}/query",
            json={
                "query": query,
                "pipeline": pipeline,
                "top_k": top_k,
                "use_cache": False,
            },
        )
        resp.raise_for_status()
    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()
    data["_client_latency_ms"] = round(elapsed_ms, 1)
    return data


async def compare_pipelines(
    query: str,
    pipeline_a: str,
    pipeline_b: str,
    api_base: str = DEFAULT_API_BASE,
    top_k: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run query through two pipelines in parallel, return both responses."""
    result_a, result_b = await asyncio.gather(
        run_pipeline_query(api_base, query, pipeline_a, top_k),
        run_pipeline_query(api_base, query, pipeline_b, top_k),
    )
    return result_a, result_b


def _render_sources(sources: list[dict[str, Any]], max_shown: int = 3) -> str:
    """Render source chunks as compact HTML list."""
    if not sources:
        return '<span style="color: #94a3b8;">No sources</span>'
    items = ""
    for s in sources[:max_shown]:
        score = s.get("score", 0)
        content_preview = html.escape(s.get("content", "")[:120])
        items += (
            f'<li style="margin-bottom: 4px; font-size: 12px; color: #475569;">'
            f"<strong>{score:.3f}</strong> — {content_preview}…</li>"
        )
    remaining = len(sources) - max_shown
    if remaining > 0:
        items += f'<li style="color: #94a3b8; font-size: 12px;">+{remaining} more</li>'
    return f'<ul style="padding-left: 16px; margin: 4px 0;">{items}</ul>'


def _render_pipeline_card(label: str, data: dict[str, Any], colour: str) -> str:
    """Render one side of the A/B comparison."""
    answer = html.escape(data.get("answer", "No answer"))
    pipeline = data.get("pipeline", "unknown")
    latency = data.get("latency_ms") or data.get("_client_latency_ms", 0)
    cache_hit = data.get("cache_hit", False)
    sources = data.get("sources", [])
    metadata = data.get("metadata", {})
    cost = metadata.get("total_cost_usd", metadata.get("embedding_cost_usd", 0))

    return f"""
    <div style="flex: 1; border: 2px solid {colour}; border-radius: 10px; padding: 16px;
                background: #ffffff; min-width: 300px;">
      <div style="display: flex; justify-content: space-between; align-items: center;
                  margin-bottom: 12px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">
        <h4 style="margin: 0; color: {colour}; font-size: 15px;">{label}: {pipeline}</h4>
        <span style="font-size: 12px; color: #64748b;">
          {latency:.0f}ms {"(cached)" if cache_hit else ""}
        </span>
      </div>

      <div style="font-size: 14px; color: #1e293b; line-height: 1.6; margin-bottom: 12px;">
        {answer}
      </div>

      <details style="margin-top: 8px;">
        <summary style="cursor: pointer; font-size: 13px; color: #475569; font-weight: 600;">
          Sources ({len(sources)})
        </summary>
        {_render_sources(sources)}
      </details>

      <div style="margin-top: 8px; font-size: 12px; color: #64748b;
                  display: flex; gap: 16px;">
        <span>Cost: ${cost:.5f}</span>
        <span>Latency: {latency:.0f}ms</span>
      </div>
    </div>"""


def render_comparison(
    query: str,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> str:
    """Render full A/B comparison as HTML."""
    card_a = _render_pipeline_card("Pipeline A", result_a, "#3b82f6")
    card_b = _render_pipeline_card("Pipeline B", result_b, "#8b5cf6")

    # Metrics comparison bar
    latency_a = result_a.get("latency_ms") or result_a.get("_client_latency_ms", 0)
    latency_b = result_b.get("latency_ms") or result_b.get("_client_latency_ms", 0)
    meta_a = result_a.get("metadata", {})
    meta_b = result_b.get("metadata", {})
    cost_a = meta_a.get("total_cost_usd", meta_a.get("embedding_cost_usd", 0))
    cost_b = meta_b.get("total_cost_usd", meta_b.get("embedding_cost_usd", 0))
    sources_a = len(result_a.get("sources", []))
    sources_b = len(result_b.get("sources", []))

    def _winner_style(val_a: float, val_b: float, lower_better: bool = True) -> tuple[str, str]:
        """Return (style_a, style_b) with the winner highlighted."""
        if val_a == val_b:
            return ("", "")
        win_a = val_a < val_b if lower_better else val_a > val_b
        green = "color: #16a34a; font-weight: 700;"
        return (green, "") if win_a else ("", green)

    ls_a, ls_b = _winner_style(latency_a, latency_b, lower_better=True)
    cs_a, cs_b = _winner_style(cost_a, cost_b, lower_better=True)
    ss_a, ss_b = _winner_style(sources_a, sources_b, lower_better=False)

    metrics_bar = f"""
    <div style="background: #f8fafc; border-radius: 8px; padding: 12px 16px; margin-top: 12px;
                display: flex; justify-content: space-around; font-size: 13px; color: #334155;">
      <div style="text-align: center;">
        <div style="font-weight: 600; margin-bottom: 4px;">Latency</div>
        <span style="{ls_a}">{latency_a:.0f}ms</span> vs <span style="{ls_b}">{latency_b:.0f}ms</span>
      </div>
      <div style="text-align: center;">
        <div style="font-weight: 600; margin-bottom: 4px;">Cost</div>
        <span style="{cs_a}">${cost_a:.5f}</span> vs <span style="{cs_b}">${cost_b:.5f}</span>
      </div>
      <div style="text-align: center;">
        <div style="font-weight: 600; margin-bottom: 4px;">Sources</div>
        <span style="{ss_a}">{sources_a}</span> vs <span style="{ss_b}">{sources_b}</span>
      </div>
    </div>"""

    return f"""
<div style="font-family: system-ui, sans-serif; max-width: 1000px; margin: 0 auto;">
  <h3 style="color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">
    A/B Comparison
  </h3>
  <p style="color: #64748b; font-size: 13px; margin-bottom: 12px;">
    Query: <em>"{html.escape(query)}"</em>
  </p>

  <div style="display: flex; gap: 12px; flex-wrap: wrap;">
    {card_a}
    {card_b}
  </div>

  {metrics_bar}
</div>
"""


async def send_comparison(
    query: str,
    pipeline_a: str,
    pipeline_b: str,
    api_base: str = DEFAULT_API_BASE,
    top_k: int = 5,
) -> None:
    """Run A/B comparison and send results as a Chainlit message."""
    try:
        import chainlit as cl

        await cl.Message(
            content=f"Running A/B comparison: **{pipeline_a}** vs **{pipeline_b}**…",
        ).send()

        result_a, result_b = await compare_pipelines(
            query,
            pipeline_a,
            pipeline_b,
            api_base,
            top_k,
        )
        content = render_comparison(query, result_a, result_b)
        await cl.Message(content=content, author="A/B Comparison").send()

    except ImportError:
        logger.warning("chainlit not installed — cannot send comparison")
    except httpx.HTTPError as exc:
        logger.error("API error during comparison: %s", exc)
        try:
            import chainlit as cl

            await cl.Message(content=f"Comparison failed: {exc}").send()
        except ImportError:
            pass
