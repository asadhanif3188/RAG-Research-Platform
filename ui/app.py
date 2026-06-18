"""RAG Research Platform — Chainlit entry point.

Run with: chainlit run ui/app.py -w
Or:       uv run chainlit run ui/app.py -w

This wires together the pipeline selector, A/B comparison, metrics dashboard,
provenance viewer, and video player into a single chat interface.
"""

from __future__ import annotations

import logging
import os

import chainlit as cl

from ui.components.pipeline_selector import PIPELINE_REGISTRY, PipelineSelector
from ui.pages.compare import compare_pipelines, render_comparison
from ui.pages.metrics_dashboard import fetch_metrics_summary, render_dashboard

logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


@cl.on_chat_start
async def on_chat_start() -> None:
    """Display the pipeline selector when a new chat session begins."""
    selector = PipelineSelector(selected="fastest_rag")
    await selector.send()
    await PipelineSelector.store_selection("fastest_rag")

    await cl.Message(
        content=(
            "Welcome to the **RAG Research Platform**. "
            "Select a pipeline above, then ask a question.\n\n"
            "**Commands:**\n"
            "- `/compare <pipeline_a> <pipeline_b>` — A/B comparison\n"
            "- `/metrics` — show metrics dashboard\n"
            "- `/select <pipeline>` — switch pipeline\n"
            "- `/pipelines` — list available pipelines"
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages — route commands or run queries."""
    text = message.content.strip()

    # ── Slash commands ──────────────────────────────────────────────────────
    if text.startswith("/"):
        await _handle_command(text)
        return

    # ── Normal query ────────────────────────────────────────────────────────
    pipeline = await PipelineSelector.get_selection()
    await _run_query(text, pipeline)


async def _handle_command(text: str) -> None:
    """Parse and dispatch slash commands."""
    parts = text.split()
    cmd = parts[0].lower()

    if cmd == "/compare":
        if len(parts) < 3:
            await cl.Message(
                content="Usage: `/compare <pipeline_a> <pipeline_b>` — e.g. `/compare fastest_rag self_rag`",
            ).send()
            return
        pipeline_a, pipeline_b = parts[1], parts[2]
        query = " ".join(parts[3:]) if len(parts) > 3 else None
        if not query:
            await cl.Message(content="Enter the query to compare:").send()
            return
        await _run_comparison(query, pipeline_a, pipeline_b)

    elif cmd == "/metrics":
        await _show_metrics()

    elif cmd == "/select":
        if len(parts) < 2:
            await cl.Message(content="Usage: `/select <pipeline>` — e.g. `/select self_rag`").send()
            return
        strategy = parts[1]
        valid = {p.strategy for p in PIPELINE_REGISTRY}
        if strategy not in valid:
            await cl.Message(
                content=f"Unknown pipeline `{strategy}`. Available: {', '.join(sorted(valid))}",
            ).send()
            return
        await PipelineSelector.store_selection(strategy)
        selector = PipelineSelector(selected=strategy)
        await selector.send()
        await cl.Message(content=f"Switched to **{strategy}**.").send()

    elif cmd == "/pipelines":
        lines = [f"- **{p.name}** (`{p.strategy}`): {p.description}" for p in PIPELINE_REGISTRY]
        await cl.Message(content="Available pipelines:\n" + "\n".join(lines)).send()

    else:
        await cl.Message(content=f"Unknown command `{cmd}`. Try `/compare`, `/metrics`, `/select`, or `/pipelines`.").send()


async def _run_query(query: str, pipeline: str) -> None:
    """Send a query to the API and display the response."""
    import httpx

    await cl.Message(content=f"Querying **{pipeline}**...").send()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{API_BASE}/query",
                json={"query": query, "pipeline": pipeline, "top_k": 5, "use_cache": True},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        await cl.Message(content=f"API error: {exc}").send()
        return

    answer = data.get("answer", "No answer returned.")
    latency = data.get("latency_ms", 0)
    sources = data.get("sources", [])
    metadata = data.get("metadata", {})
    cost = metadata.get("total_cost_usd", metadata.get("embedding_cost_usd", 0))
    cache_hit = data.get("cache_hit", False)

    # Build response message
    parts = [answer, ""]
    parts.append(f"*Pipeline: {pipeline} | Latency: {latency:.0f}ms | Cost: ${cost:.5f}*")
    if cache_hit:
        parts.append("*(served from cache)*")

    if sources:
        parts.append(f"\n**Sources** ({len(sources)} chunks):")
        for i, s in enumerate(sources[:5], 1):
            score = s.get("score", 0)
            content_preview = s.get("content", "")[:100]
            parts.append(f"{i}. (score={score:.3f}) {content_preview}...")

    await cl.Message(content="\n".join(parts)).send()

    # Show provenance if multimodal_rag
    provenance = metadata.get("provenance")
    if provenance and pipeline == "multimodal_rag":
        from ui.components.provenance_viewer import ProvenanceViewer

        viewer = ProvenanceViewer(answer, provenance)
        await viewer.send()

    # Show video player if video_rag
    if pipeline == "video_rag" and sources:
        from ui.components.video_player import render_video_player

        for s in sources[:1]:
            video_url = s.get("video_url", "")
            start_ts = s.get("start_timestamp", 0)
            end_ts = s.get("end_timestamp", 0)
            transcript = s.get("content", "")
            if video_url:
                html_content = render_video_player(video_url, start_ts, end_ts, transcript)
                await cl.Message(content=html_content).send()


async def _run_comparison(query: str, pipeline_a: str, pipeline_b: str) -> None:
    """Run A/B comparison and display results."""
    await cl.Message(content=f"Running A/B comparison: **{pipeline_a}** vs **{pipeline_b}**...").send()

    try:
        result_a, result_b = await compare_pipelines(query, pipeline_a, pipeline_b, API_BASE)
        html_content = render_comparison(query, result_a, result_b)
        await cl.Message(content=html_content, author="A/B Comparison").send()
    except Exception as exc:
        await cl.Message(content=f"Comparison failed: {exc}").send()


async def _show_metrics() -> None:
    """Fetch and display the metrics dashboard."""
    try:
        metrics = await fetch_metrics_summary(API_BASE)
        html_content = render_dashboard(metrics)
        await cl.Message(content=html_content, author="Metrics Dashboard").send()
    except Exception as exc:
        await cl.Message(content=f"Failed to load metrics: {exc}").send()
