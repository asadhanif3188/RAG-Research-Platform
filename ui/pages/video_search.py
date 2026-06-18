"""Video Search page — multi-video Q&A mode for Chainlit."""

from __future__ import annotations

import logging
from typing import Any

import chainlit as cl

from ui.components.video_player import render_segment_list, render_video_player

logger = logging.getLogger(__name__)


async def handle_video_query(
    query: str,
    pipeline: Any,
    top_k: int = 5,
    video_id: str | None = None,
) -> None:
    """Process a video search query and display results in Chainlit.

    Args:
        query: User's natural language query.
        pipeline: Initialised VideoRAGPipeline instance.
        top_k: Number of segments to retrieve.
        video_id: Optional filter for a specific video.
    """
    from shared.models.query import QueryRequest

    # Show thinking indicator
    msg = cl.Message(content="Searching video library...")
    await msg.send()

    request = QueryRequest(
        query=query,
        pipeline="video_rag",
        top_k=top_k,
        filters={"video_id": video_id} if video_id else {},
    )

    response = await pipeline.run(request)

    # Build result display
    segments_data = [
        {
            "video_id": src.document_id,
            "start_ts": src.metadata.get("start_ts", 0.0),
            "end_ts": src.metadata.get("end_ts", 0.0),
            "transcript": src.content,
            "text_score": src.metadata.get("text_score", 0.0),
            "visual_score": src.metadata.get("visual_score", 0.0),
            "fused_score": src.score,
            "url": src.metadata.get("url", ""),
            "title": src.metadata.get("title", src.document_id),
        }
        for src in response.sources
    ]

    # Group results by video
    video_ids = response.metadata.get("video_ids", [])
    video_count = len(video_ids)

    answer_md = f"**Answer:**\n{response.answer}\n\n"
    answer_md += f"*Found {len(segments_data)} segments across {video_count} video(s) "
    answer_md += f"in {response.latency_ms:.0f}ms*\n\n"

    # Render video segments
    segments_html = render_segment_list(segments_data)

    # Update the message with results
    msg.content = answer_md + segments_html
    await msg.update()


async def handle_video_list(pipeline: Any) -> None:
    """Display all indexed videos."""
    # This would typically query the video registry
    msg = cl.Message(content="**Indexed Videos:**\n\nUse the MCP server to list videos.")
    await msg.send()
