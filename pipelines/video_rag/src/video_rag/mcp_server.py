"""MCP Server — expose video search tools via FastMCP for Claude integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

if TYPE_CHECKING:
    from video_rag.knowledge_graph import KnowledgeGraph
    from video_rag.segment_retriever import SegmentRetriever

logger = logging.getLogger(__name__)

mcp = FastMCP("Video RAG")


# Module-level references set by create_mcp_server()
_retriever: SegmentRetriever | None = None
_knowledge_graph: KnowledgeGraph | None = None
_video_registry: dict[str, dict[str, Any]] = {}


def create_mcp_server(
    retriever: SegmentRetriever,
    knowledge_graph: KnowledgeGraph,
    video_registry: dict[str, dict[str, Any]] | None = None,
) -> FastMCP:
    """Configure and return the FastMCP server with video search tools.

    Args:
        retriever: Initialised SegmentRetriever for hybrid search.
        knowledge_graph: Initialised KnowledgeGraph for topic queries.
        video_registry: Optional in-memory registry of indexed videos.

    Returns:
        Configured FastMCP server instance.
    """
    global _retriever, _knowledge_graph, _video_registry  # noqa: PLW0603
    _retriever = retriever
    _knowledge_graph = knowledge_graph
    _video_registry = video_registry or {}
    return mcp


@mcp.tool()
async def search_video(
    query: str,
    top_k: int = 5,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search across indexed videos using hybrid text+visual retrieval.

    Returns timestamped video segments ranked by relevance to the query.
    Each result includes the transcript text, timestamps, and similarity scores.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default 5).
        video_id: Optional — restrict search to a specific video by ID.
    """
    if _retriever is None:
        return [{"error": "Retriever not initialised"}]

    results = await _retriever.retrieve(query=query, top_k=top_k, video_id=video_id)
    return [
        {
            "segment_id": r.segment_id,
            "video_id": r.video_id,
            "start_ts": r.start_ts,
            "end_ts": r.end_ts,
            "transcript": r.transcript,
            "text_score": round(r.text_score, 4),
            "visual_score": round(r.visual_score, 4),
            "fused_score": round(r.fused_score, 4),
        }
        for r in results
    ]


@mcp.tool()
async def get_segment(segment_id: str, video_id: str) -> dict[str, Any]:
    """Get detailed information about a specific video segment.

    Args:
        segment_id: The unique segment identifier.
        video_id: The video that contains this segment.
    """
    if _knowledge_graph is None:
        return {"error": "Knowledge graph not initialised"}

    segments = await _knowledge_graph._neo4j.get_video_segments(video_id)
    for seg in segments:
        if seg.get("segment_id") == segment_id:
            return {
                "segment_id": segment_id,
                "video_id": video_id,
                "start_ts": seg.get("start_s", 0.0),
                "end_ts": seg.get("end_s", 0.0),
                "transcript": seg.get("transcript", ""),
            }

    return {"error": f"Segment {segment_id} not found in video {video_id}"}


@mcp.tool()
async def list_videos() -> list[dict[str, Any]]:
    """List all indexed videos with their metadata.

    Returns a list of video summaries including title, URL, duration,
    and number of indexed segments.
    """
    return [
        {
            "video_id": vid,
            "title": info.get("title", ""),
            "url": info.get("url", ""),
            "duration_s": info.get("duration_s", 0.0),
            "segment_count": info.get("segment_count", 0),
        }
        for vid, info in _video_registry.items()
    ]


@mcp.tool()
async def get_transcript(video_id: str) -> dict[str, Any]:
    """Get the full transcript of a video with timestamps.

    Args:
        video_id: The unique video identifier.
    """
    if _knowledge_graph is None:
        return {"error": "Knowledge graph not initialised"}

    segments = await _knowledge_graph._neo4j.get_video_segments(video_id)
    if not segments:
        return {"error": f"No segments found for video {video_id}"}

    video_info = _video_registry.get(video_id, {})
    return {
        "video_id": video_id,
        "title": video_info.get("title", "Unknown"),
        "segments": [
            {
                "start_ts": seg.get("start_s", 0.0),
                "end_ts": seg.get("end_s", 0.0),
                "transcript": seg.get("transcript", ""),
            }
            for seg in segments
        ],
    }
