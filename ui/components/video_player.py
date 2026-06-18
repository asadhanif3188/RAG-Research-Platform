"""VideoPlayer — Chainlit component for displaying retrieved video segments."""

from __future__ import annotations

from typing import Any


def render_video_player(
    video_url: str,
    start_ts: float,
    end_ts: float,
    transcript: str = "",
    title: str = "",
    scores: dict[str, float] | None = None,
) -> str:
    """Render an HTML5 video player with auto-seek and transcript overlay.

    Returns an HTML string that can be embedded in Chainlit messages via
    ``cl.Message(content=html)``.

    Args:
        video_url: URL or local path to the video file.
        start_ts: Start timestamp in seconds.
        end_ts: End timestamp in seconds.
        transcript: Transcript text to display alongside the video.
        title: Video title.
        scores: Optional similarity scores to display.
    """
    score_html = ""
    if scores:
        score_parts = [
            f'<span style="margin-right:12px;"><b>{k}:</b> {v:.3f}</span>'
            for k, v in scores.items()
        ]
        score_html = f'<div style="margin-top:4px;font-size:0.85em;color:#666;">{"".join(score_parts)}</div>'

    start_fmt = _format_timestamp(start_ts)
    end_fmt = _format_timestamp(end_ts)

    html = f"""
<div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin:8px 0; max-width:640px;">
  <div style="font-weight:600; margin-bottom:8px;">{title or "Video Segment"}</div>
  <video width="100%" controls preload="metadata"
         style="border-radius:4px; background:#000;">
    <source src="{video_url}#t={start_ts},{end_ts}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.9em;">
    <span>&#x23F1; {start_fmt} &ndash; {end_fmt}</span>
    <span style="color:#888;">Duration: {end_ts - start_ts:.1f}s</span>
  </div>
  {score_html}
  <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:4px;
              font-size:0.9em; line-height:1.5; white-space:pre-wrap;">
    {transcript}
  </div>
</div>"""
    return html


def render_segment_list(segments: list[dict[str, Any]]) -> str:
    """Render a list of video segments as a grouped HTML display.

    Groups segments by video_id and renders each with a mini player.
    """
    # Group by video
    by_video: dict[str, list[dict[str, Any]]] = {}
    for seg in segments:
        vid = seg.get("video_id", "unknown")
        by_video.setdefault(vid, []).append(seg)

    parts: list[str] = []
    for video_id, segs in by_video.items():
        parts.append(
            f'<h3 style="margin:16px 0 8px;">Video: {video_id}</h3>'
        )
        for seg in segs:
            parts.append(
                render_video_player(
                    video_url=seg.get("url", ""),
                    start_ts=seg.get("start_ts", 0.0),
                    end_ts=seg.get("end_ts", 0.0),
                    transcript=seg.get("transcript", ""),
                    title=seg.get("title", ""),
                    scores={
                        "text": seg.get("text_score", 0.0),
                        "visual": seg.get("visual_score", 0.0),
                        "fused": seg.get("fused_score", 0.0),
                    },
                )
            )

    return "\n".join(parts)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
