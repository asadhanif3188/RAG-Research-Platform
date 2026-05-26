"""ProvenanceViewer — Chainlit component that visualises answer source attribution.

Usage in a Chainlit app:
    from ui.components.provenance_viewer import ProvenanceViewer

    viewer = ProvenanceViewer(response.answer, response.metadata.get("provenance", []))
    await viewer.send()

The component renders:
- The generated answer with sentences colour-coded by their best source chunk.
- A sidebar-style attribution table showing chunk_type, document, page, and confidence.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Colour palette for up to 6 unique source chunks
_CHUNK_COLOURS = [
    "#dbeafe",  # blue-100
    "#dcfce7",  # green-100
    "#fef9c3",  # yellow-100
    "#fce7f3",  # pink-100
    "#ede9fe",  # violet-100
    "#ffedd5",  # orange-100
]

_CHUNK_TYPE_ICONS = {
    "text": "📄",
    "image_description": "🖼️",
    "table": "📊",
    "video_transcript": "🎬",
}


@dataclass
class SourceAttribution:
    """One sentence ↔ source mapping rendered by the viewer."""

    sentence: str
    chunk_id: str
    document_id: str
    page_number: int | None
    chunk_type: str
    confidence: float


class ProvenanceViewer:
    """Render answer provenance as an HTML Chainlit element.

    Args:
        answer: The generated answer text.
        provenance: List of provenance dicts from QueryResponse.metadata["provenance"].
        title: Optional section title shown above the highlighted answer.
    """

    def __init__(
        self,
        answer: str,
        provenance: list[dict[str, Any]],
        title: str = "Answer with Source Attribution",
    ) -> None:
        self.answer = answer
        self.provenance = [
            SourceAttribution(
                sentence=p["sentence"],
                chunk_id=p["chunk_id"],
                document_id=p["document_id"],
                page_number=p.get("page_number"),
                chunk_type=p.get("chunk_type", "text"),
                confidence=p.get("confidence", 0.0),
            )
            for p in provenance
        ]
        self.title = title

    def to_html(self) -> str:
        """Render the full provenance view as an HTML string."""
        # Build chunk_id → colour index mapping
        chunk_ids = list(dict.fromkeys(p.chunk_id for p in self.provenance))
        colour_map = {cid: _CHUNK_COLOURS[i % len(_CHUNK_COLOURS)] for i, cid in enumerate(chunk_ids)}

        highlighted_answer = self._highlight_sentences(colour_map)
        attribution_table = self._attribution_table(colour_map)

        return f"""
<div style="font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto;">
  <h3 style="color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">
    {html.escape(self.title)}
  </h3>

  <div style="background: #f8fafc; border-radius: 8px; padding: 16px; margin-bottom: 16px;
              line-height: 1.8; font-size: 15px; color: #334155;">
    {highlighted_answer}
  </div>

  {attribution_table}
</div>
"""

    def _highlight_sentences(self, colour_map: dict[str, str]) -> str:
        """Return the answer HTML with attributed sentences highlighted."""
        # Build sentence → chunk_id mapping
        sentence_map: dict[str, str] = {p.sentence: p.chunk_id for p in self.provenance}

        import re
        sentences = re.split(r"(?<=[.!?])\s+", self.answer.strip())
        parts: list[str] = []

        for sentence in sentences:
            chunk_id = sentence_map.get(sentence)
            if chunk_id and chunk_id in colour_map:
                colour = colour_map[chunk_id]
                parts.append(
                    f'<span style="background-color: {colour}; border-radius: 3px; '
                    f'padding: 1px 3px;" title="Source: {html.escape(chunk_id)}">'
                    f"{html.escape(sentence)}</span>"
                )
            else:
                parts.append(html.escape(sentence))

        return " ".join(parts)

    def _attribution_table(self, colour_map: dict[str, str]) -> str:
        """Return an HTML attribution table for all provenance records."""
        if not self.provenance:
            return '<p style="color: #94a3b8; font-size: 13px;">No source attribution available.</p>'

        rows = ""
        for p in self.provenance:
            icon = _CHUNK_TYPE_ICONS.get(p.chunk_type, "📄")
            colour = colour_map.get(p.chunk_id, "#f1f5f9")
            page_str = str(p.page_number) if p.page_number is not None else "—"
            confidence_pct = f"{p.confidence * 100:.0f}%"
            sentence_preview = p.sentence[:80] + ("…" if len(p.sentence) > 80 else "")

            rows += f"""
  <tr>
    <td style="padding: 6px 10px; background: {colour}; border-radius: 4px;">
      {icon} {html.escape(p.chunk_type.replace("_", " ").title())}
    </td>
    <td style="padding: 6px 10px; font-family: monospace; font-size: 12px; color: #475569;">
      {html.escape(p.document_id)}
    </td>
    <td style="padding: 6px 10px; text-align: center; color: #64748b;">{page_str}</td>
    <td style="padding: 6px 10px; text-align: center; font-weight: 600; color: #0f172a;">
      {confidence_pct}
    </td>
    <td style="padding: 6px 10px; font-size: 13px; color: #475569; font-style: italic;">
      "{html.escape(sentence_preview)}"
    </td>
  </tr>"""

        return f"""
<details style="margin-top: 8px;">
  <summary style="cursor: pointer; font-weight: 600; color: #475569; font-size: 14px;
                  padding: 6px 0;">
    Sources ({len(self.provenance)} attributions)
  </summary>
  <table style="width: 100%; border-collapse: collapse; margin-top: 8px;
                font-size: 13px; border: 1px solid #e2e8f0; border-radius: 6px;">
    <thead>
      <tr style="background: #f1f5f9; text-align: left;">
        <th style="padding: 8px 10px; color: #475569;">Type</th>
        <th style="padding: 8px 10px; color: #475569;">Document</th>
        <th style="padding: 8px 10px; color: #475569; text-align: center;">Page</th>
        <th style="padding: 8px 10px; color: #475569; text-align: center;">Confidence</th>
        <th style="padding: 8px 10px; color: #475569;">Sentence</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</details>"""

    async def send(self, author: str = "RAG Pipeline") -> None:
        """Send the provenance view as a Chainlit HTML element."""
        try:
            import chainlit as cl

            content = self.to_html()
            await cl.Message(
                content=content,
                author=author,
                elements=[
                    cl.Text(
                        name="provenance",
                        content=content,
                        display="inline",
                    )
                ],
            ).send()
        except ImportError:
            logger.warning("chainlit not installed — provenance viewer cannot send messages")
        except Exception as exc:
            logger.error("Failed to send provenance view: %s", exc)

    def to_markdown(self) -> str:
        """Render a simple Markdown summary for non-Chainlit contexts."""
        if not self.provenance:
            return f"**Answer:**\n\n{self.answer}\n\n*No source attribution available.*"

        lines = [f"**Answer:**\n\n{self.answer}\n\n**Sources:**\n"]
        for p in self.provenance:
            icon = _CHUNK_TYPE_ICONS.get(p.chunk_type, "📄")
            page_info = f", page {p.page_number}" if p.page_number else ""
            lines.append(
                f"- {icon} `{p.chunk_type}` — **{p.document_id}**{page_info} "
                f"(confidence: {p.confidence:.0%}): *\"{p.sentence[:60]}…\"*"
            )

        return "\n".join(lines)
