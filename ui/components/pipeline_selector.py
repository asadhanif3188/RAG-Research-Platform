"""PipelineSelector — Chainlit component for selecting and monitoring RAG pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class PipelineStatus(StrEnum):
    READY = "ready"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class PipelineInfo:
    """Metadata for a single pipeline shown in the selector."""

    name: str
    strategy: str  # PipelineStrategy value
    description: str
    status: PipelineStatus = PipelineStatus.READY


# Registry of all available pipelines with display metadata
PIPELINE_REGISTRY: list[PipelineInfo] = [
    PipelineInfo(
        name="Naive RAG (Fastest)",
        strategy="fastest_rag",
        description="Baseline embed→retrieve→generate with Redis semantic cache",
    ),
    PipelineInfo(
        name="Multimodal RAG",
        strategy="multimodal_rag",
        description="Text + image + table retrieval with provenance tracking",
    ),
    PipelineInfo(
        name="Corrective RAG (CRAG)",
        strategy="corrective_rag",
        description="LangGraph with relevance grading and web search fallback",
    ),
    PipelineInfo(
        name="Self-RAG",
        strategy="self_rag",
        description="Agentic retrieval with hallucination checking and HyDE",
    ),
    PipelineInfo(
        name="Video RAG",
        strategy="video_rag",
        description="Whisper + CLIP dual retrieval with Neo4j knowledge graph",
    ),
]

_STATUS_ICONS = {
    PipelineStatus.READY: "🟢",
    PipelineStatus.LOADING: "🟡",
    PipelineStatus.ERROR: "🔴",
}


class PipelineSelector:
    """Render a pipeline selector and store the selection in Chainlit session.

    Args:
        statuses: Optional dict mapping strategy name to PipelineStatus overrides.
        selected: Currently selected pipeline strategy name.
    """

    def __init__(
        self,
        statuses: dict[str, PipelineStatus] | None = None,
        selected: str = "fastest_rag",
    ) -> None:
        self.statuses = statuses or {}
        self.selected = selected

    def _get_status(self, strategy: str) -> PipelineStatus:
        return self.statuses.get(strategy, PipelineStatus.READY)

    def to_html(self) -> str:
        """Render the pipeline selector as an HTML card grid."""
        cards = ""
        for info in PIPELINE_REGISTRY:
            status = self._get_status(info.strategy)
            icon = _STATUS_ICONS[status]
            is_selected = info.strategy == self.selected
            border = "2px solid #3b82f6" if is_selected else "1px solid #e2e8f0"
            bg = "#eff6ff" if is_selected else "#ffffff"
            check = " ✓" if is_selected else ""

            cards += f"""
      <div style="border: {border}; border-radius: 8px; padding: 12px 16px;
                  background: {bg}; cursor: pointer; transition: all 0.2s;"
           data-pipeline="{info.strategy}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong style="color: #1e293b; font-size: 14px;">{info.name}{check}</strong>
          <span title="{status.value}">{icon}</span>
        </div>
        <p style="color: #64748b; font-size: 12px; margin: 4px 0 0 0;">
          {info.description}
        </p>
      </div>"""

        return f"""
<div style="font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto;">
  <h3 style="color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;
             font-size: 16px;">
    Select Pipeline
  </h3>
  <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
              gap: 10px; margin-top: 12px;">
    {cards}
  </div>
  <p style="color: #94a3b8; font-size: 12px; margin-top: 8px;">
    Active: <strong>{self.selected}</strong>
  </p>
</div>
"""

    async def send(self, author: str = "Pipeline Selector") -> None:
        """Send the selector as a Chainlit message."""
        try:
            import chainlit as cl

            content = self.to_html()
            await cl.Message(content=content, author=author).send()
        except ImportError:
            logger.warning("chainlit not installed — pipeline selector cannot send")
        except Exception as exc:
            logger.error("Failed to send pipeline selector: %s", exc)

    @staticmethod
    async def store_selection(strategy: str) -> None:
        """Store the selected pipeline in the Chainlit user session."""
        try:
            import chainlit as cl

            cl.user_session.set("selected_pipeline", strategy)
            logger.info("Pipeline selection stored: %s", strategy)
        except ImportError:
            logger.warning("chainlit not installed — cannot store selection")

    @staticmethod
    async def get_selection() -> str:
        """Retrieve the selected pipeline from Chainlit session (default: fastest_rag)."""
        try:
            import chainlit as cl

            return cl.user_session.get("selected_pipeline") or "fastest_rag"
        except ImportError:
            return "fastest_rag"
