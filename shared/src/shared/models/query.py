"""Query request/response Pydantic models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from shared.models.retrieval import RetrievalResult


class PipelineStrategy(StrEnum):
    FASTEST_RAG = "fastest_rag"
    MULTIMODAL_RAG = "multimodal_rag"
    CORRECTIVE_RAG = "corrective_rag"
    SELF_RAG = "self_rag"
    VIDEO_RAG = "video_rag"


class QueryRequest(BaseModel):
    """Incoming query from the UI or API caller."""

    query: str = Field(min_length=1, max_length=4096)
    pipeline: PipelineStrategy = PipelineStrategy.FASTEST_RAG
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)
    use_cache: bool = True
    stream: bool = False


class QueryResponse(BaseModel):
    """Structured answer returned to the caller."""

    query: str
    answer: str
    pipeline: PipelineStrategy
    sources: list[RetrievalResult] = Field(default_factory=list)
    latency_ms: float | None = None
    cache_hit: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
