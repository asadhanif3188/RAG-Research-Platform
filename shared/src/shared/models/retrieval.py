"""Retrieval result and evaluation Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its similarity score."""

    chunk_id: str
    document_id: str
    content: str
    chunk_type: str = "text"
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """RAGAS evaluation metrics for a single query-answer pair."""

    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None

    # RAGAS metrics (0.0–1.0)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    pipeline: str | None = None
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def aggregate_score(self) -> float | None:
        """Mean of all non-None RAGAS metrics."""
        scores = [
            s for s in [
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
                self.context_recall,
            ]
            if s is not None
        ]
        return sum(scores) / len(scores) if scores else None
