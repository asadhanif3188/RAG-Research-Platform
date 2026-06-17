"""RelevanceGrader — uses Claude Haiku to grade document relevance against a query.

Outputs one of three grades:
- RELEVANT: document directly answers the query
- AMBIGUOUS: document is partially relevant, needs decomposition
- IRRELEVANT: document does not address the query
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_GRADING_PROMPT = """\
You are a relevance grading assistant. Given a user query and a retrieved document, \
assess whether the document is relevant to answering the query.

Respond with EXACTLY one JSON object (no markdown fences):
{{"grade": "<RELEVANT|AMBIGUOUS|IRRELEVANT>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Rules:
- RELEVANT: The document directly and sufficiently addresses the query.
- AMBIGUOUS: The document contains some relevant information mixed with irrelevant content.
- IRRELEVANT: The document does not address the query at all.
"""


class RelevanceGrade(StrEnum):
    RELEVANT = "RELEVANT"
    AMBIGUOUS = "AMBIGUOUS"
    IRRELEVANT = "IRRELEVANT"


class GradingResult(BaseModel):
    grade: RelevanceGrade
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class RelevanceGrader:
    """Grades document relevance using Claude Haiku for cost efficiency."""

    def __init__(
        self,
        anthropic_api_key: str = "",
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._api_key = anthropic_api_key
        self._model = model
        self._client: Any = None
        self.total_tokens_used: int = 0

    def connect(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        logger.info("RelevanceGrader ready (model=%s)", self._model)

    async def grade(self, query: str, document_content: str) -> GradingResult:
        """Grade a single document's relevance to the query."""
        import json

        user_message = f"Query: {query}\n\nDocument:\n{document_content[:3000]}"

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_GRADING_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        self.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
        raw = message.content[0].text.strip()

        try:
            parsed = json.loads(raw)
            return GradingResult(
                grade=RelevanceGrade(parsed["grade"]),
                confidence=float(parsed["confidence"]),
                reasoning=parsed.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse grading response: %s — defaulting to AMBIGUOUS", exc)
            return GradingResult(
                grade=RelevanceGrade.AMBIGUOUS,
                confidence=0.5,
                reasoning=f"Parse error: {exc}",
            )

    async def grade_batch(self, query: str, documents: list[str]) -> list[GradingResult]:
        """Grade multiple documents concurrently."""
        import asyncio

        tasks = [self.grade(query, doc) for doc in documents]
        return await asyncio.gather(*tasks)
