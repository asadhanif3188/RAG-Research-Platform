"""AnswerGrader — checks if a generated answer addresses the original question.

Returns ADDRESSES_QUESTION if the answer is a meaningful response to the query,
or DOES_NOT_ADDRESS if the answer is off-topic or evasive.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ANSWER_GRADE_PROMPT = """\
You are an answer quality assessment assistant. Given a user query and a generated \
answer, assess whether the answer meaningfully addresses the question.

Respond with EXACTLY one JSON object (no markdown fences):
{{"grade": "<ADDRESSES_QUESTION|DOES_NOT_ADDRESS>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Rules:
- ADDRESSES_QUESTION: The answer provides a direct, relevant response to the query.
- DOES_NOT_ADDRESS: The answer is off-topic, evasive, or does not actually answer \
what was asked.
- An answer that says "I don't have enough information to answer" is acceptable \
if the context truly lacks the answer — grade it ADDRESSES_QUESTION.
- An answer that talks about a different topic entirely is DOES_NOT_ADDRESS.
"""


class AnswerQuality(StrEnum):
    ADDRESSES_QUESTION = "ADDRESSES_QUESTION"
    DOES_NOT_ADDRESS = "DOES_NOT_ADDRESS"


class AnswerGradeResult(BaseModel):
    grade: AnswerQuality
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class AnswerGrader:
    """Grades answer quality using Claude Haiku."""

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
        logger.info("AnswerGrader ready (model=%s)", self._model)

    async def grade(self, query: str, answer: str) -> AnswerGradeResult:
        """Grade whether the answer addresses the original question."""
        user_message = f"Query: {query}\n\nAnswer: {answer}"

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_ANSWER_GRADE_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        self.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
        raw = message.content[0].text.strip()

        try:
            parsed = json.loads(raw)
            return AnswerGradeResult(
                grade=AnswerQuality(parsed["grade"]),
                confidence=float(parsed["confidence"]),
                reasoning=parsed.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse answer grade: %s — defaulting to DOES_NOT_ADDRESS", exc)
            return AnswerGradeResult(
                grade=AnswerQuality.DOES_NOT_ADDRESS,
                confidence=0.5,
                reasoning=f"Parse error: {exc}",
            )
