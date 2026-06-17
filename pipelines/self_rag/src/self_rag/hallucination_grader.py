"""HallucinationGrader — checks if a generated answer is grounded in retrieved documents.

Returns GROUNDED if every claim in the answer can be traced back to the
provided context, or NOT_GROUNDED if the answer contains unsupported claims.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_GROUNDING_PROMPT = """\
You are a hallucination detection assistant. Given a set of retrieved documents \
and a generated answer, assess whether the answer is fully grounded in the documents.

Respond with EXACTLY one JSON object (no markdown fences):
{{"grade": "<GROUNDED|NOT_GROUNDED>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Rules:
- GROUNDED: Every factual claim in the answer can be traced to the provided documents.
- NOT_GROUNDED: The answer contains claims, facts, or details not present in the documents.
- Focus on factual claims, not phrasing differences. Paraphrasing is acceptable.
- If the answer says "I don't have enough information", that is GROUNDED.
"""


class GroundingGrade(StrEnum):
    GROUNDED = "GROUNDED"
    NOT_GROUNDED = "NOT_GROUNDED"


class GroundingResult(BaseModel):
    grade: GroundingGrade
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class HallucinationGrader:
    """Checks answer grounding using Claude Haiku."""

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
        logger.info("HallucinationGrader ready (model=%s)", self._model)

    async def grade(self, documents: list[str], answer: str) -> GroundingResult:
        """Grade whether the answer is grounded in the provided documents."""
        context = "\n\n---\n\n".join(f"[Document {i}]\n{doc}" for i, doc in enumerate(documents, 1))

        user_message = f"<documents>\n{context}\n</documents>\n\n<answer>\n{answer}\n</answer>"

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_GROUNDING_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        self.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
        raw = message.content[0].text.strip()

        try:
            parsed = json.loads(raw)
            return GroundingResult(
                grade=GroundingGrade(parsed["grade"]),
                confidence=float(parsed["confidence"]),
                reasoning=parsed.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to parse grounding response: %s — defaulting to NOT_GROUNDED", exc
            )
            return GroundingResult(
                grade=GroundingGrade.NOT_GROUNDED,
                confidence=0.5,
                reasoning=f"Parse error: {exc}",
            )
