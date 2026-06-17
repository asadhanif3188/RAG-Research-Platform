"""RetrieveOrNot — decides whether document retrieval is needed for a query.

Simple factual or computational queries (e.g., "What is 2+2?") can be answered
directly by the LLM, while knowledge-intensive queries need retrieval.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DECISION_PROMPT = """\
You are a retrieval decision assistant. Given a user query, decide whether \
the query requires retrieving documents from a knowledge base to answer accurately.

Respond with EXACTLY one JSON object (no markdown fences):
{{"retrieve": <true|false>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Rules:
- retrieve=true: The query asks about specific facts, research, documents, or \
domain knowledge that requires external context.
- retrieve=false: The query is a simple computation, general knowledge, greeting, \
or can be answered confidently without retrieval.

Examples of retrieve=false: "What is 2+2?", "Hello!", "Translate 'cat' to Spanish"
Examples of retrieve=true: "What is RAG?", "Summarize the findings on climate change", \
"What does our policy say about refunds?"
"""


class RetrieveDecision(BaseModel):
    retrieve: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class RetrieveOrNot:
    """Decides whether retrieval is needed using Claude Haiku."""

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
        logger.info("RetrieveOrNot ready (model=%s)", self._model)

    async def decide(self, query: str) -> RetrieveDecision:
        """Decide whether retrieval is needed for the given query."""
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_DECISION_PROMPT,
            messages=[{"role": "user", "content": query}],
        )

        self.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
        raw = message.content[0].text.strip()

        try:
            parsed = json.loads(raw)
            return RetrieveDecision(
                retrieve=bool(parsed["retrieve"]),
                confidence=float(parsed["confidence"]),
                reasoning=parsed.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to parse retrieve decision: %s — defaulting to retrieve=True", exc
            )
            return RetrieveDecision(
                retrieve=True,
                confidence=0.5,
                reasoning=f"Parse error, defaulting to retrieve: {exc}",
            )
