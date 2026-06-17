"""QueryRewriter — rewrites queries to improve web search quality.

When retrieved documents are graded IRRELEVANT, the original query is rewritten
into a more search-engine-friendly form before falling back to web search.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a search query optimizer. Given a user's question, rewrite it as a \
concise, keyword-rich web search query that will yield the most relevant results. \
Expand acronyms, add context terms, and remove conversational filler.

Output ONLY the rewritten query, nothing else.\
"""


class QueryRewriter:
    """Rewrites user queries for better web search performance."""

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
        logger.info("QueryRewriter ready (model=%s)", self._model)

    async def rewrite(self, query: str) -> str:
        """Rewrite a query for web search."""
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_REWRITE_PROMPT,
            messages=[{"role": "user", "content": query}],
        )

        self.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
        return message.content[0].text.strip()
