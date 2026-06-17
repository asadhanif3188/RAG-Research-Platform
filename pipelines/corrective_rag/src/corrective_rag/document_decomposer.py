"""DocumentDecomposer — extracts relevant sub-sections from ambiguous documents.

When a document is graded AMBIGUOUS, this component uses Claude to identify
and extract only the parts that are relevant to the user's query.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """\
You are an information extraction assistant. Given a user query and a document, \
extract ONLY the sentences and passages that are directly relevant to the query. \
Preserve the original wording. If no part is relevant, respond with "NO_RELEVANT_CONTENT".

Output the extracted text only, no preamble or explanation.\
"""


class DocumentDecomposer:
    """Extracts relevant sub-sections from partially relevant documents."""

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
        logger.info("DocumentDecomposer ready (model=%s)", self._model)

    async def decompose(self, query: str, document_content: str) -> str | None:
        """Extract relevant sub-sections from a document.

        Returns the extracted text, or None if no relevant content found.
        """
        user_message = f"Query: {query}\n\nDocument:\n{document_content[:4000]}"

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_DECOMPOSE_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        self.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
        text = message.content[0].text.strip()

        if text == "NO_RELEVANT_CONTENT":
            return None
        return text

    async def decompose_batch(self, query: str, documents: list[str]) -> list[str]:
        """Decompose multiple documents, filtering out empty results."""
        import asyncio

        tasks = [self.decompose(query, doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
