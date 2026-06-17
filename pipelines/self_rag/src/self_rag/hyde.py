"""HyDEQueryExpander — Hypothetical Document Embeddings for improved retrieval.

Instead of embedding the raw query, HyDE generates a hypothetical document that
would answer the query, then embeds *that* document. The resulting embedding
often captures the answer's semantic space better than the original query.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

_HYDE_PROMPT = """\
You are a knowledgeable research assistant. Given a user's question, write a \
short, factual paragraph (3-5 sentences) that would serve as an ideal answer \
to the question. Write as if you are excerpting from a reliable reference document.

Do NOT say "I think" or "In my opinion". Write authoritatively and factually. \
Output ONLY the paragraph, nothing else.\
"""


class HyDEQueryExpander:
    """Generates hypothetical documents and embeds them for improved retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        anthropic_api_key: str = "",
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._embedding_service = embedding_service
        self._api_key = anthropic_api_key
        self._model = model
        self._client: Any = None
        self.total_tokens_used: int = 0

    def connect(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        logger.info("HyDEQueryExpander ready (model=%s)", self._model)

    async def generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=_HYDE_PROMPT,
            messages=[{"role": "user", "content": query}],
        )

        self.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
        return message.content[0].text.strip()

    async def expand(self, query: str) -> list[float]:
        """Generate a hypothetical document and return its embedding.

        Returns the embedding vector of the hypothetical document, which can
        be used for retrieval instead of the original query embedding.
        """
        hypothetical = await self.generate_hypothetical(query)
        logger.info(
            "HyDE generated hypothetical (%d chars) for: %s...",
            len(hypothetical),
            query[:40],
        )
        return await self._embedding_service.embed(hypothetical)
