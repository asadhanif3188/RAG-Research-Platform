"""WebSearcher — Tavily API integration for web search fallback.

Used when document grading returns IRRELEVANT, indicating the vector store
does not contain relevant information for the query.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchResult(BaseModel):
    """A single web search result."""

    title: str
    url: str
    content: str
    score: float = Field(ge=0.0, le=1.0, default=0.0)


class WebSearcher:
    """Searches the web via Tavily API when local retrieval fails."""

    def __init__(
        self,
        tavily_api_key: str = "",
        max_results: int = 5,
    ) -> None:
        self._api_key = tavily_api_key
        self._max_results = max_results
        self._client: Any = None

    def connect(self) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=self._api_key)
        logger.info("WebSearcher ready (max_results=%d)", self._max_results)

    async def search(self, query: str) -> list[WebSearchResult]:
        """Execute a web search and return deduplicated results."""
        import asyncio

        # Tavily's Python client is synchronous — run in executor
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._client.search(
                query=query,
                max_results=self._max_results,
                search_depth="advanced",
                include_answer=False,
            ),
        )

        results = []
        seen_urls: set[str] = set()

        for item in raw.get("results", []):
            url = item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=url,
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                )
            )

        logger.info("Web search returned %d results for: %s...", len(results), query[:60])
        return results
