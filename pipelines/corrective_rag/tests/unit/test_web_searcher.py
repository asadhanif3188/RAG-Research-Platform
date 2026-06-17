"""Unit tests for WebSearcher — Tavily API mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from corrective_rag.web_searcher import WebSearcher, WebSearchResult


class TestWebSearcher:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        searcher = WebSearcher(tavily_api_key="tvly-test")
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "RAG Overview",
                    "url": "https://example.com/rag",
                    "content": "RAG combines retrieval with generation.",
                    "score": 0.95,
                },
                {
                    "title": "RAG Tutorial",
                    "url": "https://example.com/tutorial",
                    "content": "Step-by-step guide to RAG.",
                    "score": 0.88,
                },
            ]
        }
        searcher._client = mock_client

        results = await searcher.search("What is RAG?")

        assert len(results) == 2
        assert isinstance(results[0], WebSearchResult)
        assert results[0].title == "RAG Overview"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_deduplicates_urls(self):
        searcher = WebSearcher(tavily_api_key="tvly-test")
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "RAG 1", "url": "https://example.com/rag", "content": "v1", "score": 0.9},
                {"title": "RAG 2", "url": "https://example.com/rag", "content": "v2", "score": 0.8},
            ]
        }
        searcher._client = mock_client

        results = await searcher.search("What is RAG?")

        assert len(results) == 1
        assert results[0].title == "RAG 1"

    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self):
        searcher = WebSearcher(tavily_api_key="tvly-test")
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        searcher._client = mock_client

        results = await searcher.search("nonexistent topic")

        assert results == []
