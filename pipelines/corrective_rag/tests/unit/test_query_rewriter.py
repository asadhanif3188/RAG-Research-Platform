"""Unit tests for QueryRewriter — Claude API mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from corrective_rag.query_rewriter import QueryRewriter


def _make_rewriter() -> tuple[QueryRewriter, AsyncMock]:
    rewriter = QueryRewriter(anthropic_api_key="sk-test")
    mock_client = AsyncMock()
    rewriter._client = mock_client
    return rewriter, mock_client


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=30, output_tokens=20)
    return msg


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_rewrite_expands_query(self):
        rewriter, client = _make_rewriter()
        client.messages.create = AsyncMock(
            return_value=_mock_response(
                "Retrieval-Augmented Generation techniques and applications"
            )
        )

        result = await rewriter.rewrite("What is RAG?")

        assert result == "Retrieval-Augmented Generation techniques and applications"
        assert rewriter.total_tokens_used == 50

    @pytest.mark.asyncio
    async def test_rewrite_calls_claude_with_query(self):
        rewriter, client = _make_rewriter()
        client.messages.create = AsyncMock(return_value=_mock_response("expanded query"))

        await rewriter.rewrite("test query")

        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "test query"
