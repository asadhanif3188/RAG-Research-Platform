"""Unit tests for HyDEQueryExpander — Claude API and embedding service mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from self_rag.hyde import HyDEQueryExpander


def _make_expander() -> tuple[HyDEQueryExpander, AsyncMock, AsyncMock]:
    mock_embedding = AsyncMock()
    mock_embedding.embed = AsyncMock(return_value=[0.1] * 3072)

    expander = HyDEQueryExpander(
        embedding_service=mock_embedding,
        anthropic_api_key="sk-test",
    )
    mock_client = AsyncMock()
    expander._client = mock_client
    return expander, mock_client, mock_embedding


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=60, output_tokens=80)
    return msg


class TestHyDEQueryExpander:
    @pytest.mark.asyncio
    async def test_generate_hypothetical(self):
        expander, client, _ = _make_expander()
        hypothetical = (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with text generation to produce more accurate answers."
        )
        client.messages.create = AsyncMock(return_value=_mock_response(hypothetical))

        result = await expander.generate_hypothetical("What is RAG?")

        assert result == hypothetical
        assert expander.total_tokens_used == 140

    @pytest.mark.asyncio
    async def test_expand_returns_embedding(self):
        expander, client, mock_embedding = _make_expander()
        hypothetical = "RAG combines retrieval with generation."
        client.messages.create = AsyncMock(return_value=_mock_response(hypothetical))

        embedding = await expander.expand("What is RAG?")

        assert len(embedding) == 3072
        # Verify the embedding service was called with the hypothetical document
        mock_embedding.embed.assert_called_once_with(hypothetical)

    @pytest.mark.asyncio
    async def test_expand_tracks_tokens(self):
        expander, client, _ = _make_expander()
        client.messages.create = AsyncMock(return_value=_mock_response("hypothetical doc"))

        await expander.expand("query 1")
        await expander.expand("query 2")

        assert expander.total_tokens_used == 280  # 140 * 2
