"""Unit tests for DocumentDecomposer — Claude API mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from corrective_rag.document_decomposer import DocumentDecomposer


def _make_decomposer() -> tuple[DocumentDecomposer, AsyncMock]:
    decomposer = DocumentDecomposer(anthropic_api_key="sk-test")
    mock_client = AsyncMock()
    decomposer._client = mock_client
    return decomposer, mock_client


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=100, output_tokens=50)
    return msg


class TestDocumentDecomposer:
    @pytest.mark.asyncio
    async def test_decompose_returns_extracted_text(self):
        decomposer, client = _make_decomposer()
        client.messages.create = AsyncMock(
            return_value=_mock_response(
                "RAG combines retrieval with generation for better answers."
            )
        )

        result = await decomposer.decompose(
            "What is RAG?",
            "RAG combines retrieval with generation for better answers. The weather is nice today.",
        )

        assert result == "RAG combines retrieval with generation for better answers."
        assert decomposer.total_tokens_used == 150

    @pytest.mark.asyncio
    async def test_decompose_returns_none_when_no_relevant_content(self):
        decomposer, client = _make_decomposer()
        client.messages.create = AsyncMock(return_value=_mock_response("NO_RELEVANT_CONTENT"))

        result = await decomposer.decompose("What is RAG?", "The weather is sunny today.")

        assert result is None

    @pytest.mark.asyncio
    async def test_decompose_batch_filters_nones(self):
        decomposer, client = _make_decomposer()
        call_count = 0
        responses = ["Relevant part about RAG.", "NO_RELEVANT_CONTENT"]

        async def side_effect(**kwargs):
            nonlocal call_count
            resp = _mock_response(responses[call_count])
            call_count += 1
            return resp

        client.messages.create = AsyncMock(side_effect=side_effect)

        results = await decomposer.decompose_batch("What is RAG?", ["doc1", "doc2"])

        assert len(results) == 1
        assert results[0] == "Relevant part about RAG."
