"""Unit tests for RetrieveOrNot — Claude API mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from self_rag.retrieve_or_not import RetrieveDecision, RetrieveOrNot


def _make_decider() -> tuple[RetrieveOrNot, AsyncMock]:
    decider = RetrieveOrNot(anthropic_api_key="sk-test")
    mock_client = AsyncMock()
    decider._client = mock_client
    return decider, mock_client


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=40, output_tokens=20)
    return msg


class TestRetrieveOrNot:
    @pytest.mark.asyncio
    async def test_decide_retrieve_true(self):
        decider, client = _make_decider()
        resp = json.dumps({"retrieve": True, "confidence": 0.95, "reasoning": "Knowledge query"})
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await decider.decide("What is RAG?")

        assert isinstance(result, RetrieveDecision)
        assert result.retrieve is True
        assert result.confidence == 0.95
        assert decider.total_tokens_used == 60

    @pytest.mark.asyncio
    async def test_decide_retrieve_false(self):
        decider, client = _make_decider()
        resp = json.dumps({"retrieve": False, "confidence": 0.9, "reasoning": "Simple math"})
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await decider.decide("What is 2+2?")

        assert result.retrieve is False
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_decide_handles_malformed_json(self):
        decider, client = _make_decider()
        client.messages.create = AsyncMock(return_value=_mock_response("not json"))

        result = await decider.decide("test query")

        # Defaults to retrieve=True on parse failure
        assert result.retrieve is True
        assert result.confidence == 0.5
