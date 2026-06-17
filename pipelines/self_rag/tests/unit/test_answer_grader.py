"""Unit tests for AnswerGrader — Claude API mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from self_rag.answer_grader import AnswerGrader, AnswerGradeResult, AnswerQuality


def _make_grader() -> tuple[AnswerGrader, AsyncMock]:
    grader = AnswerGrader(anthropic_api_key="sk-test")
    mock_client = AsyncMock()
    grader._client = mock_client
    return grader, mock_client


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=50, output_tokens=20)
    return msg


class TestAnswerGrader:
    @pytest.mark.asyncio
    async def test_grade_addresses_question(self):
        grader, client = _make_grader()
        resp = json.dumps(
            {"grade": "ADDRESSES_QUESTION", "confidence": 0.93, "reasoning": "Direct answer"}
        )
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade("What is RAG?", "RAG is Retrieval-Augmented Generation.")

        assert isinstance(result, AnswerGradeResult)
        assert result.grade == AnswerQuality.ADDRESSES_QUESTION
        assert result.confidence == 0.93
        assert grader.total_tokens_used == 70

    @pytest.mark.asyncio
    async def test_grade_does_not_address(self):
        grader, client = _make_grader()
        resp = json.dumps(
            {"grade": "DOES_NOT_ADDRESS", "confidence": 0.87, "reasoning": "Off topic"}
        )
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade("What is RAG?", "The weather today is sunny and warm.")

        assert result.grade == AnswerQuality.DOES_NOT_ADDRESS
        assert result.confidence == 0.87

    @pytest.mark.asyncio
    async def test_grade_handles_malformed_json(self):
        grader, client = _make_grader()
        client.messages.create = AsyncMock(return_value=_mock_response("garbage"))

        result = await grader.grade("query", "answer")

        assert result.grade == AnswerQuality.DOES_NOT_ADDRESS
        assert result.confidence == 0.5
