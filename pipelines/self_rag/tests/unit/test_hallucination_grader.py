"""Unit tests for HallucinationGrader — Claude API mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from self_rag.hallucination_grader import GroundingGrade, GroundingResult, HallucinationGrader


def _make_grader() -> tuple[HallucinationGrader, AsyncMock]:
    grader = HallucinationGrader(anthropic_api_key="sk-test")
    mock_client = AsyncMock()
    grader._client = mock_client
    return grader, mock_client


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=80, output_tokens=25)
    return msg


class TestHallucinationGrader:
    @pytest.mark.asyncio
    async def test_grade_grounded(self):
        grader, client = _make_grader()
        resp = json.dumps(
            {"grade": "GROUNDED", "confidence": 0.92, "reasoning": "All claims supported"}
        )
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade(
            ["RAG stands for Retrieval-Augmented Generation."],
            "RAG is Retrieval-Augmented Generation.",
        )

        assert isinstance(result, GroundingResult)
        assert result.grade == GroundingGrade.GROUNDED
        assert result.confidence == 0.92
        assert grader.total_tokens_used == 105

    @pytest.mark.asyncio
    async def test_grade_not_grounded(self):
        grader, client = _make_grader()
        resp = json.dumps(
            {"grade": "NOT_GROUNDED", "confidence": 0.88, "reasoning": "Unsupported claim"}
        )
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade(
            ["The weather is sunny today."],
            "RAG was invented in 2019 by Facebook AI Research.",
        )

        assert result.grade == GroundingGrade.NOT_GROUNDED
        assert result.confidence == 0.88

    @pytest.mark.asyncio
    async def test_grade_handles_malformed_json(self):
        grader, client = _make_grader()
        client.messages.create = AsyncMock(return_value=_mock_response("invalid"))

        result = await grader.grade(["doc"], "answer")

        assert result.grade == GroundingGrade.NOT_GROUNDED
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_grade_multiple_documents(self):
        grader, client = _make_grader()
        resp = json.dumps({"grade": "GROUNDED", "confidence": 0.95, "reasoning": "Well supported"})
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade(
            ["Doc 1 about RAG.", "Doc 2 about embeddings.", "Doc 3 about retrieval."],
            "RAG uses embeddings for retrieval.",
        )

        assert result.grade == GroundingGrade.GROUNDED
        # Verify the message was created with concatenated documents
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "[Document 1]" in user_msg
        assert "[Document 3]" in user_msg
