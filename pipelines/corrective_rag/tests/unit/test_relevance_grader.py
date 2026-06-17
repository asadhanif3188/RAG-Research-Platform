"""Unit tests for RelevanceGrader — Claude API mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corrective_rag.relevance_grader import GradingResult, RelevanceGrade, RelevanceGrader


def _make_grader() -> tuple[RelevanceGrader, AsyncMock]:
    grader = RelevanceGrader(anthropic_api_key="sk-test")
    mock_client = AsyncMock()
    grader._client = mock_client
    return grader, mock_client


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=50, output_tokens=20)
    return msg


class TestRelevanceGrader:
    @pytest.mark.asyncio
    async def test_grade_relevant(self):
        grader, client = _make_grader()
        resp = json.dumps({"grade": "RELEVANT", "confidence": 0.95, "reasoning": "Direct match"})
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade(
            "What is RAG?", "RAG stands for Retrieval-Augmented Generation."
        )

        assert isinstance(result, GradingResult)
        assert result.grade == RelevanceGrade.RELEVANT
        assert result.confidence == 0.95
        assert grader.total_tokens_used == 70

    @pytest.mark.asyncio
    async def test_grade_ambiguous(self):
        grader, client = _make_grader()
        resp = json.dumps({"grade": "AMBIGUOUS", "confidence": 0.6, "reasoning": "Partial match"})
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade("What is RAG?", "ML techniques include various approaches.")

        assert result.grade == RelevanceGrade.AMBIGUOUS
        assert result.confidence == 0.6

    @pytest.mark.asyncio
    async def test_grade_irrelevant(self):
        grader, client = _make_grader()
        resp = json.dumps({"grade": "IRRELEVANT", "confidence": 0.9, "reasoning": "No match"})
        client.messages.create = AsyncMock(return_value=_mock_response(resp))

        result = await grader.grade("What is RAG?", "The weather today is sunny.")

        assert result.grade == RelevanceGrade.IRRELEVANT
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_grade_handles_malformed_json(self):
        grader, client = _make_grader()
        client.messages.create = AsyncMock(return_value=_mock_response("not json at all"))

        result = await grader.grade("test query", "test doc")

        assert result.grade == RelevanceGrade.AMBIGUOUS
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_grade_batch(self):
        grader, client = _make_grader()
        responses = [
            json.dumps({"grade": "RELEVANT", "confidence": 0.9, "reasoning": "match"}),
            json.dumps({"grade": "IRRELEVANT", "confidence": 0.85, "reasoning": "no match"}),
        ]
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            resp = _mock_response(responses[call_count])
            call_count += 1
            return resp

        client.messages.create = AsyncMock(side_effect=side_effect)

        results = await grader.grade_batch("What is RAG?", ["doc1", "doc2"])

        assert len(results) == 2
        assert results[0].grade == RelevanceGrade.RELEVANT
        assert results[1].grade == RelevanceGrade.IRRELEVANT
