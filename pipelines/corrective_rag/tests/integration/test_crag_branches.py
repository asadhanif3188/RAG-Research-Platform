"""Integration tests for the full CRAG graph — one test per decision branch.

All external APIs (Claude, Tavily, vector store) are mocked, but the
LangGraph execution is real — testing the actual graph compilation, routing,
and state transitions.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corrective_rag.document_decomposer import DocumentDecomposer
from corrective_rag.graph import CRAGGraph
from corrective_rag.query_rewriter import QueryRewriter
from corrective_rag.relevance_grader import RelevanceGrader
from corrective_rag.web_searcher import WebSearcher
from shared.models.retrieval import RetrievalResult


def _mock_claude_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=50, output_tokens=30)
    return msg


def _make_retrieval_results(n: int = 3) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"chunk-{i}",
            document_id="doc-001",
            content=f"Content about RAG technique #{i}.",
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


def _build_graph(
    grader_grades: list[dict],
    decomposed_text: str | None = None,
    rewritten_query: str = "expanded query",
    web_results: list[dict] | None = None,
) -> CRAGGraph:
    """Build a CRAGGraph with fully mocked components."""

    # Mock vector store
    mock_vs = AsyncMock()
    mock_vs.search = AsyncMock(return_value=_make_retrieval_results(len(grader_grades)))

    # Mock embedding service
    mock_emb = AsyncMock()
    mock_emb.embed = AsyncMock(return_value=[0.1] * 3072)

    # Mock relevance grader
    grader = RelevanceGrader(anthropic_api_key="sk-test")
    grader._client = AsyncMock()
    call_idx = 0

    async def grade_side_effect(**kwargs):
        nonlocal call_idx
        grade_data = grader_grades[call_idx % len(grader_grades)]
        call_idx += 1
        return _mock_claude_response(json.dumps(grade_data))

    grader._client.messages.create = AsyncMock(side_effect=grade_side_effect)

    # Mock document decomposer
    decomposer = DocumentDecomposer(anthropic_api_key="sk-test")
    decomposer._client = AsyncMock()
    decomposer._client.messages.create = AsyncMock(
        return_value=_mock_claude_response(decomposed_text or "NO_RELEVANT_CONTENT")
    )

    # Mock query rewriter
    rewriter = QueryRewriter(anthropic_api_key="sk-test")
    rewriter._client = AsyncMock()
    rewriter._client.messages.create = AsyncMock(
        return_value=_mock_claude_response(rewritten_query)
    )

    # Mock web searcher
    searcher = WebSearcher(tavily_api_key="tvly-test")
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "results": web_results
        or [
            {
                "title": "Web Result",
                "url": "https://example.com",
                "content": "Web content about RAG.",
                "score": 0.85,
            }
        ]
    }
    searcher._client = mock_tavily

    graph = CRAGGraph(
        vector_store=mock_vs,
        embedding_service=mock_emb,
        relevance_grader=grader,
        document_decomposer=decomposer,
        query_rewriter=rewriter,
        web_searcher=searcher,
        anthropic_api_key="sk-test",
    )

    # Mock the generation LLM client
    mock_llm = AsyncMock()
    mock_llm.messages.create = AsyncMock(
        return_value=_mock_claude_response("RAG is Retrieval-Augmented Generation.")
    )
    graph._llm_client = mock_llm

    return graph


class TestCRAGRelevantBranch:
    """RELEVANT path: retrieve → grade → generate → END."""

    @pytest.mark.asyncio
    async def test_relevant_branch_generates_directly(self):
        grades = [
            {"grade": "RELEVANT", "confidence": 0.95, "reasoning": "Direct match"},
            {"grade": "RELEVANT", "confidence": 0.9, "reasoning": "Good match"},
        ]
        graph = _build_graph(grades)

        result = await graph.run("What is RAG?")

        assert result["answer"] == "RAG is Retrieval-Augmented Generation."
        assert result["overall_grade"] == "RELEVANT"
        assert "retrieve" in result["decision_path"]
        assert "grade_documents:RELEVANT" in result["decision_path"]
        assert "generate" in result["decision_path"]
        # Should NOT go through decompose or web search
        assert "decompose_docs" not in result["decision_path"]
        assert "web_search" not in result["decision_path"]


class TestCRAGAmbiguousBranch:
    """AMBIGUOUS path: retrieve → grade → decompose → generate → END."""

    @pytest.mark.asyncio
    async def test_ambiguous_branch_decomposes_then_generates(self):
        grades = [
            {"grade": "AMBIGUOUS", "confidence": 0.6, "reasoning": "Partial"},
            {"grade": "IRRELEVANT", "confidence": 0.7, "reasoning": "Off topic"},
        ]
        graph = _build_graph(grades, decomposed_text="Extracted relevant part about RAG.")

        result = await graph.run("What is RAG?")

        assert result["answer"] == "RAG is Retrieval-Augmented Generation."
        assert result["overall_grade"] == "AMBIGUOUS"
        assert "decompose_docs" in result["decision_path"]
        assert "generate" in result["decision_path"]
        assert "web_search" not in result["decision_path"]


class TestCRAGIrrelevantBranch:
    """IRRELEVANT path: retrieve → grade → rewrite → web_search → generate → END."""

    @pytest.mark.asyncio
    async def test_irrelevant_branch_falls_back_to_web(self):
        grades = [
            {"grade": "IRRELEVANT", "confidence": 0.9, "reasoning": "No match"},
            {"grade": "IRRELEVANT", "confidence": 0.85, "reasoning": "Unrelated"},
        ]
        graph = _build_graph(grades, rewritten_query="Retrieval-Augmented Generation overview")

        result = await graph.run("What is RAG?")

        assert result["answer"] == "RAG is Retrieval-Augmented Generation."
        assert result["overall_grade"] == "IRRELEVANT"
        assert "rewrite_query" in result["decision_path"]
        assert "web_search" in result["decision_path"]
        assert "generate" in result["decision_path"]
        assert "decompose_docs" not in result["decision_path"]
        assert result["rewritten_query"] == "Retrieval-Augmented Generation overview"
