"""Integration tests for the full Self-RAG graph — one test per decision branch.

All external APIs (Claude, Tavily, vector store, embeddings) are mocked, but the
LangGraph execution is real — testing actual graph compilation, routing,
and state transitions through each decision path.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corrective_rag.document_decomposer import DocumentDecomposer
from corrective_rag.query_rewriter import QueryRewriter
from corrective_rag.relevance_grader import RelevanceGrader
from corrective_rag.web_searcher import WebSearcher
from self_rag.answer_grader import AnswerGrader
from self_rag.graph import SelfRAGGraph
from self_rag.hallucination_grader import HallucinationGrader
from self_rag.hyde import HyDEQueryExpander
from self_rag.retrieve_or_not import RetrieveOrNot
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
    retrieve_decision: bool = True,
    grader_grades: list[dict] | None = None,
    grounding_grade: str = "GROUNDED",
    answer_quality: str = "ADDRESSES_QUESTION",
    decomposed_text: str | None = None,
    rewritten_query: str = "expanded query",
    web_results: list[dict] | None = None,
    hyde_hypothetical: str = "Hypothetical document about RAG.",
) -> SelfRAGGraph:
    """Build a SelfRAGGraph with fully mocked components."""

    if grader_grades is None:
        grader_grades = [
            {"grade": "RELEVANT", "confidence": 0.95, "reasoning": "Direct match"},
        ]

    # Mock vector store
    mock_vs = AsyncMock()
    mock_vs.search = AsyncMock(return_value=_make_retrieval_results(len(grader_grades)))

    # Mock embedding service
    mock_emb = AsyncMock()
    mock_emb.embed = AsyncMock(return_value=[0.1] * 3072)

    # Mock RetrieveOrNot
    retrieve_or_not = RetrieveOrNot(anthropic_api_key="sk-test")
    retrieve_or_not._client = AsyncMock()
    retrieve_resp = json.dumps(
        {
            "retrieve": retrieve_decision,
            "confidence": 0.9,
            "reasoning": "test",
        }
    )
    retrieve_or_not._client.messages.create = AsyncMock(
        return_value=_mock_claude_response(retrieve_resp)
    )

    # Mock relevance grader
    grader = RelevanceGrader(anthropic_api_key="sk-test")
    grader._client = AsyncMock()
    grade_idx = 0

    async def grade_side_effect(**kwargs):
        nonlocal grade_idx
        grade_data = grader_grades[grade_idx % len(grader_grades)]
        grade_idx += 1
        return _mock_claude_response(json.dumps(grade_data))

    grader._client.messages.create = AsyncMock(side_effect=grade_side_effect)

    # Mock hallucination grader
    hall_grader = HallucinationGrader(anthropic_api_key="sk-test")
    hall_grader._client = AsyncMock()
    grounding_resp = json.dumps(
        {
            "grade": grounding_grade,
            "confidence": 0.9,
            "reasoning": "test",
        }
    )
    hall_grader._client.messages.create = AsyncMock(
        return_value=_mock_claude_response(grounding_resp)
    )

    # Mock answer grader
    ans_grader = AnswerGrader(anthropic_api_key="sk-test")
    ans_grader._client = AsyncMock()
    answer_resp = json.dumps(
        {
            "grade": answer_quality,
            "confidence": 0.9,
            "reasoning": "test",
        }
    )
    ans_grader._client.messages.create = AsyncMock(return_value=_mock_claude_response(answer_resp))

    # Mock HyDE expander
    hyde = HyDEQueryExpander(
        embedding_service=mock_emb,
        anthropic_api_key="sk-test",
    )
    hyde._client = AsyncMock()
    hyde._client.messages.create = AsyncMock(return_value=_mock_claude_response(hyde_hypothetical))

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

    graph = SelfRAGGraph(
        vector_store=mock_vs,
        embedding_service=mock_emb,
        retrieve_or_not=retrieve_or_not,
        relevance_grader=grader,
        hallucination_grader=hall_grader,
        answer_grader=ans_grader,
        hyde_expander=hyde,
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


class TestRetrieveOrNotYesBranch:
    """retrieve_or_not: YES → retrieve → grade → generate → grounding → answer → END."""

    @pytest.mark.asyncio
    async def test_retrieval_path_with_relevant_grounded_answer(self):
        graph = _build_graph(
            retrieve_decision=True,
            grader_grades=[{"grade": "RELEVANT", "confidence": 0.95, "reasoning": "match"}],
            grounding_grade="GROUNDED",
            answer_quality="ADDRESSES_QUESTION",
        )

        result = await graph.run("What is RAG?")

        assert result["answer"] == "RAG is Retrieval-Augmented Generation."
        assert "retrieve_or_not:YES" in result["decision_path"]
        assert "retrieve" in result["decision_path"]
        assert "grade_relevance:RELEVANT" in result["decision_path"]
        assert "generate" in result["decision_path"]
        assert "grade_grounding:GROUNDED" in result["decision_path"]
        assert "grade_answer:ADDRESSES_QUESTION" in result["decision_path"]
        assert "direct_generate" not in result["decision_path"]


class TestRetrieveOrNotNoBranch:
    """retrieve_or_not: NO → direct_generate → END."""

    @pytest.mark.asyncio
    async def test_no_retrieval_path(self):
        graph = _build_graph(retrieve_decision=False)

        result = await graph.run("What is 2+2?")

        assert result["answer"] == "RAG is Retrieval-Augmented Generation."
        assert "retrieve_or_not:NO" in result["decision_path"]
        assert "direct_generate" in result["decision_path"]
        # Should NOT go through retrieval or grading
        assert "retrieve" not in result["decision_path"]
        assert "grade_relevance:RELEVANT" not in result["decision_path"]


class TestGroundingFailureHyDERetry:
    """NOT_GROUNDED → hyde_expand → retrieve → grade → generate → grade_grounding (retry)."""

    @pytest.mark.asyncio
    async def test_grounding_failure_triggers_hyde_retry(self):
        # First grounding check: NOT_GROUNDED, then after retry: GROUNDED
        hall_grader = HallucinationGrader(anthropic_api_key="sk-test")
        hall_grader._client = AsyncMock()
        call_count = 0

        async def grounding_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_claude_response(
                    json.dumps(
                        {"grade": "NOT_GROUNDED", "confidence": 0.8, "reasoning": "unsupported"}
                    )
                )
            return _mock_claude_response(
                json.dumps({"grade": "GROUNDED", "confidence": 0.9, "reasoning": "supported"})
            )

        hall_grader._client.messages.create = AsyncMock(side_effect=grounding_side_effect)

        # Build graph manually to inject the stateful hallucination grader
        graph = _build_graph(grounding_grade="NOT_GROUNDED")
        # Replace with our stateful mock
        graph._hallucination_grader = hall_grader

        result = await graph.run("What is RAG?")

        assert result["answer"] == "RAG is Retrieval-Augmented Generation."
        path = result["decision_path"]
        assert any("hyde_expand" in step for step in path)
        assert result.get("attempts", 0) >= 1


class TestAnswerQualityFailureRewrite:
    """DOES_NOT_ADDRESS → rewrite_query → web_search → generate → ... (retry)."""

    @pytest.mark.asyncio
    async def test_answer_quality_failure_triggers_rewrite(self):
        ans_grader = AnswerGrader(anthropic_api_key="sk-test")
        ans_grader._client = AsyncMock()
        call_count = 0

        async def answer_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_claude_response(
                    json.dumps(
                        {"grade": "DOES_NOT_ADDRESS", "confidence": 0.85, "reasoning": "off topic"}
                    )
                )
            return _mock_claude_response(
                json.dumps(
                    {"grade": "ADDRESSES_QUESTION", "confidence": 0.9, "reasoning": "on topic"}
                )
            )

        ans_grader._client.messages.create = AsyncMock(side_effect=answer_side_effect)

        graph = _build_graph(answer_quality="DOES_NOT_ADDRESS")
        graph._answer_grader = ans_grader

        result = await graph.run("What is RAG?")

        path = result["decision_path"]
        assert any("rewrite_query" in step for step in path)


class TestFullSelfRAGAllPaths:
    """Full Self-RAG graph with all decision nodes exercised."""

    @pytest.mark.asyncio
    async def test_full_relevant_grounded_path(self):
        graph = _build_graph(
            retrieve_decision=True,
            grader_grades=[
                {"grade": "RELEVANT", "confidence": 0.95, "reasoning": "match"},
                {"grade": "RELEVANT", "confidence": 0.9, "reasoning": "good"},
            ],
            grounding_grade="GROUNDED",
            answer_quality="ADDRESSES_QUESTION",
        )

        result = await graph.run("What is RAG?", top_k=5)

        assert result["answer"]
        assert len(result["decision_path"]) >= 5
        assert result["node_timings"]
        assert "retrieve_or_not" in result["node_timings"]

    @pytest.mark.asyncio
    async def test_irrelevant_web_search_path(self):
        graph = _build_graph(
            retrieve_decision=True,
            grader_grades=[
                {"grade": "IRRELEVANT", "confidence": 0.9, "reasoning": "no match"},
                {"grade": "IRRELEVANT", "confidence": 0.85, "reasoning": "unrelated"},
            ],
            grounding_grade="GROUNDED",
            answer_quality="ADDRESSES_QUESTION",
        )

        result = await graph.run("What is the weather today?")

        path = result["decision_path"]
        assert any("grade_relevance:IRRELEVANT" in step for step in path)
        assert "rewrite_query" in path
        assert "web_search" in path
