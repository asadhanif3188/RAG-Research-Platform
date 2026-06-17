"""Unit tests for SelfRAGGraph — tests routing logic without running the full graph."""

from __future__ import annotations

from unittest.mock import MagicMock

from corrective_rag.relevance_grader import GradingResult, RelevanceGrade
from self_rag.answer_grader import AnswerQuality
from self_rag.graph import MAX_RETRIES, SelfRAGGraph
from self_rag.hallucination_grader import GroundingGrade


def _make_graph() -> SelfRAGGraph:
    """Create a SelfRAGGraph with all mocked dependencies for logic-only tests."""
    return SelfRAGGraph(
        vector_store=MagicMock(),
        embedding_service=MagicMock(),
        retrieve_or_not=MagicMock(),
        relevance_grader=MagicMock(),
        hallucination_grader=MagicMock(),
        answer_grader=MagicMock(),
        hyde_expander=MagicMock(),
        document_decomposer=MagicMock(),
        query_rewriter=MagicMock(),
        web_searcher=MagicMock(),
    )


class TestOverallGradeDetermination:
    def test_relevant_when_high_confidence(self):
        graph = _make_graph()
        grades = [
            GradingResult(grade=RelevanceGrade.RELEVANT, confidence=0.9),
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.8),
        ]
        assert graph._determine_overall_grade(grades) == RelevanceGrade.RELEVANT

    def test_irrelevant_when_all_irrelevant(self):
        graph = _make_graph()
        grades = [
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.85),
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.9),
        ]
        assert graph._determine_overall_grade(grades) == RelevanceGrade.IRRELEVANT

    def test_ambiguous_when_mixed(self):
        graph = _make_graph()
        grades = [
            GradingResult(grade=RelevanceGrade.AMBIGUOUS, confidence=0.6),
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.7),
        ]
        assert graph._determine_overall_grade(grades) == RelevanceGrade.AMBIGUOUS

    def test_empty_grades_irrelevant(self):
        graph = _make_graph()
        assert graph._determine_overall_grade([]) == RelevanceGrade.IRRELEVANT


class TestRetrieveRouting:
    def test_routes_to_retrieve_when_needed(self):
        graph = _make_graph()
        state = {"retrieve_needed": True}
        assert graph._route_retrieve_decision(state) == "retrieve"

    def test_routes_to_direct_generate_when_not_needed(self):
        graph = _make_graph()
        state = {"retrieve_needed": False}
        assert graph._route_retrieve_decision(state) == "direct_generate"


class TestGroundingRouting:
    def test_routes_to_grade_answer_when_grounded(self):
        graph = _make_graph()
        state = {"grounding_grade": GroundingGrade.GROUNDED, "attempts": 0}
        assert graph._route_on_grounding(state) == "grade_answer"

    def test_routes_to_hyde_when_not_grounded_and_retries_available(self):
        graph = _make_graph()
        state = {"grounding_grade": GroundingGrade.NOT_GROUNDED, "attempts": 0}
        assert graph._route_on_grounding(state) == "hyde_expand"

    def test_routes_to_grade_answer_when_retries_exhausted(self):
        graph = _make_graph()
        state = {"grounding_grade": GroundingGrade.NOT_GROUNDED, "attempts": MAX_RETRIES}
        assert graph._route_on_grounding(state) == "grade_answer"


class TestAnswerQualityRouting:
    def test_routes_to_end_when_addresses_question(self):
        graph = _make_graph()
        state = {"answer_quality": AnswerQuality.ADDRESSES_QUESTION, "attempts": 0}
        assert graph._route_on_answer_quality(state) == "__end__"

    def test_routes_to_rewrite_when_does_not_address(self):
        graph = _make_graph()
        state = {"answer_quality": AnswerQuality.DOES_NOT_ADDRESS, "attempts": 0}
        assert graph._route_on_answer_quality(state) == "rewrite_query"

    def test_routes_to_end_when_retries_exhausted(self):
        graph = _make_graph()
        state = {"answer_quality": AnswerQuality.DOES_NOT_ADDRESS, "attempts": MAX_RETRIES}
        assert graph._route_on_answer_quality(state) == "__end__"
