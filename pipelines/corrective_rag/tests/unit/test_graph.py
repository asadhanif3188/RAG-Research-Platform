"""Unit tests for CRAGGraph — tests the overall grade determination logic."""

from __future__ import annotations

from corrective_rag.graph import CRAGGraph
from corrective_rag.relevance_grader import GradingResult, RelevanceGrade


class TestOverallGradeDetermination:
    """Test _determine_overall_grade without running the full graph."""

    def _make_graph(self) -> CRAGGraph:
        """Create a CRAGGraph with all None dependencies for logic-only tests."""
        from unittest.mock import MagicMock

        return CRAGGraph(
            vector_store=MagicMock(),
            embedding_service=MagicMock(),
            relevance_grader=MagicMock(),
            document_decomposer=MagicMock(),
            query_rewriter=MagicMock(),
            web_searcher=MagicMock(),
        )

    def test_relevant_when_high_confidence_relevant(self):
        graph = self._make_graph()
        grades = [
            GradingResult(grade=RelevanceGrade.RELEVANT, confidence=0.9),
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.8),
        ]
        assert graph._determine_overall_grade(grades) == RelevanceGrade.RELEVANT

    def test_irrelevant_when_all_irrelevant(self):
        graph = self._make_graph()
        grades = [
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.85),
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.9),
        ]
        assert graph._determine_overall_grade(grades) == RelevanceGrade.IRRELEVANT

    def test_ambiguous_when_mixed(self):
        graph = self._make_graph()
        grades = [
            GradingResult(grade=RelevanceGrade.AMBIGUOUS, confidence=0.6),
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.7),
        ]
        assert graph._determine_overall_grade(grades) == RelevanceGrade.AMBIGUOUS

    def test_irrelevant_when_empty(self):
        graph = self._make_graph()
        assert graph._determine_overall_grade([]) == RelevanceGrade.IRRELEVANT

    def test_ambiguous_when_relevant_low_confidence(self):
        graph = self._make_graph()
        grades = [
            GradingResult(grade=RelevanceGrade.RELEVANT, confidence=0.5),
            GradingResult(grade=RelevanceGrade.IRRELEVANT, confidence=0.8),
        ]
        # RELEVANT with confidence < 0.7 doesn't count as high-confidence relevant
        assert graph._determine_overall_grade(grades) == RelevanceGrade.AMBIGUOUS
