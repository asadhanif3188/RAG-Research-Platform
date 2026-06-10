"""Unit tests for RAGASRunner with mock LLM responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.eval.ragas_runner import RAGASRunner
from shared.models.retrieval import EvalResult


class TestRAGASRunner:
    @pytest.fixture
    def mock_eval_result_df(self):
        """Pandas-like mock for RAGAS evaluate() result."""
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "faithfulness": 0.85,
                    "answer_relevancy": 0.90,
                    "context_precision": 0.75,
                    "context_recall": 0.80,
                }
            ]
        )
        result = MagicMock()
        result.to_pandas.return_value = df
        return result

    @pytest.mark.asyncio
    async def test_evaluate_returns_eval_result(self, mock_eval_result_df):
        runner = RAGASRunner(llm=MagicMock(), embeddings=MagicMock())
        runner._initialized = True

        with (
            patch("ragas.evaluate", return_value=mock_eval_result_df),
            patch("datasets.Dataset"),
        ):
            result = await runner.evaluate(
                query="What is RAG?",
                answer="RAG stands for Retrieval-Augmented Generation.",
                contexts=["RAG combines retrieval with generation."],
                ground_truth="Retrieval-Augmented Generation.",
                pipeline="fastest_rag",
            )

        assert isinstance(result, EvalResult)
        assert result.faithfulness == pytest.approx(0.85)
        assert result.answer_relevancy == pytest.approx(0.90)
        assert result.context_precision == pytest.approx(0.75)
        assert result.context_recall == pytest.approx(0.80)
        assert result.pipeline == "fastest_rag"

    @pytest.mark.asyncio
    async def test_evaluate_without_ground_truth(self, mock_eval_result_df):
        runner = RAGASRunner(llm=MagicMock(), embeddings=MagicMock())
        runner._initialized = True

        # Remove context_recall from df since no ground_truth
        import pandas as pd

        df_no_recall = pd.DataFrame(
            [
                {
                    "faithfulness": 0.85,
                    "answer_relevancy": 0.90,
                    "context_precision": 0.75,
                }
            ]
        )
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = df_no_recall

        with (
            patch("ragas.evaluate", return_value=mock_result),
            patch("datasets.Dataset"),
        ):
            result = await runner.evaluate(
                query="What is RAG?",
                answer="RAG is a retrieval technique.",
                contexts=["Some context."],
            )

        assert result.context_recall is None

    @pytest.mark.asyncio
    async def test_evaluate_handles_exception_gracefully(self):
        runner = RAGASRunner(llm=MagicMock(), embeddings=MagicMock())
        runner._initialized = True

        with (
            patch("ragas.evaluate", side_effect=RuntimeError("API error")),
            patch("datasets.Dataset"),
        ):
            result = await runner.evaluate(query="q", answer="a", contexts=["c"])

        assert isinstance(result, EvalResult)
        assert result.faithfulness is None
        assert result.answer_relevancy is None

    @pytest.mark.asyncio
    async def test_evaluate_batch_returns_list(self, mock_eval_result_df):
        runner = RAGASRunner(llm=MagicMock(), embeddings=MagicMock())
        runner._initialized = True

        import pandas as pd

        df = pd.DataFrame(
            [
                {"faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": 0.7},
                {"faithfulness": 0.7, "answer_relevancy": 0.85, "context_precision": 0.65},
            ]
        )
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = df

        samples = [
            {"query": "q1", "answer": "a1", "contexts": ["c1"]},
            {"query": "q2", "answer": "a2", "contexts": ["c2"]},
        ]

        with (
            patch("ragas.evaluate", return_value=mock_result),
            patch("datasets.Dataset"),
        ):
            results = await runner.evaluate_batch(samples, pipeline="test")

        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)
        assert results[0].faithfulness == pytest.approx(0.8)
        assert results[1].faithfulness == pytest.approx(0.7)
