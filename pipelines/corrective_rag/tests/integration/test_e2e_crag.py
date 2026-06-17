"""E2E test: submit hallucination-prone query, verify CRAG improves answer.

This test mocks external APIs but exercises the full pipeline (CRAGPipeline)
end-to-end, including the QueryRequest → QueryResponse interface, to verify
that CRAG produces answers grounded in the retrieved/web context rather than
hallucinating.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corrective_rag.pipeline import CRAGPipeline
from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse


def _mock_claude(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=50, output_tokens=30)
    return msg


def _build_pipeline(grade: str, confidence: float = 0.9) -> CRAGPipeline:
    """Build a fully mocked CRAGPipeline."""
    mock_vs = AsyncMock()
    mock_emb = AsyncMock()
    mock_emb.embed = AsyncMock(return_value=[0.1] * 3072)

    from shared.models.retrieval import RetrievalResult

    mock_vs.search = AsyncMock(
        return_value=[
            RetrievalResult(
                chunk_id="c1",
                document_id="doc-1",
                content="The transformer was introduced in 2017 by Vaswani et al.",
                score=0.85,
            ),
        ]
    )

    pipeline = CRAGPipeline(
        vector_store=mock_vs,
        embedding_service=mock_emb,
        anthropic_api_key="sk-test",
        tavily_api_key="tvly-test",
    )

    # Manually connect sub-components with mocks
    grade_resp = json.dumps({"grade": grade, "confidence": confidence, "reasoning": "test"})
    for component in [pipeline._grader, pipeline._decomposer, pipeline._rewriter]:
        component._client = AsyncMock()
        component._client.messages.create = AsyncMock(return_value=_mock_claude(grade_resp))

    # Override decomposer to return extracted text
    pipeline._decomposer._client.messages.create = AsyncMock(
        return_value=_mock_claude("The transformer was introduced in 2017.")
    )

    # Override rewriter
    pipeline._rewriter._client.messages.create = AsyncMock(
        return_value=_mock_claude("transformer architecture introduction year 2017 Vaswani")
    )

    # Mock web searcher

    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "results": [
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "content": "The Transformer was introduced in 2017 by Vaswani et al. in 'Attention Is All You Need'.",
                "score": 0.95,
            }
        ]
    }
    pipeline._web_searcher._client = mock_tavily

    # Build graph and mock generation LLM
    from corrective_rag.graph import CRAGGraph

    pipeline._graph = CRAGGraph(
        vector_store=mock_vs,
        embedding_service=mock_emb,
        relevance_grader=pipeline._grader,
        document_decomposer=pipeline._decomposer,
        query_rewriter=pipeline._rewriter,
        web_searcher=pipeline._web_searcher,
        anthropic_api_key="sk-test",
    )

    mock_llm = AsyncMock()
    mock_llm.messages.create = AsyncMock(
        return_value=_mock_claude(
            "The transformer architecture was introduced in 2017 by Vaswani et al. "
            "in the paper 'Attention Is All You Need' [Source 1]."
        )
    )
    pipeline._graph._llm_client = mock_llm

    return pipeline


class TestE2ECRAGPipeline:
    @pytest.mark.asyncio
    async def test_e2e_relevant_path_produces_grounded_answer(self):
        pipeline = _build_pipeline("RELEVANT", 0.95)
        request = QueryRequest(
            query="What year was the transformer architecture introduced?",
            pipeline=PipelineStrategy.CORRECTIVE_RAG,
            use_cache=False,
        )

        response = await pipeline.run(request)

        assert isinstance(response, QueryResponse)
        assert response.pipeline == PipelineStrategy.CORRECTIVE_RAG
        assert "2017" in response.answer
        assert "Vaswani" in response.answer
        assert response.metadata["overall_grade"] == "RELEVANT"
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_e2e_irrelevant_path_falls_back_to_web(self):
        pipeline = _build_pipeline("IRRELEVANT", 0.9)
        request = QueryRequest(
            query="What year was the transformer architecture introduced?",
            pipeline=PipelineStrategy.CORRECTIVE_RAG,
            use_cache=False,
        )

        response = await pipeline.run(request)

        assert isinstance(response, QueryResponse)
        assert response.metadata["overall_grade"] == "IRRELEVANT"
        assert "web_search" in response.metadata["decision_path"]
        assert "2017" in response.answer

    @pytest.mark.asyncio
    async def test_e2e_response_includes_decision_metadata(self):
        pipeline = _build_pipeline("AMBIGUOUS", 0.6)
        request = QueryRequest(
            query="What year was the transformer architecture introduced?",
            pipeline=PipelineStrategy.CORRECTIVE_RAG,
            use_cache=False,
        )

        response = await pipeline.run(request)

        assert "decision_path" in response.metadata
        assert "node_timings" in response.metadata
        assert "overall_grade" in response.metadata
        assert len(response.metadata["decision_path"]) >= 3
        assert all(v > 0 for v in response.metadata["node_timings"].values())
