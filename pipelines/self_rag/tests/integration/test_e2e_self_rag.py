"""End-to-end integration test for the SelfRAGPipeline wrapper.

Tests the full pipeline interface (QueryRequest → QueryResponse) with all
external dependencies mocked. Verifies the pipeline produces correct
metadata including decision paths, grades, and token counts.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from self_rag.pipeline import SelfRAGPipeline
from shared.models.query import PipelineStrategy, QueryRequest
from shared.models.retrieval import RetrievalResult


def _mock_claude_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=50, output_tokens=30)
    return msg


class TestSelfRAGPipelineE2E:
    @pytest.mark.asyncio
    async def test_pipeline_run_returns_query_response(self):
        # Build pipeline with mocked internals
        mock_vs = AsyncMock()
        mock_vs.search = AsyncMock(
            return_value=[
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="RAG combines retrieval with generation.",
                    score=0.95,
                )
            ]
        )

        mock_emb = AsyncMock()
        mock_emb.embed = AsyncMock(return_value=[0.1] * 3072)

        pipeline = SelfRAGPipeline(
            vector_store=mock_vs,
            embedding_service=mock_emb,
            anthropic_api_key="sk-test",
            tavily_api_key="tvly-test",
        )

        # Mock all internal component clients
        for component in [
            pipeline._retrieve_or_not,
            pipeline._hallucination_grader,
            pipeline._answer_grader,
            pipeline._hyde,
            pipeline._grader,
            pipeline._decomposer,
            pipeline._rewriter,
        ]:
            component._client = AsyncMock()

        # RetrieveOrNot: YES
        pipeline._retrieve_or_not._client.messages.create = AsyncMock(
            return_value=_mock_claude_response(
                json.dumps({"retrieve": True, "confidence": 0.9, "reasoning": "knowledge query"})
            )
        )

        # Relevance: RELEVANT
        pipeline._grader._client.messages.create = AsyncMock(
            return_value=_mock_claude_response(
                json.dumps({"grade": "RELEVANT", "confidence": 0.95, "reasoning": "match"})
            )
        )

        # Grounding: GROUNDED
        pipeline._hallucination_grader._client.messages.create = AsyncMock(
            return_value=_mock_claude_response(
                json.dumps({"grade": "GROUNDED", "confidence": 0.92, "reasoning": "supported"})
            )
        )

        # Answer quality: ADDRESSES_QUESTION
        pipeline._answer_grader._client.messages.create = AsyncMock(
            return_value=_mock_claude_response(
                json.dumps(
                    {"grade": "ADDRESSES_QUESTION", "confidence": 0.93, "reasoning": "direct"}
                )
            )
        )

        # Web searcher mock
        pipeline._web_searcher._client = MagicMock()

        # Build the graph
        pipeline._graph = None
        from self_rag.graph import SelfRAGGraph

        pipeline._graph = SelfRAGGraph(
            vector_store=mock_vs,
            embedding_service=mock_emb,
            retrieve_or_not=pipeline._retrieve_or_not,
            relevance_grader=pipeline._grader,
            hallucination_grader=pipeline._hallucination_grader,
            answer_grader=pipeline._answer_grader,
            hyde_expander=pipeline._hyde,
            document_decomposer=pipeline._decomposer,
            query_rewriter=pipeline._rewriter,
            web_searcher=pipeline._web_searcher,
            anthropic_api_key="sk-test",
        )

        # Mock the LLM client for generation
        mock_llm = AsyncMock()
        mock_llm.messages.create = AsyncMock(
            return_value=_mock_claude_response("RAG is Retrieval-Augmented Generation.")
        )
        pipeline._graph._llm_client = mock_llm

        # Run the pipeline
        request = QueryRequest(query="What is RAG?", top_k=3, use_cache=False)
        response = await pipeline.run(request)

        assert response.query == "What is RAG?"
        assert response.answer == "RAG is Retrieval-Augmented Generation."
        assert response.pipeline == PipelineStrategy.SELF_RAG
        assert response.latency_ms > 0
        assert len(response.sources) == 1
        assert response.sources[0].chunk_id == "c1"

        # Verify metadata
        meta = response.metadata
        assert "decision_path" in meta
        assert "grounding_grade" in meta
        assert "answer_quality" in meta
        assert meta["generation_model"] == "claude-sonnet-4-6"
        assert meta["grading_model"] == "claude-haiku-4-5-20251001"
