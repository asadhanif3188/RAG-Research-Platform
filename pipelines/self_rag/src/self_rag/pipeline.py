"""SelfRAGPipeline — wraps the LangGraph Self-RAG graph to match the pipeline interface.

Provides the same `run(QueryRequest) -> QueryResponse` interface as all other
pipelines, enabling seamless API integration via the pipeline router.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from corrective_rag.document_decomposer import DocumentDecomposer
from corrective_rag.query_rewriter import QueryRewriter
from corrective_rag.relevance_grader import RelevanceGrader
from corrective_rag.web_searcher import WebSearcher
from self_rag.answer_grader import AnswerGrader
from self_rag.graph import SelfRAGGraph
from self_rag.hallucination_grader import HallucinationGrader
from self_rag.hyde import HyDEQueryExpander
from self_rag.retrieve_or_not import RetrieveOrNot
from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse
from shared.models.retrieval import RetrievalResult

if TYPE_CHECKING:
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)


class SelfRAGPipeline:
    """Self-RAG pipeline with adaptive retrieval, hallucination detection, and HyDE.

    Args:
        vector_store: Connected VectorStoreClient.
        embedding_service: Connected EmbeddingService.
        anthropic_api_key: Key for Claude API.
        tavily_api_key: Key for Tavily web search.
        generation_model: Claude model for answer generation.
        grading_model: Claude model for all grading decisions (defaults to Haiku).
        langfuse_client: Optional LangFuse client for tracing.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        anthropic_api_key: str = "",
        tavily_api_key: str = "",
        generation_model: str = "claude-sonnet-4-6",
        grading_model: str = "claude-haiku-4-5-20251001",
        langfuse_client: Any = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._anthropic_api_key = anthropic_api_key
        self._tavily_api_key = tavily_api_key
        self._generation_model = generation_model
        self._grading_model = grading_model
        self._langfuse = langfuse_client

        # Self-RAG specific components
        self._retrieve_or_not = RetrieveOrNot(
            anthropic_api_key=anthropic_api_key,
            model=grading_model,
        )
        self._hallucination_grader = HallucinationGrader(
            anthropic_api_key=anthropic_api_key,
            model=grading_model,
        )
        self._answer_grader = AnswerGrader(
            anthropic_api_key=anthropic_api_key,
            model=grading_model,
        )
        self._hyde = HyDEQueryExpander(
            embedding_service=embedding_service,
            anthropic_api_key=anthropic_api_key,
            model=grading_model,
        )

        # Reused from CRAG
        self._grader = RelevanceGrader(
            anthropic_api_key=anthropic_api_key,
            model=grading_model,
        )
        self._decomposer = DocumentDecomposer(
            anthropic_api_key=anthropic_api_key,
            model=grading_model,
        )
        self._rewriter = QueryRewriter(
            anthropic_api_key=anthropic_api_key,
            model=grading_model,
        )
        self._web_searcher = WebSearcher(tavily_api_key=tavily_api_key)

        self._graph: SelfRAGGraph | None = None

    def connect(self) -> None:
        """Initialize all sub-components and build the graph."""
        self._retrieve_or_not.connect()
        self._hallucination_grader.connect()
        self._answer_grader.connect()
        self._hyde.connect()
        self._grader.connect()
        self._decomposer.connect()
        self._rewriter.connect()
        self._web_searcher.connect()

        self._graph = SelfRAGGraph(
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            retrieve_or_not=self._retrieve_or_not,
            relevance_grader=self._grader,
            hallucination_grader=self._hallucination_grader,
            answer_grader=self._answer_grader,
            hyde_expander=self._hyde,
            document_decomposer=self._decomposer,
            query_rewriter=self._rewriter,
            web_searcher=self._web_searcher,
            anthropic_api_key=self._anthropic_api_key,
            generation_model=self._generation_model,
            langfuse_client=self._langfuse,
        )
        self._graph.connect()
        logger.info("SelfRAGPipeline fully connected")

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the Self-RAG pipeline for a single query."""
        assert self._graph is not None, "Call connect() before run()"

        start = time.perf_counter()

        result = await self._graph.run(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters or None,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        sources = [
            RetrievalResult(
                chunk_id=doc.get("chunk_id", ""),
                document_id=doc.get("document_id", ""),
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
            )
            for doc in result.get("documents", [])
        ]

        return QueryResponse(
            query=request.query,
            answer=result.get("answer", ""),
            pipeline=PipelineStrategy.SELF_RAG,
            sources=sources,
            latency_ms=elapsed_ms,
            cache_hit=False,
            metadata={
                "generation_model": self._generation_model,
                "grading_model": self._grading_model,
                "decision_path": result.get("decision_path", []),
                "overall_grade": result.get("overall_grade", ""),
                "grounding_grade": result.get("grounding_grade", ""),
                "answer_quality": result.get("answer_quality", ""),
                "attempts": result.get("attempts", 0),
                "node_timings": result.get("node_timings", {}),
                "retrieve_or_not_tokens": self._retrieve_or_not.total_tokens_used,
                "grader_tokens": self._grader.total_tokens_used,
                "hallucination_grader_tokens": self._hallucination_grader.total_tokens_used,
                "answer_grader_tokens": self._answer_grader.total_tokens_used,
                "hyde_tokens": self._hyde.total_tokens_used,
                "decomposer_tokens": self._decomposer.total_tokens_used,
                "rewriter_tokens": self._rewriter.total_tokens_used,
            },
        )
