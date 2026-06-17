"""CRAGGraph — LangGraph StateGraph implementing Corrective RAG.

Flow:
  retrieve → grade_documents → (branch on overall grade)
    RELEVANT    → generate → END
    AMBIGUOUS   → decompose_docs → generate → END
    IRRELEVANT  → rewrite_query → web_search → generate → END

Every node is traced via LangFuse for full observability.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, StateGraph

from corrective_rag.relevance_grader import GradingResult, RelevanceGrade

if TYPE_CHECKING:
    from corrective_rag.document_decomposer import DocumentDecomposer
    from corrective_rag.query_rewriter import QueryRewriter
    from corrective_rag.relevance_grader import RelevanceGrader
    from corrective_rag.web_searcher import WebSearcher
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)


class CRAGState(TypedDict, total=False):
    """State flowing through the CRAG graph."""

    query: str
    top_k: int
    filters: dict[str, Any]
    documents: list[dict[str, Any]]
    grades: list[dict[str, Any]]
    overall_grade: str
    decomposed_docs: list[str]
    rewritten_query: str
    web_results: list[dict[str, Any]]
    context: str
    answer: str
    decision_path: list[str]
    node_timings: dict[str, float]


class CRAGGraph:
    """Corrective RAG graph with relevance grading and adaptive retrieval.

    Args:
        vector_store: Connected VectorStoreClient.
        embedding_service: Connected EmbeddingService.
        relevance_grader: Connected RelevanceGrader.
        document_decomposer: Connected DocumentDecomposer.
        query_rewriter: Connected QueryRewriter.
        web_searcher: Connected WebSearcher.
        anthropic_api_key: Key for Claude generation.
        generation_model: Claude model for answer generation.
        langfuse_client: Optional LangFuse client for tracing.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        relevance_grader: RelevanceGrader,
        document_decomposer: DocumentDecomposer,
        query_rewriter: QueryRewriter,
        web_searcher: WebSearcher,
        anthropic_api_key: str = "",
        generation_model: str = "claude-sonnet-4-6",
        langfuse_client: Any = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._grader = relevance_grader
        self._decomposer = document_decomposer
        self._rewriter = query_rewriter
        self._web_searcher = web_searcher
        self._anthropic_api_key = anthropic_api_key
        self._generation_model = generation_model
        self._langfuse = langfuse_client
        self._llm_client: Any = None
        self._graph = self._build_graph()

    def connect(self) -> None:
        import anthropic

        self._llm_client = anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)
        logger.info("CRAGGraph ready (generation_model=%s)", self._generation_model)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(CRAGState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("generate", self._generate)
        graph.add_node("decompose_docs", self._decompose_docs)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("web_search", self._web_search)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_documents")

        graph.add_conditional_edges(
            "grade_documents",
            self._route_on_grade,
            {
                "generate": "generate",
                "decompose_docs": "decompose_docs",
                "rewrite_query": "rewrite_query",
            },
        )

        graph.add_edge("decompose_docs", "generate")
        graph.add_edge("rewrite_query", "web_search")
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def _trace_span(self, trace: Any, name: str) -> Any:
        """Create a LangFuse span if tracing is enabled."""
        if trace is not None:
            return trace.span(name=name)
        return None

    async def _retrieve(self, state: CRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "retrieve")

        query_embedding = await self._embedding_service.embed(state["query"])
        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=state.get("top_k", 5),
            filters=state.get("filters") or None,
        )

        documents = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]

        elapsed = (time.perf_counter() - start) * 1000
        if span:
            span.end(output={"num_docs": len(documents), "latency_ms": elapsed})

        logger.info("Retrieved %d documents (%.1fms)", len(documents), elapsed)
        return {
            "documents": documents,
            "decision_path": [*state.get("decision_path", []), "retrieve"],
            "node_timings": {**state.get("node_timings", {}), "retrieve": elapsed},
        }

    async def _grade_documents(self, state: CRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "grade_documents")

        documents = state.get("documents", [])
        if not documents:
            elapsed = (time.perf_counter() - start) * 1000
            if span:
                span.end(output={"overall_grade": "IRRELEVANT", "latency_ms": elapsed})
            return {
                "grades": [],
                "overall_grade": RelevanceGrade.IRRELEVANT,
                "decision_path": [*state.get("decision_path", []), "grade_documents:IRRELEVANT"],
                "node_timings": {**state.get("node_timings", {}), "grade_documents": elapsed},
            }

        contents = [doc["content"] for doc in documents]
        grading_results = await self._grader.grade_batch(state["query"], contents)

        grades = [
            {"grade": r.grade, "confidence": r.confidence, "reasoning": r.reasoning}
            for r in grading_results
        ]

        overall_grade = self._determine_overall_grade(grading_results)
        elapsed = (time.perf_counter() - start) * 1000

        if span:
            span.end(
                output={
                    "grades": [g["grade"] for g in grades],
                    "overall_grade": overall_grade,
                    "latency_ms": elapsed,
                }
            )

        logger.info("Grading complete: overall=%s (%.1fms)", overall_grade, elapsed)
        return {
            "grades": grades,
            "overall_grade": overall_grade,
            "decision_path": [
                *state.get("decision_path", []),
                f"grade_documents:{overall_grade}",
            ],
            "node_timings": {**state.get("node_timings", {}), "grade_documents": elapsed},
        }

    def _determine_overall_grade(self, grades: list[GradingResult]) -> str:
        """Determine the overall grade from individual document grades.

        Strategy:
        - If ANY document is RELEVANT with high confidence, use RELEVANT.
        - If all are IRRELEVANT, use IRRELEVANT.
        - Otherwise, use AMBIGUOUS.
        """
        if not grades:
            return RelevanceGrade.IRRELEVANT

        relevant_count = sum(
            1 for g in grades if g.grade == RelevanceGrade.RELEVANT and g.confidence >= 0.7
        )
        irrelevant_count = sum(1 for g in grades if g.grade == RelevanceGrade.IRRELEVANT)

        if relevant_count > 0:
            return RelevanceGrade.RELEVANT
        if irrelevant_count == len(grades):
            return RelevanceGrade.IRRELEVANT
        return RelevanceGrade.AMBIGUOUS

    def _route_on_grade(self, state: CRAGState) -> str:
        """Conditional edge: route based on overall grade."""
        grade = state.get("overall_grade", RelevanceGrade.IRRELEVANT)
        match grade:
            case RelevanceGrade.RELEVANT:
                return "generate"
            case RelevanceGrade.AMBIGUOUS:
                return "decompose_docs"
            case _:
                return "rewrite_query"

    async def _decompose_docs(self, state: CRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "decompose_docs")

        documents = state.get("documents", [])
        contents = [doc["content"] for doc in documents]
        decomposed = await self._decomposer.decompose_batch(state["query"], contents)

        elapsed = (time.perf_counter() - start) * 1000
        if span:
            span.end(output={"num_decomposed": len(decomposed), "latency_ms": elapsed})

        logger.info(
            "Decomposed %d docs → %d relevant excerpts (%.1fms)",
            len(documents),
            len(decomposed),
            elapsed,
        )
        return {
            "decomposed_docs": decomposed,
            "decision_path": [*state.get("decision_path", []), "decompose_docs"],
            "node_timings": {**state.get("node_timings", {}), "decompose_docs": elapsed},
        }

    async def _rewrite_query(self, state: CRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "rewrite_query")

        rewritten = await self._rewriter.rewrite(state["query"])

        elapsed = (time.perf_counter() - start) * 1000
        if span:
            span.end(output={"rewritten_query": rewritten, "latency_ms": elapsed})

        logger.info(
            "Query rewritten: '%s' → '%s' (%.1fms)", state["query"][:40], rewritten[:40], elapsed
        )
        return {
            "rewritten_query": rewritten,
            "decision_path": [*state.get("decision_path", []), "rewrite_query"],
            "node_timings": {**state.get("node_timings", {}), "rewrite_query": elapsed},
        }

    async def _web_search(self, state: CRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "web_search")

        search_query = state.get("rewritten_query", state["query"])
        results = await self._web_searcher.search(search_query)

        web_results = [r.model_dump() for r in results]

        elapsed = (time.perf_counter() - start) * 1000
        if span:
            span.end(output={"num_results": len(web_results), "latency_ms": elapsed})

        logger.info("Web search returned %d results (%.1fms)", len(web_results), elapsed)
        return {
            "web_results": web_results,
            "decision_path": [*state.get("decision_path", []), "web_search"],
            "node_timings": {**state.get("node_timings", {}), "web_search": elapsed},
        }

    async def _generate(self, state: CRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "generate")

        context = self._build_context(state)

        system_prompt = (
            "You are a precise research assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain enough information to "
            "answer, say so clearly. Be concise and factual. Cite sources by referencing "
            "[Source N]."
        )

        user_message = (
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {state['query']}\n\n"
            "Answer based only on the context above:"
        )

        message = await self._llm_client.messages.create(
            model=self._generation_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = message.content[0].text
        elapsed = (time.perf_counter() - start) * 1000

        if span:
            span.end(
                output={
                    "answer_length": len(answer),
                    "tokens": message.usage.input_tokens + message.usage.output_tokens,
                    "latency_ms": elapsed,
                }
            )

        logger.info("Generated answer (%d chars, %.1fms)", len(answer), elapsed)
        return {
            "answer": answer,
            "context": context,
            "decision_path": [*state.get("decision_path", []), "generate"],
            "node_timings": {**state.get("node_timings", {}), "generate": elapsed},
        }

    def _build_context(self, state: CRAGState) -> str:
        """Build context string from the best available sources."""
        overall_grade = state.get("overall_grade", "")
        parts: list[str] = []

        if overall_grade == RelevanceGrade.RELEVANT:
            # Use original retrieved documents
            for i, doc in enumerate(state.get("documents", []), 1):
                parts.append(f"[Source {i}] (score={doc['score']:.3f})\n{doc['content']}")

        elif overall_grade == RelevanceGrade.AMBIGUOUS:
            # Use decomposed excerpts
            for i, excerpt in enumerate(state.get("decomposed_docs", []), 1):
                parts.append(f"[Source {i}] (decomposed excerpt)\n{excerpt}")

        else:
            # Use web search results
            for i, result in enumerate(state.get("web_results", []), 1):
                parts.append(
                    f"[Source {i}] ({result.get('url', 'web')})\n{result.get('content', '')}"
                )

        return "\n\n---\n\n".join(parts) if parts else "(no context available)"

    async def run(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> CRAGState:
        """Execute the full CRAG graph.

        Returns the final state with answer, decision path, and timings.
        """
        trace = None
        if self._langfuse:
            trace = self._langfuse.trace(name="crag_pipeline", input={"query": query})

        initial_state: CRAGState = {
            "query": query,
            "top_k": top_k,
            "filters": filters or {},
            "documents": [],
            "grades": [],
            "overall_grade": "",
            "decomposed_docs": [],
            "rewritten_query": "",
            "web_results": [],
            "context": "",
            "answer": "",
            "decision_path": [],
            "node_timings": {},
        }

        result = await self._graph.ainvoke(initial_state)

        if trace:
            trace.update(
                output={
                    "answer": result.get("answer", "")[:200],
                    "decision_path": result.get("decision_path", []),
                    "total_latency_ms": sum(result.get("node_timings", {}).values()),
                }
            )

        return result
