"""SelfRAGGraph — LangGraph StateGraph implementing Self-RAG with adaptive retry.

Flow:
  retrieve_or_not →
    NO  → direct_generate → END
    YES → retrieve → grade_relevance → (branch)
      RELEVANT  → generate → grade_grounding → (branch)
        GROUNDED    → grade_answer → (branch)
          ADDRESSES   → END
          FAILS       → rewrite_query → retrieve → ... (max 2 retries)
        NOT_GROUNDED → hyde_expand → retrieve → ... (max 2 retries)
      AMBIGUOUS → decompose_docs → generate → grade_grounding → ...
      IRRELEVANT → rewrite_query → web_search → generate → grade_grounding → ...

Every node is traced via LangFuse for full observability.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, StateGraph

from corrective_rag.relevance_grader import GradingResult, RelevanceGrade
from self_rag.answer_grader import AnswerQuality
from self_rag.hallucination_grader import GroundingGrade

if TYPE_CHECKING:
    from corrective_rag.document_decomposer import DocumentDecomposer
    from corrective_rag.query_rewriter import QueryRewriter
    from corrective_rag.relevance_grader import RelevanceGrader
    from corrective_rag.web_searcher import WebSearcher
    from self_rag.answer_grader import AnswerGrader
    from self_rag.hallucination_grader import HallucinationGrader
    from self_rag.hyde import HyDEQueryExpander
    from self_rag.retrieve_or_not import RetrieveOrNot
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class SelfRAGState(TypedDict, total=False):
    """State flowing through the Self-RAG graph."""

    query: str
    top_k: int
    filters: dict[str, Any]
    retrieve_needed: bool
    documents: list[dict[str, Any]]
    grades: list[dict[str, Any]]
    overall_grade: str
    decomposed_docs: list[str]
    rewritten_query: str
    web_results: list[dict[str, Any]]
    context: str
    answer: str
    grounding_grade: str
    grounding_confidence: float
    answer_quality: str
    answer_quality_confidence: float
    attempts: int
    decision_path: list[str]
    node_timings: dict[str, float]


class SelfRAGGraph:
    """Self-RAG graph with adaptive retrieval, hallucination detection, and retry loops.

    Extends CRAG with three additional decision nodes:
    - RetrieveOrNot: skip retrieval for simple queries
    - HallucinationGrader: verify answer grounding
    - AnswerGrader: verify answer addresses the question

    Plus HyDE query expansion for improved retrieval on retry.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        retrieve_or_not: RetrieveOrNot,
        relevance_grader: RelevanceGrader,
        hallucination_grader: HallucinationGrader,
        answer_grader: AnswerGrader,
        hyde_expander: HyDEQueryExpander,
        document_decomposer: DocumentDecomposer,
        query_rewriter: QueryRewriter,
        web_searcher: WebSearcher,
        anthropic_api_key: str = "",
        generation_model: str = "claude-sonnet-4-6",
        langfuse_client: Any = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._retrieve_or_not = retrieve_or_not
        self._grader = relevance_grader
        self._hallucination_grader = hallucination_grader
        self._answer_grader = answer_grader
        self._hyde = hyde_expander
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
        logger.info("SelfRAGGraph ready (generation_model=%s)", self._generation_model)

    # ── Graph construction ──────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(SelfRAGState)

        graph.add_node("retrieve_or_not", self._decide_retrieve)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_relevance", self._grade_relevance)
        graph.add_node("generate", self._generate)
        graph.add_node("direct_generate", self._direct_generate)
        graph.add_node("decompose_docs", self._decompose_docs)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("web_search", self._web_search)
        graph.add_node("grade_grounding", self._grade_grounding)
        graph.add_node("hyde_expand", self._hyde_expand)
        graph.add_node("grade_answer", self._grade_answer)

        graph.set_entry_point("retrieve_or_not")

        # After retrieve_or_not decision
        graph.add_conditional_edges(
            "retrieve_or_not",
            self._route_retrieve_decision,
            {"retrieve": "retrieve", "direct_generate": "direct_generate"},
        )

        graph.add_edge("retrieve", "grade_relevance")

        # After relevance grading (reuse CRAG logic)
        graph.add_conditional_edges(
            "grade_relevance",
            self._route_on_relevance,
            {
                "generate": "generate",
                "decompose_docs": "decompose_docs",
                "rewrite_query": "rewrite_query",
            },
        )

        graph.add_edge("decompose_docs", "generate")
        graph.add_edge("rewrite_query", "web_search")
        graph.add_edge("web_search", "generate")

        # After generation, check grounding
        graph.add_edge("generate", "grade_grounding")

        # After grounding check
        graph.add_conditional_edges(
            "grade_grounding",
            self._route_on_grounding,
            {"grade_answer": "grade_answer", "hyde_expand": "hyde_expand"},
        )

        # After HyDE expansion, retrieve again
        graph.add_edge("hyde_expand", "retrieve")

        # After answer quality check
        graph.add_conditional_edges(
            "grade_answer",
            self._route_on_answer_quality,
            {END: END, "rewrite_query": "rewrite_query"},
        )

        # Direct generate goes straight to END
        graph.add_edge("direct_generate", END)

        return graph.compile()

    # ── Utility ─────────────────────────────────────────────────────────────

    def _trace_span(self, trace: Any, name: str) -> Any:
        if trace is not None:
            return trace.span(name=name)
        return None

    def _append_path(self, state: SelfRAGState, step: str) -> list[str]:
        return [*state.get("decision_path", []), step]

    def _update_timings(self, state: SelfRAGState, node: str, elapsed: float) -> dict[str, float]:
        timings = dict(state.get("node_timings", {}))
        # Append attempt number if key already exists
        key = node if node not in timings else f"{node}_{state.get('attempts', 0)}"
        timings[key] = elapsed
        return timings

    # ── Node implementations ────────────────────────────────────────────────

    async def _decide_retrieve(self, state: SelfRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "retrieve_or_not")

        decision = await self._retrieve_or_not.decide(state["query"])

        elapsed = (time.perf_counter() - start) * 1000
        if span:
            span.end(output={"retrieve": decision.retrieve, "latency_ms": elapsed})

        label = "YES" if decision.retrieve else "NO"
        logger.info(
            "RetrieveOrNot: %s (confidence=%.2f, %.1fms)", label, decision.confidence, elapsed
        )

        return {
            "retrieve_needed": decision.retrieve,
            "attempts": 0,
            "decision_path": self._append_path(state, f"retrieve_or_not:{label}"),
            "node_timings": self._update_timings(state, "retrieve_or_not", elapsed),
        }

    def _route_retrieve_decision(self, state: SelfRAGState) -> str:
        return "retrieve" if state.get("retrieve_needed", True) else "direct_generate"

    async def _retrieve(self, state: SelfRAGState) -> dict[str, Any]:
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
            "decision_path": self._append_path(state, "retrieve"),
            "node_timings": self._update_timings(state, "retrieve", elapsed),
        }

    async def _grade_relevance(self, state: SelfRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "grade_relevance")

        documents = state.get("documents", [])
        if not documents:
            elapsed = (time.perf_counter() - start) * 1000
            if span:
                span.end(output={"overall_grade": "IRRELEVANT", "latency_ms": elapsed})
            return {
                "grades": [],
                "overall_grade": RelevanceGrade.IRRELEVANT,
                "decision_path": self._append_path(state, "grade_relevance:IRRELEVANT"),
                "node_timings": self._update_timings(state, "grade_relevance", elapsed),
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
            span.end(output={"overall_grade": overall_grade, "latency_ms": elapsed})

        logger.info("Relevance grading: overall=%s (%.1fms)", overall_grade, elapsed)
        return {
            "grades": grades,
            "overall_grade": overall_grade,
            "decision_path": self._append_path(state, f"grade_relevance:{overall_grade}"),
            "node_timings": self._update_timings(state, "grade_relevance", elapsed),
        }

    def _determine_overall_grade(self, grades: list[GradingResult]) -> str:
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

    def _route_on_relevance(self, state: SelfRAGState) -> str:
        grade = state.get("overall_grade", RelevanceGrade.IRRELEVANT)
        match grade:
            case RelevanceGrade.RELEVANT:
                return "generate"
            case RelevanceGrade.AMBIGUOUS:
                return "decompose_docs"
            case _:
                return "rewrite_query"

    async def _decompose_docs(self, state: SelfRAGState) -> dict[str, Any]:
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
            "Decomposed %d docs → %d excerpts (%.1fms)", len(documents), len(decomposed), elapsed
        )
        return {
            "decomposed_docs": decomposed,
            "decision_path": self._append_path(state, "decompose_docs"),
            "node_timings": self._update_timings(state, "decompose_docs", elapsed),
        }

    async def _rewrite_query(self, state: SelfRAGState) -> dict[str, Any]:
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
            "decision_path": self._append_path(state, "rewrite_query"),
            "node_timings": self._update_timings(state, "rewrite_query", elapsed),
        }

    async def _web_search(self, state: SelfRAGState) -> dict[str, Any]:
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
            "decision_path": self._append_path(state, "web_search"),
            "node_timings": self._update_timings(state, "web_search", elapsed),
        }

    async def _generate(self, state: SelfRAGState) -> dict[str, Any]:
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
            span.end(output={"answer_length": len(answer), "latency_ms": elapsed})

        logger.info("Generated answer (%d chars, %.1fms)", len(answer), elapsed)
        return {
            "answer": answer,
            "context": context,
            "decision_path": self._append_path(state, "generate"),
            "node_timings": self._update_timings(state, "generate", elapsed),
        }

    async def _direct_generate(self, state: SelfRAGState) -> dict[str, Any]:
        """Generate answer without retrieval for simple queries."""
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "direct_generate")

        system_prompt = (
            "You are a helpful assistant. Answer the user's question directly and concisely. "
            "This is a straightforward question that does not require external documents."
        )

        message = await self._llm_client.messages.create(
            model=self._generation_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": state["query"]}],
        )

        answer = message.content[0].text
        elapsed = (time.perf_counter() - start) * 1000

        if span:
            span.end(output={"answer_length": len(answer), "latency_ms": elapsed})

        logger.info("Direct generated answer (%d chars, %.1fms)", len(answer), elapsed)
        return {
            "answer": answer,
            "context": "(no retrieval needed)",
            "decision_path": self._append_path(state, "direct_generate"),
            "node_timings": self._update_timings(state, "direct_generate", elapsed),
        }

    async def _grade_grounding(self, state: SelfRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "grade_grounding")

        documents = state.get("documents", [])
        doc_contents = [doc["content"] for doc in documents]
        answer = state.get("answer", "")

        result = await self._hallucination_grader.grade(doc_contents, answer)

        elapsed = (time.perf_counter() - start) * 1000
        if span:
            span.end(output={"grade": result.grade, "latency_ms": elapsed})

        logger.info(
            "Grounding check: %s (confidence=%.2f, %.1fms)",
            result.grade,
            result.confidence,
            elapsed,
        )
        return {
            "grounding_grade": result.grade,
            "grounding_confidence": result.confidence,
            "decision_path": self._append_path(state, f"grade_grounding:{result.grade}"),
            "node_timings": self._update_timings(state, "grade_grounding", elapsed),
        }

    def _route_on_grounding(self, state: SelfRAGState) -> str:
        grade = state.get("grounding_grade", GroundingGrade.NOT_GROUNDED)
        attempts = state.get("attempts", 0)

        if grade == GroundingGrade.GROUNDED:
            return "grade_answer"

        # Not grounded — retry with HyDE if under max retries
        if attempts < MAX_RETRIES:
            return "hyde_expand"

        # Exhausted retries — accept current answer
        logger.warning("Max retries (%d) reached for grounding — accepting answer", MAX_RETRIES)
        return "grade_answer"

    async def _hyde_expand(self, state: SelfRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "hyde_expand")

        hyde_embedding = await self._hyde.expand(state["query"])

        # Override the embedding service's next embed call by storing the HyDE embedding
        # We retrieve using this embedding in the next retrieve step
        results = await self._vector_store.search(
            query_embedding=hyde_embedding,
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

        attempts = state.get("attempts", 0) + 1
        logger.info(
            "HyDE expand + retrieve: %d docs (attempt %d, %.1fms)",
            len(documents),
            attempts,
            elapsed,
        )
        return {
            "documents": documents,
            "attempts": attempts,
            "decision_path": self._append_path(state, f"hyde_expand(attempt={attempts})"),
            "node_timings": self._update_timings(state, "hyde_expand", elapsed),
        }

    async def _grade_answer(self, state: SelfRAGState) -> dict[str, Any]:
        start = time.perf_counter()
        trace = state.get("_trace")
        span = self._trace_span(trace, "grade_answer")

        result = await self._answer_grader.grade(state["query"], state.get("answer", ""))

        elapsed = (time.perf_counter() - start) * 1000
        if span:
            span.end(output={"grade": result.grade, "latency_ms": elapsed})

        logger.info(
            "Answer quality: %s (confidence=%.2f, %.1fms)",
            result.grade,
            result.confidence,
            elapsed,
        )
        return {
            "answer_quality": result.grade,
            "answer_quality_confidence": result.confidence,
            "decision_path": self._append_path(state, f"grade_answer:{result.grade}"),
            "node_timings": self._update_timings(state, "grade_answer", elapsed),
        }

    def _route_on_answer_quality(self, state: SelfRAGState) -> str:
        quality = state.get("answer_quality", AnswerQuality.DOES_NOT_ADDRESS)
        attempts = state.get("attempts", 0)

        if quality == AnswerQuality.ADDRESSES_QUESTION:
            return END

        # Answer doesn't address the question — rewrite and retry if under max
        if attempts < MAX_RETRIES:
            return "rewrite_query"

        logger.warning(
            "Max retries (%d) reached for answer quality — accepting answer", MAX_RETRIES
        )
        return END

    def _build_context(self, state: SelfRAGState) -> str:
        overall_grade = state.get("overall_grade", "")
        parts: list[str] = []

        if overall_grade == RelevanceGrade.RELEVANT:
            for i, doc in enumerate(state.get("documents", []), 1):
                parts.append(f"[Source {i}] (score={doc['score']:.3f})\n{doc['content']}")

        elif overall_grade == RelevanceGrade.AMBIGUOUS:
            for i, excerpt in enumerate(state.get("decomposed_docs", []), 1):
                parts.append(f"[Source {i}] (decomposed excerpt)\n{excerpt}")

        else:
            # IRRELEVANT or web fallback
            web_results = state.get("web_results", [])
            if web_results:
                for i, result in enumerate(web_results, 1):
                    parts.append(
                        f"[Source {i}] ({result.get('url', 'web')})\n{result.get('content', '')}"
                    )
            else:
                for i, doc in enumerate(state.get("documents", []), 1):
                    parts.append(f"[Source {i}] (score={doc['score']:.3f})\n{doc['content']}")

        return "\n\n---\n\n".join(parts) if parts else "(no context available)"

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> SelfRAGState:
        """Execute the full Self-RAG graph.

        Returns the final state with answer, decision path, grades, and timings.
        """
        trace = None
        if self._langfuse:
            trace = self._langfuse.trace(name="self_rag_pipeline", input={"query": query})

        initial_state: SelfRAGState = {
            "query": query,
            "top_k": top_k,
            "filters": filters or {},
            "retrieve_needed": True,
            "documents": [],
            "grades": [],
            "overall_grade": "",
            "decomposed_docs": [],
            "rewritten_query": "",
            "web_results": [],
            "context": "",
            "answer": "",
            "grounding_grade": "",
            "grounding_confidence": 0.0,
            "answer_quality": "",
            "answer_quality_confidence": 0.0,
            "attempts": 0,
            "decision_path": [],
            "node_timings": {},
        }

        result = await self._graph.ainvoke(initial_state)

        if trace:
            trace.update(
                output={
                    "answer": result.get("answer", "")[:200],
                    "decision_path": result.get("decision_path", []),
                    "grounding_grade": result.get("grounding_grade", ""),
                    "answer_quality": result.get("answer_quality", ""),
                    "attempts": result.get("attempts", 0),
                    "total_latency_ms": sum(result.get("node_timings", {}).values()),
                }
            )

        return result
