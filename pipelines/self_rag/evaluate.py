"""RAGAS evaluation script — compare Naive RAG vs. CRAG vs. Self-RAG.

Usage:
    python -m self_rag.evaluate          # from repo root
    python pipelines/self_rag/evaluate.py

Requires:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
    Running PostgreSQL with pgvector + seeded data
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "tests" / "test_dataset.json"


async def run_evaluation() -> None:
    from corrective_rag.pipeline import CRAGPipeline
    from fastest_rag.pipeline import NaiveRAGPipeline
    from self_rag.pipeline import SelfRAGPipeline
    from shared.config import get_settings
    from shared.embeddings.service import EmbeddingService
    from shared.eval.ragas_runner import RAGASRunner
    from shared.models.query import QueryRequest
    from shared.storage.vector_store import PgVectorClient

    settings = get_settings()

    # ── Shared services ──────────────────────────────────────────────────
    embedding_service = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    embedding_service.connect()

    vector_store = PgVectorClient(database_url=settings.database_url)
    await vector_store.connect()

    # ── Pipelines ────────────────────────────────────────────────────────
    naive = NaiveRAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        anthropic_api_key=settings.anthropic_api_key,
    )
    naive.connect()

    crag = CRAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        anthropic_api_key=settings.anthropic_api_key,
        tavily_api_key=settings.tavily_api_key,
    )
    crag.connect()

    self_rag = SelfRAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        anthropic_api_key=settings.anthropic_api_key,
        tavily_api_key=settings.tavily_api_key,
    )
    self_rag.connect()

    # ── Load dataset ─────────────────────────────────────────────────────
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    ragas = RAGASRunner(
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
    )

    # ── Evaluate all pipelines ───────────────────────────────────────────
    naive_samples = []
    crag_samples = []
    self_rag_samples = []

    for item in dataset[:10]:  # first 10 for cost control
        query = item["query"]
        ground_truth = item["ground_truth"]
        logger.info("Evaluating: %s", query[:60])

        for pipeline, samples, _name in [
            (naive, naive_samples, "naive"),
            (crag, crag_samples, "crag"),
            (self_rag, self_rag_samples, "self_rag"),
        ]:
            req = QueryRequest(query=query, use_cache=False)
            resp = await pipeline.run(req)
            samples.append(
                {
                    "query": query,
                    "answer": resp.answer,
                    "contexts": [s.content for s in resp.sources],
                    "ground_truth": ground_truth,
                }
            )

    naive_results = await ragas.evaluate_batch(naive_samples, pipeline="fastest_rag")
    crag_results = await ragas.evaluate_batch(crag_samples, pipeline="corrective_rag")
    self_rag_results = await ragas.evaluate_batch(self_rag_samples, pipeline="self_rag")

    # ── Print comparison table ───────────────────────────────────────────
    def avg(results: list, attr: str) -> float:
        vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 76)
    print("RAGAS EVALUATION: Naive RAG vs. CRAG vs. Self-RAG")
    print("=" * 76)
    print(f"{'Metric':<25} {'Naive RAG':>12} {'CRAG':>12} {'Self-RAG':>12} {'Delta':>12}")
    print("-" * 76)

    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        n = avg(naive_results, metric)
        c = avg(crag_results, metric)
        s = avg(self_rag_results, metric)
        delta = s - n
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<25} {n:>11.3f} {c:>11.3f} {s:>11.3f} {sign}{delta:>11.3f}")

    naive_agg = sum(r.aggregate_score() or 0 for r in naive_results) / max(len(naive_results), 1)
    crag_agg = sum(r.aggregate_score() or 0 for r in crag_results) / max(len(crag_results), 1)
    self_rag_agg = sum(r.aggregate_score() or 0 for r in self_rag_results) / max(
        len(self_rag_results), 1
    )
    print("-" * 76)
    print(
        f"{'Aggregate'::<25} {naive_agg:>11.3f} {crag_agg:>11.3f} {self_rag_agg:>11.3f} "
        f"{'+' if self_rag_agg >= naive_agg else ''}{self_rag_agg - naive_agg:>11.3f}"
    )
    print("=" * 76)

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "naive_rag": [r.model_dump(mode="json") for r in naive_results],
        "corrective_rag": [r.model_dump(mode="json") for r in crag_results],
        "self_rag": [r.model_dump(mode="json") for r in self_rag_results],
    }
    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
