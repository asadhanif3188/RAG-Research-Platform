"""RAGAS evaluation script — compare Naive RAG vs. CRAG on hallucination-prone queries.

Usage:
    python -m corrective_rag.evaluate          # from repo root
    python pipelines/corrective_rag/evaluate.py

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
    from shared.config import get_settings
    from shared.embeddings.service import EmbeddingService
    from shared.eval.ragas_runner import RAGASRunner
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

    # ── Load dataset ─────────────────────────────────────────────────────
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    ragas = RAGASRunner(
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
    )

    # ── Evaluate both pipelines ──────────────────────────────────────────
    from shared.models.query import QueryRequest

    naive_samples = []
    crag_samples = []

    for item in dataset[:10]:  # start with first 10 for cost control
        query = item["query"]
        ground_truth = item["ground_truth"]
        logger.info("Evaluating: %s", query[:60])

        # Naive RAG
        naive_req = QueryRequest(query=query, use_cache=False)
        naive_resp = await naive.run(naive_req)
        naive_samples.append(
            {
                "query": query,
                "answer": naive_resp.answer,
                "contexts": [s.content for s in naive_resp.sources],
                "ground_truth": ground_truth,
            }
        )

        # CRAG
        crag_req = QueryRequest(query=query, use_cache=False)
        crag_resp = await crag.run(crag_req)
        crag_samples.append(
            {
                "query": query,
                "answer": crag_resp.answer,
                "contexts": [s.content for s in crag_resp.sources],
                "ground_truth": ground_truth,
            }
        )

    naive_results = await ragas.evaluate_batch(naive_samples, pipeline="fastest_rag")
    crag_results = await ragas.evaluate_batch(crag_samples, pipeline="corrective_rag")

    # ── Print comparison table ───────────────────────────────────────────
    def avg(results: list, attr: str) -> float:
        vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION: Naive RAG vs. Corrective RAG (CRAG)")
    print("=" * 60)
    print(f"{'Metric':<25} {'Naive RAG':>12} {'CRAG':>12} {'Delta':>12}")
    print("-" * 60)

    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        n = avg(naive_results, metric)
        c = avg(crag_results, metric)
        delta = c - n
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<25} {n:>11.3f} {c:>11.3f} {sign}{delta:>11.3f}")

    naive_agg = sum(r.aggregate_score() or 0 for r in naive_results) / max(len(naive_results), 1)
    crag_agg = sum(r.aggregate_score() or 0 for r in crag_results) / max(len(crag_results), 1)
    print("-" * 60)
    print(
        f"{'Aggregate'::<25} {naive_agg:>11.3f} {crag_agg:>11.3f} {'+' if crag_agg >= naive_agg else ''}{crag_agg - naive_agg:>11.3f}"
    )
    print("=" * 60)

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "naive_rag": [r.model_dump(mode="json") for r in naive_results],
        "corrective_rag": [r.model_dump(mode="json") for r in crag_results],
    }
    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
