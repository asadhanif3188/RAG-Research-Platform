"""BenchmarkRunner — measures latency, throughput, and recall@k for vector retrieval.

Compares full precision, binary quantized, and scalar quantized Qdrant collections.

Metrics:
  - Latency: p50, p95, p99 (milliseconds per query)
  - Throughput: queries per second (QPS)
  - Recall@k: fraction of true top-k results present in quantised top-k

Usage:
    runner = BenchmarkRunner(qdrant_host="localhost")
    await runner.connect()
    results = await runner.run(n_queries=1000, top_k=10)
    runner.save(results, "benchmark_results.json")
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastest_rag.quantization import CollectionVariant

logger = logging.getLogger(__name__)


@dataclass
class VariantMetrics:
    variant: str
    n_queries: int
    top_k: int
    latencies_ms: list[float] = field(default_factory=list)
    recall_at_k: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    @property
    def mean_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def qps(self) -> float:
        total_s = sum(self.latencies_ms) / 1000
        return self.n_queries / total_s if total_s > 0 else 0.0

    @property
    def mean_recall(self) -> float:
        return statistics.mean(self.recall_at_k) if self.recall_at_k else 0.0

    def _percentile(self, p: int) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = math.ceil(p / 100 * len(sorted_latencies)) - 1
        return sorted_latencies[max(0, idx)]

    def summary(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "n_queries": self.n_queries,
            "top_k": self.top_k,
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "mean_ms": round(self.mean_latency_ms, 3),
            "qps": round(self.qps, 1),
            "recall_at_k": round(self.mean_recall, 4),
        }


@dataclass
class BenchmarkReport:
    variants: list[VariantMetrics]
    vector_size: int
    run_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def summaries(self) -> list[dict[str, Any]]:
        return [v.summary() for v in self.variants]

    def relative_speedup(
        self,
        baseline: CollectionVariant = CollectionVariant.FULL_PRECISION,
    ) -> dict[str, float]:
        """Compute p99 latency speedup of each variant vs the baseline."""
        baseline_metrics = next((v for v in self.variants if v.variant == baseline.value), None)
        if baseline_metrics is None or baseline_metrics.p99 == 0:
            return {}
        return {
            v.variant: round(baseline_metrics.p99 / v.p99, 2) for v in self.variants if v.p99 > 0
        }

    def recall_drop(
        self, baseline: CollectionVariant = CollectionVariant.FULL_PRECISION
    ) -> dict[str, float]:
        """Recall drop relative to baseline (positive = worse recall)."""
        baseline_metrics = next((v for v in self.variants if v.variant == baseline.value), None)
        if baseline_metrics is None:
            return {}
        base_recall = baseline_metrics.mean_recall
        return {v.variant: round(base_recall - v.mean_recall, 4) for v in self.variants}


class BenchmarkRunner:
    """Run latency and recall benchmarks across Qdrant collection variants.

    Args:
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant port.
        qdrant_api_key: Optional API key.
        concurrency: Number of concurrent queries during throughput test.
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: str = "",
        concurrency: int = 10,
    ) -> None:
        self._host = qdrant_host
        self._port = qdrant_port
        self._api_key = qdrant_api_key or None
        self._concurrency = concurrency
        self._client: Any = None

    async def connect(self) -> None:
        from qdrant_client import AsyncQdrantClient

        self._client = AsyncQdrantClient(host=self._host, port=self._port, api_key=self._api_key)
        logger.info("BenchmarkRunner connected to Qdrant")

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def run(
        self,
        n_queries: int = 1000,
        top_k: int = 10,
        vector_size: int = 3072,
        variants: list[CollectionVariant] | None = None,
        seed: int = 42,
    ) -> BenchmarkReport:
        """Run benchmark for all specified variants.

        Uses full-precision results as ground truth for recall calculation.
        """
        if variants is None:
            variants = list(CollectionVariant)

        random.seed(seed)
        query_vectors = self._generate_query_vectors(n_queries, vector_size, seed)

        # Compute ground truth using full precision first
        logger.info("Computing ground truth (full precision, %d queries)...", n_queries)
        ground_truth = await self._batch_search(
            collection=CollectionVariant.FULL_PRECISION.value,
            vectors=query_vectors,
            top_k=top_k,
            record_latency=False,
        )
        true_ids: list[set[str]] = [{str(r["id"]) for r in hits} for hits in ground_truth]

        # Benchmark each variant
        variant_metrics: list[VariantMetrics] = []

        for variant in variants:
            logger.info("Benchmarking '%s' (%d queries)...", variant.value, n_queries)
            metrics = await self._benchmark_variant(
                variant=variant,
                query_vectors=query_vectors,
                top_k=top_k,
                true_ids=true_ids,
            )
            variant_metrics.append(metrics)
            logger.info("  %s", metrics.summary())

        import datetime

        return BenchmarkReport(
            variants=variant_metrics,
            vector_size=vector_size,
            run_at=datetime.datetime.utcnow().isoformat() + "Z",
            metadata={"n_queries": n_queries, "top_k": top_k, "seed": seed},
        )

    async def run_sequential_latency(
        self,
        collection: str,
        query_vectors: list[list[float]],
        top_k: int = 10,
    ) -> list[float]:
        """Run queries sequentially and return per-query latency in ms."""
        latencies: list[float] = []
        for vec in query_vectors:
            t0 = time.perf_counter()
            await self._client.search(
                collection_name=collection,
                query_vector=vec,
                limit=top_k,
                with_payload=False,
            )
            latencies.append((time.perf_counter() - t0) * 1000)
        return latencies

    @staticmethod
    def save(report: BenchmarkReport, path: str | Path) -> None:
        """Serialize report to JSON."""
        data = {
            "run_at": report.run_at,
            "vector_size": report.vector_size,
            "metadata": report.metadata,
            "summaries": report.summaries(),
            "speedup_vs_full_precision": report.relative_speedup(),
            "recall_drop_vs_full_precision": report.recall_drop(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Benchmark report saved to %s", path)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _benchmark_variant(
        self,
        variant: CollectionVariant,
        query_vectors: list[list[float]],
        top_k: int,
        true_ids: list[set[str]],
    ) -> VariantMetrics:
        metrics = VariantMetrics(
            variant=variant.value,
            n_queries=len(query_vectors),
            top_k=top_k,
        )

        semaphore = asyncio.Semaphore(self._concurrency)

        async def query_one(i: int, vec: list[float]) -> None:
            async with semaphore:
                t0 = time.perf_counter()
                hits = await self._client.search(
                    collection_name=variant.value,
                    query_vector=vec,
                    limit=top_k,
                    with_payload=False,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                metrics.latencies_ms.append(latency_ms)

                # Recall@k: fraction of true top-k IDs in this result
                retrieved_ids = {str(h.id) for h in hits}
                recall = len(retrieved_ids & true_ids[i]) / len(true_ids[i]) if true_ids[i] else 1.0
                metrics.recall_at_k.append(recall)

        await asyncio.gather(*[query_one(i, v) for i, v in enumerate(query_vectors)])
        return metrics

    async def _batch_search(
        self,
        collection: str,
        vectors: list[list[float]],
        top_k: int,
        record_latency: bool = True,
    ) -> list[list[dict[str, Any]]]:
        semaphore = asyncio.Semaphore(self._concurrency)
        results: list[list[dict[str, Any]]] = [[] for _ in vectors]

        async def search_one(i: int, vec: list[float]) -> None:
            async with semaphore:
                hits = await self._client.search(
                    collection_name=collection,
                    query_vector=vec,
                    limit=top_k,
                    with_payload=False,
                )
                results[i] = [{"id": str(h.id), "score": h.score} for h in hits]

        await asyncio.gather(*[search_one(i, v) for i, v in enumerate(vectors)])
        return results

    @staticmethod
    def _generate_query_vectors(n: int, vector_size: int, seed: int) -> list[list[float]]:
        random.seed(seed)
        vectors: list[list[float]] = []
        for _ in range(n):
            vec = [random.gauss(0, 1) for _ in range(vector_size)]
            norm = math.sqrt(sum(x * x for x in vec))
            vectors.append([x / (norm + 1e-9) for x in vec])
        return vectors
