"""Unit tests for BenchmarkRunner — synthetic latency measurements."""

from __future__ import annotations

import math
import statistics
import tempfile
from pathlib import Path

import pytest

from fastest_rag.benchmark import BenchmarkReport, BenchmarkRunner, VariantMetrics
from fastest_rag.quantization import CollectionVariant


def make_metrics(variant: str, latencies: list[float], recalls: list[float]) -> VariantMetrics:
    m = VariantMetrics(variant=variant, n_queries=len(latencies), top_k=10)
    m.latencies_ms = latencies
    m.recall_at_k = recalls
    return m


class TestVariantMetrics:
    def test_p50(self):
        m = make_metrics("test", list(range(1, 101)), [1.0] * 100)
        assert m.p50 == statistics.median(range(1, 101))

    def test_p99_captures_tail_latency(self):
        # With 99 values of 10.0 and 1 outlier at 100.0, p99 picks the 99th percentile
        # ceil(99/100 * 100) - 1 = index 98 → 10.0 (99% of values are 10.0)
        m = make_metrics("test", [10.0] * 99 + [100.0], [1.0] * 100)
        assert m.p99 == 10.0
        # With fewer values below the outlier, p99 captures it
        m2 = make_metrics("test", [10.0] * 98 + [100.0, 100.0], [1.0] * 100)
        assert m2.p99 == 100.0

    def test_qps_is_positive(self):
        m = make_metrics("test", [10.0] * 100, [1.0] * 100)
        assert m.qps > 0

    def test_mean_recall(self):
        m = make_metrics("test", [5.0] * 4, [0.8, 0.9, 1.0, 0.7])
        assert m.mean_recall == pytest.approx(0.85)

    def test_empty_latencies(self):
        m = VariantMetrics(variant="empty", n_queries=0, top_k=5)
        assert m.p50 == 0.0
        assert m.p99 == 0.0
        assert m.qps == 0.0

    def test_summary_has_required_keys(self):
        m = make_metrics("rag_full_precision", [10.0, 20.0, 30.0], [0.9, 0.85, 0.95])
        s = m.summary()
        assert "p50_ms" in s
        assert "p95_ms" in s
        assert "p99_ms" in s
        assert "qps" in s
        assert "recall_at_k" in s


class TestBenchmarkReport:
    def _make_report(self) -> BenchmarkReport:
        fp = make_metrics(CollectionVariant.FULL_PRECISION.value, [50.0] * 100, [1.0] * 100)
        bq = make_metrics(CollectionVariant.BINARY_QUANTIZED.value, [20.0] * 100, [0.95] * 100)
        sq = make_metrics(CollectionVariant.SCALAR_QUANTIZED.value, [30.0] * 100, [0.98] * 100)
        return BenchmarkReport(variants=[fp, bq, sq], vector_size=3072, run_at="2026-01-01T00:00:00Z")

    def test_summaries_has_all_variants(self):
        report = self._make_report()
        assert len(report.summaries()) == 3

    def test_relative_speedup_bq_is_faster(self):
        report = self._make_report()
        speedup = report.relative_speedup()
        bq_key = CollectionVariant.BINARY_QUANTIZED.value
        assert speedup[bq_key] > 1.0  # BQ is 2.5x faster in this test

    def test_recall_drop_bq_is_positive(self):
        report = self._make_report()
        drop = report.recall_drop()
        bq_key = CollectionVariant.BINARY_QUANTIZED.value
        assert drop[bq_key] == pytest.approx(0.05)

    def test_save_and_load(self):
        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        BenchmarkRunner.save(report, path)
        loaded = BenchmarkRunner.load(path)

        assert "summaries" in loaded
        assert len(loaded["summaries"]) == 3
        assert loaded["vector_size"] == 3072
        path.unlink()


class TestBenchmarkRunnerUnit:
    def test_generate_query_vectors_count(self):
        vecs = BenchmarkRunner._generate_query_vectors(n=50, vector_size=128, seed=1)
        assert len(vecs) == 50
        assert len(vecs[0]) == 128

    def test_generate_query_vectors_are_unit_normalised(self):
        vecs = BenchmarkRunner._generate_query_vectors(n=5, vector_size=64, seed=0)
        for vec in vecs:
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-5

    def test_generate_query_vectors_reproducible(self):
        v1 = BenchmarkRunner._generate_query_vectors(10, 32, seed=42)
        v2 = BenchmarkRunner._generate_query_vectors(10, 32, seed=42)
        assert v1 == v2

    def test_generate_query_vectors_different_seeds(self):
        v1 = BenchmarkRunner._generate_query_vectors(10, 32, seed=1)
        v2 = BenchmarkRunner._generate_query_vectors(10, 32, seed=2)
        assert v1 != v2
