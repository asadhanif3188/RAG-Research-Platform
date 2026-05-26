"""Streamlit benchmark dashboard for the Fastest RAG Stack.

Launch with:
    streamlit run pipelines/fastest_rag/src/fastest_rag/dashboard.py

Features:
  - Latency comparison chart (p50/p95/p99) across full precision, BQ, SQ
  - Cache hit rate over time (updates live if connected to live Redis)
  - Throughput (QPS) bar chart
  - Recall@k metrics table
  - Raw benchmark JSON viewer
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fastest RAG — Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
)

RESULTS_DIR = Path(os.getenv("BENCHMARK_RESULTS_DIR", "benchmark_results"))
LIVE_STATS_KEY = "rag:benchmark:cache_stats"

VARIANT_LABELS = {
    "rag_full_precision": "Full Precision",
    "rag_binary_quantized": "Binary Quantized (BQ)",
    "rag_scalar_quantized": "Scalar Quantized (SQ)",
}

VARIANT_COLORS = {
    "Full Precision": "#636EFA",
    "Binary Quantized (BQ)": "#EF553B",
    "Scalar Quantized (SQ)": "#00CC96",
}


def load_report(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def available_reports() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.json"), reverse=True)


def build_latency_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        label = VARIANT_LABELS.get(s["variant"], s["variant"])
        for pct, key in [("p50", "p50_ms"), ("p95", "p95_ms"), ("p99", "p99_ms")]:
            rows.append({"Variant": label, "Percentile": pct, "Latency (ms)": s[key]})
    return pd.DataFrame(rows)


def build_qps_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Variant": VARIANT_LABELS.get(s["variant"], s["variant"]),
            "QPS": s["qps"],
        }
        for s in summaries
    ])


def build_recall_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Variant": VARIANT_LABELS.get(s["variant"], s["variant"]),
            "Recall@k": s["recall_at_k"],
            "p50 (ms)": s["p50_ms"],
            "p95 (ms)": s["p95_ms"],
            "p99 (ms)": s["p99_ms"],
            "QPS": s["qps"],
        }
        for s in summaries
    ])


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Fastest RAG")
st.sidebar.markdown("**Benchmark Dashboard**")

reports = available_reports()
if reports:
    selected_path = st.sidebar.selectbox(
        "Select benchmark run",
        options=reports,
        format_func=lambda p: p.stem,
    )
    report = load_report(selected_path)
else:
    report = None
    st.sidebar.info("No benchmark results found. Run `BenchmarkRunner` first.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Live cache stats**")
redis_url = st.sidebar.text_input("Redis URL", value="redis://localhost:6379/0")
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)

if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
st.title("⚡ Fastest RAG — Benchmark Dashboard")

if report is None:
    st.warning(
        "No benchmark report loaded. "
        "Run the benchmark with `BenchmarkRunner` and save results to `benchmark_results/`."
    )
    st.info(
        "**Quick start:**\n"
        "```python\n"
        "from fastest_rag.benchmark import BenchmarkRunner\n"
        "runner = BenchmarkRunner()\n"
        "await runner.connect()\n"
        "report = await runner.run(n_queries=1000, top_k=10)\n"
        "BenchmarkRunner.save(report, 'benchmark_results/run_01.json')\n"
        "```"
    )
    st.stop()

summaries = report.get("summaries", [])
run_at = report.get("run_at", "unknown")
metadata = report.get("metadata", {})
speedup = report.get("speedup_vs_full_precision", {})
recall_drop = report.get("recall_drop_vs_full_precision", {})

st.caption(f"Run at: {run_at} · {metadata.get('n_queries', '?')} queries · top_k={metadata.get('top_k', '?')}")

# ── KPI metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

bq_summary = next((s for s in summaries if "binary" in s["variant"]), None)
fp_summary = next((s for s in summaries if "full" in s["variant"]), None)

if bq_summary and fp_summary:
    speedup_val = speedup.get(bq_summary["variant"], 1.0)
    recall_drop_val = recall_drop.get(bq_summary["variant"], 0.0)
    col1.metric("BQ p99 Latency", f"{bq_summary['p99_ms']:.1f} ms")
    col2.metric("BQ Speedup vs Full", f"{speedup_val:.1f}×", delta=f"{speedup_val - 1:.1%} faster")
    col3.metric("BQ Recall Drop", f"{recall_drop_val:.2%}", delta_color="inverse")
    col4.metric("Full Precision p99", f"{fp_summary['p99_ms']:.1f} ms")

st.markdown("---")

# ── Latency comparison ────────────────────────────────────────────────────────
st.subheader("Latency Comparison (ms)")
latency_df = build_latency_df(summaries)
fig_latency = px.bar(
    latency_df,
    x="Percentile",
    y="Latency (ms)",
    color="Variant",
    barmode="group",
    color_discrete_map=VARIANT_COLORS,
    text_auto=".1f",
)
fig_latency.update_layout(height=400, legend_title_text="Collection variant")
st.plotly_chart(fig_latency, use_container_width=True)

col_left, col_right = st.columns(2)

# ── QPS chart ─────────────────────────────────────────────────────────────────
with col_left:
    st.subheader("Throughput (QPS)")
    qps_df = build_qps_df(summaries)
    fig_qps = px.bar(
        qps_df,
        x="Variant",
        y="QPS",
        color="Variant",
        color_discrete_map=VARIANT_COLORS,
        text_auto=".0f",
    )
    fig_qps.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_qps, use_container_width=True)

# ── Speedup radar ─────────────────────────────────────────────────────────────
with col_right:
    st.subheader("Speedup vs Full Precision (p99)")
    if speedup:
        labels = [VARIANT_LABELS.get(k, k) for k in speedup]
        values = list(speedup.values())
        fig_speedup = go.Figure(go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.2f}×" for v in values],
            textposition="outside",
            marker_color=[VARIANT_COLORS.get(l, "#888") for l in labels],
        ))
        fig_speedup.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="baseline")
        fig_speedup.update_layout(height=350, yaxis_title="Speedup factor (×)")
        st.plotly_chart(fig_speedup, use_container_width=True)

# ── Recall table ──────────────────────────────────────────────────────────────
st.subheader("Recall@k & Full Metrics")
recall_df = build_recall_df(summaries)
st.dataframe(recall_df.style.format({
    "Recall@k": "{:.4f}",
    "p50 (ms)": "{:.2f}",
    "p95 (ms)": "{:.2f}",
    "p99 (ms)": "{:.2f}",
    "QPS": "{:.0f}",
}).background_gradient(subset=["Recall@k"], cmap="RdYlGn"), use_container_width=True)

# ── Cache stats ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Redis Semantic Cache Stats")

try:
    import redis

    r = redis.from_url(redis_url)
    r.ping()

    cache_keys = r.keys("rag:semantic_cache:*")
    hit_rate_raw = r.get("rag:metrics:hit_rate")
    total_requests_raw = r.get("rag:metrics:total_requests")

    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Cached Entries", len(cache_keys))
    col_c2.metric("Hit Rate", f"{float(hit_rate_raw or 0):.1%}")
    col_c3.metric("Total Requests", int(total_requests_raw or 0))

except Exception as e:
    st.info(f"Redis not reachable at `{redis_url}`. Cache stats unavailable. ({e})")

# ── Raw JSON viewer ───────────────────────────────────────────────────────────
with st.expander("Raw benchmark JSON"):
    st.json(report)
