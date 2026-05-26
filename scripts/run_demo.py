#!/usr/bin/env python3
"""
RAG Research Platform — Demo Runner
====================================
Runs end-to-end demonstrations of Fastest RAG and Multimodal RAG pipelines.

Two modes:
  --mock   (default) All external services are mocked. No Docker / API keys needed.
  --live             Connects to real services. Requires Docker stack + .env file.

Usage
-----
  # Mock mode (instant, no setup required)
  python scripts/run_demo.py

  # Live mode (requires Docker + .env)
  python scripts/run_demo.py --live

  # Run only one pipeline
  python scripts/run_demo.py --pipeline fastest_rag
  python scripts/run_demo.py --pipeline multimodal_rag

  # Custom query
  python scripts/run_demo.py --query "What is retrieval-augmented generation?"

  # Verbose output (show full sources)
  python scripts/run_demo.py --verbose
"""

from __future__ import annotations

# Ensure UTF-8 output on Windows (cp1252 default terminal cannot encode box chars)
import io as _io
import sys as _sys
if hasattr(_sys.stdout, "buffer"):
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ── Add repo root to path so workspace packages resolve ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared" / "src"))
sys.path.insert(0, str(ROOT / "pipelines" / "fastest_rag" / "src"))
sys.path.insert(0, str(ROOT / "pipelines" / "multimodal_rag" / "src"))
sys.path.insert(0, str(ROOT / "api" / "src"))

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
MAGENTA = "\033[35m"
RED    = "\033[31m"

def h1(text: str) -> None:
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")

def h2(text: str) -> None:
    print(f"\n{BOLD}{CYAN}  >> {text}{RESET}")

def ok(text: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {text}")

def info(text: str) -> None:
    print(f"  {DIM}{text}{RESET}")

def warn(text: str) -> None:
    print(f"  {YELLOW}[!] {text}{RESET}")

def err(text: str) -> None:
    print(f"  {RED}[ERR]{RESET} {text}")

def kv(key: str, value: Any) -> None:
    print(f"  {DIM}{key}:{RESET} {value}")

# ── Mock factories ─────────────────────────────────────────────────────────────

def make_mock_embedding_service() -> Any:
    """Return a mock EmbeddingService that produces unit vectors."""
    dims = 3072
    val = 1.0 / math.sqrt(dims)
    fake_embedding = [val] * dims

    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=fake_embedding)
    svc.embed_batch = AsyncMock(return_value=[fake_embedding])
    svc.total_tokens_used = 12
    svc.total_cost_usd = 0.0000016
    svc.connect = MagicMock()
    return svc


def make_mock_vector_store(multimodal: bool = False) -> Any:
    """Return a mock VectorStoreClient with realistic sample chunks."""
    from shared.models.retrieval import RetrievalResult

    text_chunks = [
        RetrievalResult(
            chunk_id="c-text-001",
            document_id="attention-is-all-you-need",
            content=(
                "Retrieval-Augmented Generation (RAG) combines a retrieval step with a "
                "language model to answer questions grounded in real documents. "
                "The retrieval step fetches the most relevant passages using vector similarity, "
                "while the generation step synthesises a fluent answer from those passages."
            ),
            chunk_type="text",
            score=0.94,
            metadata={"page_number": 2},
        ),
        RetrievalResult(
            chunk_id="c-text-002",
            document_id="attention-is-all-you-need",
            content=(
                "Dense retrieval encodes both queries and documents as vectors in the same "
                "embedding space. Similarity is measured by cosine distance. "
                "HNSW indexing enables sub-millisecond approximate nearest-neighbour search "
                "even on corpora with millions of vectors."
            ),
            chunk_type="text",
            score=0.88,
            metadata={"page_number": 4},
        ),
    ]

    if multimodal:
        image_chunk = RetrievalResult(
            chunk_id="c-img-001",
            document_id="attention-is-all-you-need",
            content=(
                "Figure 3 shows a bar chart comparing recall@10 across three retrieval "
                "strategies: BM25 (0.62), Dense (0.81), and Multimodal RAG (0.93). "
                "The multimodal approach achieves the highest recall by incorporating "
                "both textual and visual chunk types."
            ),
            chunk_type="image_description",
            score=0.85,
            metadata={"page_number": 7},
        )
        table_chunk = RetrievalResult(
            chunk_id="c-tbl-001",
            document_id="attention-is-all-you-need",
            content=(
                "Benchmark results table (Table 2).\n"
                "| Model          | Faithfulness | Relevancy | Context Recall |\n"
                "|----------------|--------------|-----------|----------------|\n"
                "| Text-only RAG  | 0.81         | 0.76      | 0.74           |\n"
                "| Multimodal RAG | 0.89         | 0.83      | 0.91           |"
            ),
            chunk_type="table",
            score=0.79,
            metadata={"page_number": 9},
        )
        results = text_chunks + [image_chunk, table_chunk]
    else:
        results = text_chunks

    store = AsyncMock()
    store.search = AsyncMock(return_value=results)
    store.connect = AsyncMock()
    return store


def make_mock_cache(cache_hit: bool = False) -> Any:
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None if not cache_hit else {
        "answer": "This answer was served from the semantic cache.",
        "sources": [],
        "metadata": {},
    })
    cache.set = AsyncMock()
    cache.health_check = AsyncMock(return_value=True)
    cache.stats = MagicMock(return_value={
        "hits": 3, "misses": 7, "hit_rate": 0.3,
        "avg_hit_latency_ms": 1.2, "avg_miss_latency_ms": 4.8,
    })
    cache.connect = AsyncMock()
    return cache


def make_mock_anthropic_client(answer: str) -> Any:
    msg = MagicMock()
    msg.content = [MagicMock(text=answer)]
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=msg)
    return client


# ── Pipeline builders ─────────────────────────────────────────────────────────

async def build_fastest_rag_mock(query: str) -> Any:
    from fastest_rag.pipeline import NaiveRAGPipeline

    emb_svc = make_mock_embedding_service()
    vec_store = make_mock_vector_store(multimodal=False)
    cache = make_mock_cache()

    answer = (
        "Retrieval-Augmented Generation (RAG) is a technique that combines vector-based "
        "retrieval with language model generation [Chunk 1]. The dense retrieval step uses "
        "HNSW-indexed embeddings for sub-millisecond nearest-neighbour search [Chunk 2]."
    )

    pipeline = NaiveRAGPipeline(
        vector_store=vec_store,
        embedding_service=emb_svc,
        cache_layer=cache,
        anthropic_api_key="sk-mock",
    )
    pipeline._client = make_mock_anthropic_client(answer)
    return pipeline


async def build_multimodal_rag_mock(query: str) -> Any:
    from multimodal_rag.pipeline import MultimodalRAGPipeline

    emb_svc = make_mock_embedding_service()
    vec_store = make_mock_vector_store(multimodal=True)
    cache = make_mock_cache()

    answer = (
        "Retrieval-Augmented Generation combines dense retrieval with language model "
        "generation [Chunk 1 (Text)]. "
        "Figure 3 demonstrates that multimodal retrieval achieves 0.93 recall@10, "
        "outperforming text-only approaches [Chunk 3 (Image)]. "
        "Table 2 confirms the improvement: multimodal RAG scores 0.91 on context recall "
        "versus 0.74 for text-only [Chunk 4 (Table)]."
    )

    pipeline = MultimodalRAGPipeline(
        vector_store=vec_store,
        embedding_service=emb_svc,
        cache_layer=cache,
        anthropic_api_key="sk-mock",
        track_provenance=True,
    )
    pipeline._client = make_mock_anthropic_client(answer)
    return pipeline


async def build_fastest_rag_live() -> Any:
    """Build NaiveRAGPipeline connected to real services (requires .env + Docker)."""
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from shared.config import get_settings
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import PgVectorClient
    from fastest_rag.cache_layer import CacheLayer
    from fastest_rag.pipeline import NaiveRAGPipeline

    settings = get_settings()
    emb_svc = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )
    emb_svc.connect()

    vec_store = PgVectorClient(database_url=settings.database_url)
    await vec_store.connect()

    cache = CacheLayer(
        redis_url=settings.redis_url,
        similarity_threshold=settings.semantic_cache_threshold,
        ttl=settings.redis_cache_ttl,
        embedding_service=emb_svc,
    )
    await cache.connect()

    pipeline = NaiveRAGPipeline(
        vector_store=vec_store,
        embedding_service=emb_svc,
        cache_layer=cache,
        anthropic_api_key=settings.anthropic_api_key,
    )
    pipeline.connect()
    return pipeline


async def build_multimodal_rag_live() -> Any:
    """Build MultimodalRAGPipeline connected to real services (requires .env + Docker)."""
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from shared.config import get_settings
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import PgVectorClient
    from fastest_rag.cache_layer import CacheLayer
    from multimodal_rag.pipeline import MultimodalRAGPipeline

    settings = get_settings()
    emb_svc = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )
    emb_svc.connect()

    vec_store = PgVectorClient(database_url=settings.database_url)
    await vec_store.connect()

    cache = CacheLayer(
        redis_url=settings.redis_url,
        similarity_threshold=settings.semantic_cache_threshold,
        ttl=settings.redis_cache_ttl,
        embedding_service=emb_svc,
    )
    await cache.connect()

    pipeline = MultimodalRAGPipeline(
        vector_store=vec_store,
        embedding_service=emb_svc,
        cache_layer=cache,
        anthropic_api_key=settings.anthropic_api_key,
        track_provenance=True,
    )
    pipeline.connect()
    return pipeline


# ── Display helpers ────────────────────────────────────────────────────────────

CHUNK_TYPE_ICONS = {
    "text": "[TXT]",
    "image_description": "[IMG]",
    "table": "[TBL]",
    "video_transcript": "[VID]",
}


def print_response(response: Any, verbose: bool = False) -> None:
    from shared.models.query import QueryResponse

    r: QueryResponse = response

    h2("Answer")
    # Wrap long lines
    words = r.answer.split()
    line, lines = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 80:
            lines.append(" ".join(line[:-1]))
            line = [w]
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        print(f"  {ln}")

    h2("Metadata")
    kv("Pipeline", r.pipeline)
    kv("Latency", f"{r.latency_ms:.1f} ms" if r.latency_ms else "n/a")
    kv("Cache hit", "✓ yes" if r.cache_hit else "✗ no")
    kv("Sources", len(r.sources))

    if r.sources:
        h2("Retrieved Sources")
        for i, s in enumerate(r.sources, 1):
            icon = CHUNK_TYPE_ICONS.get(s.chunk_type, "📄")
            page = f"  page {s.metadata.get('page_number', '?')}"
            print(f"  {icon} [{i}] {BOLD}{s.document_id}{RESET}{page}  "
                  f"score={GREEN}{s.score:.3f}{RESET}  type={CYAN}{s.chunk_type}{RESET}")
            if verbose:
                preview = s.content[:120].replace("\n", " ")
                print(f"      {DIM}{preview}…{RESET}")

    # Provenance (multimodal only)
    provenance = r.metadata.get("provenance", [])
    if provenance:
        h2("Provenance Attribution")
        for p in provenance:
            icon = CHUNK_TYPE_ICONS.get(p.get("chunk_type", "text"), "📄")
            page = f"p.{p['page_number']}" if p.get("page_number") else "—"
            confidence = p.get("confidence", 0)
            bar_len = int(confidence * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            sentence_preview = p["sentence"][:70]
            print(f"  {icon} {CYAN}{bar}{RESET} {confidence:.0%}  "
                  f"{DIM}{page}  \"{sentence_preview}…\"{RESET}")

    # Chunk type distribution (multimodal only)
    dist = r.metadata.get("chunk_type_distribution", {})
    if dist:
        h2("Chunk Type Distribution")
        for ct, count in dist.items():
            icon = CHUNK_TYPE_ICONS.get(ct, "📄")
            print(f"  {icon} {ct}: {count}")


def print_comparison(fastest_r: Any, multimodal_r: Any) -> None:
    h1("Side-by-Side Comparison")
    rows = [
        ("Latency",
         f"{fastest_r.latency_ms:.1f} ms" if fastest_r.latency_ms else "n/a",
         f"{multimodal_r.latency_ms:.1f} ms" if multimodal_r.latency_ms else "n/a"),
        ("Sources retrieved", str(len(fastest_r.sources)), str(len(multimodal_r.sources))),
        ("Cache hit", "yes" if fastest_r.cache_hit else "no",
         "yes" if multimodal_r.cache_hit else "no"),
        ("Provenance records", "—",
         str(len(multimodal_r.metadata.get("provenance", [])))),
        ("Chunk types", "text only",
         ", ".join(multimodal_r.metadata.get("chunk_type_distribution", {}).keys()) or "text"),
    ]
    print(f"  {'Metric':<26} {'Fastest RAG':<22} {'Multimodal RAG':<22}")
    print(f"  {'-' * 26} {'-' * 22} {'-' * 22}")
    for label, v1, v2 in rows:
        print(f"  {label:<26} {v1:<22} {v2:<22}")


# ── Main demo logic ───────────────────────────────────────────────────────────

async def run_pipeline_demo(
    pipeline_name: str,
    query: str,
    live: bool,
    verbose: bool,
) -> Any:
    from shared.models.query import PipelineStrategy, QueryRequest

    strategy_map = {
        "fastest_rag": PipelineStrategy.FASTEST_RAG,
        "multimodal_rag": PipelineStrategy.MULTIMODAL_RAG,
    }
    strategy = strategy_map[pipeline_name]

    h1(f"{'Fastest RAG' if pipeline_name == 'fastest_rag' else 'Multimodal RAG'} Pipeline Demo")

    if live:
        info("Mode: LIVE — connecting to real services")
    else:
        info("Mode: MOCK — using simulated services (no infrastructure required)")

    kv("Query", f'"{query}"')
    kv("Strategy", pipeline_name)

    # Build pipeline
    if live:
        builder = build_fastest_rag_live if pipeline_name == "fastest_rag" else build_multimodal_rag_live
        try:
            pipeline = await builder()
        except Exception as exc:
            err(f"Failed to connect to live services: {exc}")
            err("Is Docker running? Did you copy .env.example → .env and fill in the keys?")
            return None
    else:
        builder = (
            (lambda q: build_fastest_rag_mock(q))
            if pipeline_name == "fastest_rag"
            else (lambda q: build_multimodal_rag_mock(q))
        )
        pipeline = await builder(query)

    # Execute
    request = QueryRequest(
        query=query,
        pipeline=strategy,
        top_k=5,
        use_cache=not live,  # cache disabled in live mode to always see fresh retrieval
    )

    print()
    info("Running pipeline…")
    t0 = time.perf_counter()
    try:
        response = await pipeline.run(request)
    except Exception as exc:
        err(f"Pipeline execution failed: {exc}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None

    elapsed_wall = (time.perf_counter() - t0) * 1000
    ok(f"Pipeline completed in {elapsed_wall:.0f} ms (wall clock)")

    print_response(response, verbose=verbose)
    return response


async def main(args: argparse.Namespace) -> None:
    query = args.query
    live = args.live
    verbose = args.verbose
    pipelines = (
        [args.pipeline] if args.pipeline else ["fastest_rag", "multimodal_rag"]
    )

    h1("RAG Research Platform — Demo Runner")
    info(f"Python path includes: shared, fastest_rag, multimodal_rag, api")
    info(f"Mode: {'LIVE' if live else 'MOCK'}")
    info(f"Pipelines: {', '.join(pipelines)}")

    responses: dict[str, Any] = {}
    for name in pipelines:
        response = await run_pipeline_demo(name, query, live, verbose)
        if response:
            responses[name] = response

    if len(responses) == 2:
        print_comparison(responses["fastest_rag"], responses["multimodal_rag"])

    h1("Demo Complete")
    ok(f"Ran {len(responses)}/{len(pipelines)} pipeline(s) successfully")
    if not live:
        info("Tip: run with --live to use real infrastructure (Docker + .env).")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Research Platform — demo both RAG pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Connect to real services (requires Docker + .env). Default: mock mode.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["fastest_rag", "multimodal_rag"],
        default=None,
        help="Run only one pipeline. Default: both.",
    )
    parser.add_argument(
        "--query",
        default="What is Retrieval-Augmented Generation and how does it work?",
        help="Query to run against both pipelines.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full source content previews.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="as_json",
        help="Dump raw JSON responses to stdout instead of formatted output.",
    )

    args = parser.parse_args()

    if args.as_json:
        # JSON dump mode — suppress colour output, print raw dicts
        async def json_mode() -> None:
            from shared.models.query import PipelineStrategy, QueryRequest

            pipelines = (
                [args.pipeline] if args.pipeline else ["fastest_rag", "multimodal_rag"]
            )
            results: dict[str, Any] = {}
            for name in pipelines:
                build_fn = (
                    build_fastest_rag_mock if name == "fastest_rag" else build_multimodal_rag_mock
                )
                pipeline = await build_fn(args.query)
                strategy = (
                    PipelineStrategy.FASTEST_RAG
                    if name == "fastest_rag"
                    else PipelineStrategy.MULTIMODAL_RAG
                )
                request = QueryRequest(query=args.query, pipeline=strategy, top_k=5)
                response = await pipeline.run(request)
                results[name] = json.loads(response.model_dump_json())
            print(json.dumps(results, indent=2))

        asyncio.run(json_mode())
    else:
        asyncio.run(main(args))
