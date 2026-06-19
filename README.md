# RAG Research Platform

[![CI](https://github.com/asadhanif3188/rag-research-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/asadhanif3188/rag-research-platform/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A unified monorepo showcasing **five production-grade RAG pipeline strategies** — built as a single platform with shared infrastructure, an A/B comparison UI, real-time metrics dashboard, and RAGAS-based evaluation.

> **Why this project?** Built to compare RAG strategies head-to-head on the same corpus, with real metrics — not just vibes.

## Architecture

```
                            ┌──────────────────────────┐
                            │     Chainlit UI           │
                            │  ┌──────┐ ┌───────────┐  │
                            │  │Select│ │A/B Compare│  │
                            │  │Panel │ │   View    │  │
                            │  └──┬───┘ └─────┬─────┘  │
                            │     │           │         │
                            │  ┌──▼───────────▼──────┐  │
                            │  │  Metrics Dashboard  │  │
                            │  └─────────┬───────────┘  │
                            └────────────┼──────────────┘
                                         │ HTTP
                            ┌────────────▼──────────────┐
                            │    FastAPI Router          │
                            │    POST /query             │
                            │    GET  /metrics/summary   │
                            │    POST /benchmark/run     │
                            ├───────────────────────────-┤
                            │  Observability Middleware   │
                            │  (cost tracking, LangFuse) │
                            └────────────┬──────────────┘
                                         │
              ┌──────────┬───────────┬───┴───┬───────────┐
              ▼          ▼           ▼       ▼           ▼
        ┌──────────┐┌──────────┐┌────────┐┌────────┐┌─────────┐
        │ Fastest  ││Multimodal││  CRAG  ││Self-RAG││Video RAG│
        │   RAG    ││   RAG    ││        ││        ││ + MCP   │
        │          ││          ││LangGraph││Agentic ││Whisper  │
        │  Redis   ││Provenance││+Tavily ││+HyDE   ││+CLIP    │
        │  Cache   ││ Tracker  ││Fallback││Grading ││+Neo4j   │
        └────┬─────┘└────┬─────┘└───┬────┘└───┬────┘└────┬────┘
             │           │          │         │          │
             └─────┬─────┴────┬─────┴─────┬───┘          │
                   ▼          ▼           ▼              ▼
            ┌───────────┐┌─────────┐┌──────────┐ ┌───────────┐
            │ pgvector  ││  Redis  ││ Embedding│ │  Neo4j    │
            │(Postgres) ││ Cache   ││ Service  │ │  Graph    │
            └───────────┘└─────────┘└──────────┘ └───────────┘
```

```
rag-research-platform/
├── shared/              ← Core: models, storage, embeddings, ingestion, eval
├── pipelines/
│   ├── fastest_rag/     ← Phase 1: Baseline + Redis semantic cache
│   ├── multimodal_rag/  ← Phase 2: Text + image + table with provenance
│   ├── corrective_rag/  ← Phase 3: LangGraph + relevance grading + web search
│   ├── self_rag/        ← Phase 4: Agentic graph + hallucination detection
│   └── video_rag/       ← Phase 5: Whisper + CLIP + Neo4j knowledge graph
├── api/                 ← FastAPI router + cost tracking middleware
├── ui/                  ← Chainlit: pipeline selector, A/B compare, metrics
├── infra/               ← SQL init scripts
├── docs/                ← Deployment guide
└── docker-compose.yml
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- Docker & Docker Compose

### Setup

```bash
# Clone and configure
git clone https://github.com/asadhanif3188/rag-research-platform
cd rag-research-platform
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY

# Start infrastructure
docker-compose up -d

# Install all workspace packages
uv sync --all-packages

# Start the API server
uv run uvicorn api.src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start the Chainlit UI (separate terminal)
uv run chainlit run ui/app.py --port 8501
```

## Pipeline Strategies

### 1. Naive RAG (Fastest)

**When to use:** Lowest latency, cost-optimized queries where accuracy is acceptable.

- Embed → retrieve top-k → generate with Claude
- **Redis semantic cache** with configurable similarity threshold (default 0.92)
- Tracks cache hit/miss rates and embedding costs

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?", "pipeline": "fastest_rag"}'
```

### 2. Multimodal RAG

**When to use:** Documents with images, tables, and mixed content types.

- Per-type retrieval quotas: 50% text / 30% image / 20% table
- **Provenance tracking**: maps each answer sentence to its source chunk
- Colour-coded attribution UI with confidence scores

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me the performance comparison table", "pipeline": "multimodal_rag"}'
```

### 3. Corrective RAG (CRAG)

**When to use:** Queries that may need external knowledge when local docs are insufficient.

- **LangGraph** state machine: retrieve → grade → route
- Three-way routing: RELEVANT → generate, IRRELEVANT → web search (Tavily), AMBIGUOUS → decompose
- Claude Haiku for cost-efficient relevance grading

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest ML trends in 2026?", "pipeline": "corrective_rag"}'
```

### 4. Self-RAG

**When to use:** High-stakes queries requiring maximum answer quality and grounding.

- Most complex pipeline with **4 decision nodes**
- Retrieve-or-not → relevance grading → hallucination check → answer quality verification
- **HyDE** (Hypothetical Document Expansion) for ambiguous queries
- Max 2 retry loops before returning best-effort answer

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the transformer attention mechanism in detail", "pipeline": "self_rag"}'
```

### 5. Video RAG with MCP

**When to use:** Searching and querying video content by text or visual similarity.

- **Whisper** transcription → timestamp-chunked segments
- **CLIP** visual embeddings for keyframe search
- Dual retrieval: `fused_score = text_weight × text_score + visual_weight × visual_score`
- **Neo4j** knowledge graph: Video → Topic → Segment relationships
- **MCP server** exposes Claude-callable tools: `search_video`, `get_segment`, `list_videos`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find the section about embeddings in the lecture", "pipeline": "video_rag"}'
```

## A/B Comparison

Compare any two pipelines side-by-side on the same query:

- Runs both pipelines in parallel
- Displays answers, sources, cost, and latency
- Highlights the winner for each metric (lower latency/cost = green)
- Available in the Chainlit UI compare page

## Metrics Dashboard

Real-time pipeline performance monitoring:

| Metric | Description |
|--------|-------------|
| **RAGAS Faithfulness** | How grounded is the answer in retrieved context |
| **RAGAS Relevancy** | How relevant is the answer to the query |
| **Context Precision** | Fraction of retrieved chunks that are relevant |
| **Context Recall** | Fraction of relevant chunks that were retrieved |
| **p50/p95/p99 Latency** | Response time percentiles |
| **QPS** | Queries per second throughput |
| **Avg Cost** | USD cost per query (LLM + embedding tokens) |
| **Cache Hit Rate** | Semantic cache effectiveness (Fastest RAG) |

Export metrics as CSV for external analysis.

## API Reference

### POST /query

Route a query to a specific pipeline.

```json
{
  "query": "What is RAG?",
  "pipeline": "fastest_rag",
  "top_k": 5,
  "use_cache": true
}
```

**Response:**

```json
{
  "query": "What is RAG?",
  "answer": "RAG (Retrieval-Augmented Generation) combines...",
  "pipeline": "fastest_rag",
  "sources": [
    {
      "chunk_id": "abc123",
      "document_id": "paper-001",
      "content": "Retrieved chunk text...",
      "score": 0.94,
      "metadata": {"page_number": 3}
    }
  ],
  "latency_ms": 245.3,
  "cache_hit": false,
  "metadata": {
    "total_cost_usd": 0.00035,
    "input_tokens": 1200,
    "output_tokens": 350
  }
}
```

### GET /metrics/summary

Aggregated metrics per pipeline (RAGAS scores, latency percentiles, cost, cache stats).

### GET /metrics/history?pipeline=self_rag&limit=100

Time-series query metrics, optionally filtered by pipeline.

### POST /benchmark/run

Queue an async benchmark run comparing full-precision vs quantized retrieval.

### GET /health

Service health check.

## Shared Infrastructure

### Key Modules

| Module | Description |
|--------|-------------|
| `shared.config` | Pydantic Settings — all config from `.env` |
| `shared.models` | `QueryRequest`, `QueryResponse`, `RetrievalResult`, `EvalResult` |
| `shared.storage.vector_store` | `PgVectorClient` — pgvector with cosine similarity |
| `shared.storage.cache` | `RedisSemanticCache` — embedding-based query cache |
| `shared.storage.neo4j_client` | `Neo4jClient` — video knowledge graph |
| `shared.embeddings.service` | `EmbeddingService` — batched OpenAI embeddings |
| `shared.ingestion.pipeline` | PDF → chunks → embeddings → vector store |
| `shared.eval.ragas_runner` | RAGAS evaluation (faithfulness, relevancy, precision, recall) |

### Ingest a PDF

```python
from shared.ingestion.pipeline import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline(vector_store=vs, embedding_service=emb)
result = await pipeline.ingest_pdf("paper.pdf")
print(f"Stored {result.total_chunks} chunks in {result.elapsed_seconds:.2f}s")
```

## Environment Variables

See [.env.example](.env.example) for the full list.

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for LLM generation |
| `OPENAI_API_KEY` | Yes | OpenAI key for text-embedding-3-large |
| `TAVILY_API_KEY` | Yes | Tavily key for CRAG web search fallback |
| `DATABASE_URL` | No | PostgreSQL+asyncpg connection string |
| `REDIS_URL` | No | Redis connection string |
| `NEO4J_URI` | No | Neo4j Bolt URI |
| `LANGFUSE_SECRET_KEY` | No | LangFuse observability secret |

## Cost Estimates

| Pipeline | Avg Cost/Query | Notes |
|----------|---------------|-------|
| Fastest RAG | ~$0.0003 | Cached queries are free |
| Multimodal RAG | ~$0.0012 | More context = more tokens |
| CRAG | ~$0.0008 | Haiku grading + occasional web search |
| Self-RAG | ~$0.0015 | Multiple LLM calls (grading + generation) |
| Video RAG | ~$0.0010 | Dual retrieval + generation |

**At 1000 queries/day:** ~$74/month total (API + infrastructure). See [docs/deployment.md](docs/deployment.md).

## Testing

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m unit

# Integration tests (requires Docker services)
uv run pytest -m integration

# E2E API tests
uv run pytest api/tests/e2e/ -v

# Coverage
uv run pytest --cov=shared --cov=api --cov-report=term-missing
```

## Linting

```bash
uv run ruff format .
uv run ruff check .
```

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Shared Infrastructure | **Done** | Vector store, ingestion, eval harness |
| 1 — Fastest RAG Stack | **Done** | Baseline + Redis semantic cache + benchmarks |
| 2 — Multimodal RAG | **Done** | Vision descriptions + provenance tracking |
| 3 — Corrective RAG (CRAG) | **Done** | LangGraph + relevance grading + Tavily fallback |
| 4 — Self-RAG | **Done** | Agentic decision graph + HyDE + hallucination check |
| 5 — Video RAG with MCP | **Done** | Whisper + CLIP + Neo4j + MCP server |
| 6 — Integration & Polish | **Done** | Unified UI, A/B comparison, metrics, deployment |

## Key Design Decisions

- **Monorepo over separate repos** — Shared embedding service, vector store, ingestion pipeline, and RAGAS eval harness would otherwise be duplicated 5x. uv workspaces give per-pipeline isolation with shared dependencies.
- **Claude over GPT-4 for generation** — Sonnet for generation, Haiku for grading. Using Haiku for yes/no grading decisions reduces cost ~10x vs. using a frontier model.
- **Three storage engines for three access patterns** — pgvector for semantic retrieval (lives alongside app data), Redis for sub-millisecond semantic caching, Neo4j for video knowledge graph traversal.
- **RAGAS for evaluation** — Industry-standard RAG metrics (faithfulness, relevancy, precision, recall) enabling apples-to-apples pipeline comparison on the same 50-query test set.
- **Chainlit over Next.js** — Production-quality chat UI out-of-the-box with streaming, file upload, and custom components. FastAPI backend is fully decoupled for future frontend swaps.

For the full rationale, see [docs/architecture-decisions.md](docs/architecture-decisions.md).

## Tech Stack

Python 3.12 · uv workspaces · FastAPI · Chainlit · LangGraph · Claude API · OpenAI Embeddings · pgvector · Qdrant · Redis · Neo4j · LangFuse · RAGAS · Whisper · CLIP · MCP · Pydantic v2

## Future Work

- **Streaming responses**: SSE streaming for real-time answer generation
- **Fine-tuned graders**: Train task-specific relevance/hallucination graders on domain data
- **Nomic Embeddings**: Replace OpenAI embeddings with open-weight Nomic-embed for cost savings
