# Architecture Decisions: RAG Research Platform

**Author:** Asad Hanif

---

## Table of Contents

1. [Project Overview & Architecture Decision](#1-project-overview--architecture-decision)
2. [Repository Structure](#2-repository-structure)
3. [Technology Stack & Dependencies](#3-technology-stack--dependencies)
4. [Implementation Phases](#4-implementation-phases)
5. [LangGraph Graph Designs](#5-langgraph-graph-designs)
6. [Data Models (Pydantic Schemas)](#6-data-models-pydantic-schemas)
7. [Evaluation Strategy (RAGAS)](#7-evaluation-strategy-ragas)
8. [API Design (FastAPI Endpoints)](#8-api-design-fastapi-endpoints)
9. [Testing Strategy](#9-testing-strategy)
10. [Observability & Monitoring Setup](#10-observability--monitoring-setup)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. Project Overview & Architecture Decision

### Five Projects as One Unified RAG Research Platform

Rather than five isolated demos, all five projects are built as configurable pipeline strategies within a single **RAG Research Platform**. A shared UI lets users select which retrieval mode to use, run A/B comparisons, and view metrics side-by-side — making the portfolio story far more compelling than disconnected projects.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  rag-research-platform (monorepo)                   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │               Shared UI (Chainlit / Streamlit)                 │ │
│  │   Pipeline selector · Query input · Answer + provenance view   │ │
│  │   Metrics dashboard · A/B comparison mode                      │ │
│  └───────────────────────────┬────────────────────────────────────┘ │
│                              │                                      │
│         ┌────────────────────▼──────────────────────┐               │
│         │          Pipeline Router (FastAPI)        │               │
│         │  Routes queries to selected RAG strategy  │               │
│         └──┬───────────┬──────────┬──────┬──────────┘               │
│            │           │          │      │                          │
│     ┌──────▼──┐  ┌─────▼──┐ ┌─────▼─┐ ┌──▼──────┐  ┌───────────┐    │
│     │Video RAG│  │  CRAG  │ │Fastest│ │Multimod.│  │Self-RAG   │    │
│     │(MCP)    │  │        │ │ Stack │ │  RAG    │  │(LangGraph)│    │
│     └────┬────┘  └────┬───┘ └────┬──┘ └──┬──────┘  └────┬──────┘    │
│          └────────────┴──────────┴───────┴──────────────┘           │
│                              │                                      │
│  ┌───────────────────────────▼────────────────────────────────────┐ │
│  │                    Shared Infrastructure                       │ │
│  │  Vector store (pgvector/Qdrant) · Embedding service            │ │
│  │  Document ingestion pipeline · RAGAS eval harness              │ │
│  │  LangFuse observability · Redis cache · Neo4j (for Video RAG)  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Monorepo Decision: Yes, Single Monorepo

**Rationale:**
- Shared embedding service, vector store clients, document ingestion, and RAGAS evaluation code would otherwise be duplicated across all 5 projects
- Single CI/CD pipeline and Docker Compose for local development
- Portfolio reviewers clone one repo and see the entire system with cross-project comparisons
- Shared Pydantic schemas ensure consistent data contracts between pipeline strategies

**Package manager:** `uv` with workspace support

### Build Order

| Order | Project | Why This Order |
|-------|---------|----------------|
| 1 | **Shared Infrastructure** | Vector store, ingestion, eval harness — all projects depend on this |
| 2 | **Fastest RAG Stack** | Baseline naive RAG + binary quantization — simplest pipeline, establishes the benchmark |
| 3 | **Multimodal RAG** | Extends ingestion pipeline with vision capabilities — highest job market demand |
| 4 | **Corrective RAG (CRAG)** | Adds self-assessment loop on top of existing vector retrieval |
| 5 | **Self-RAG with LangGraph** | Extends CRAG with full agentic decision graph |
| 6 | **MCP-powered RAG over Videos** | Most unique, most complex — built last when all foundations are in place |

---

## 2. Repository Structure

```
rag-research-platform/
├── pyproject.toml                  # uv workspace root
├── docker-compose.yml              # pgvector, Qdrant, Redis, Neo4j, LangFuse
├── .env.example
├── README.md
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint (ruff) + type check (mypy) + unit tests on every PR
│       ├── integration-tests.yml   # Integration tests with Docker services (on merge to main)
│       └── eval.yml                # RAGAS evaluation on demand (workflow_dispatch)
│
├── shared/                         # Shared Python package
│   ├── pyproject.toml
│   └── src/shared/
│       ├── config.py               # Pydantic settings (env vars)
│       ├── models/                 # Shared Pydantic data models
│       │   ├── document.py
│       │   ├── query.py
│       │   └── retrieval.py
│       ├── storage/
│       │   ├── vector_store.py     # pgvector + Qdrant abstraction
│       │   ├── cache.py            # Redis semantic cache
│       │   └── neo4j_client.py     # For Video RAG knowledge graph
│       ├── ingestion/
│       │   ├── pdf_parser.py       # PyMuPDF + vision model descriptions
│       │   ├── video_parser.py     # Whisper transcription + CLIP embeddings
│       │   └── chunking.py         # Fixed, semantic, scene-based chunkers
│       ├── embeddings/
│       │   └── service.py          # Unified embedding service (OpenAI/local)
│       └── eval/
│           └── ragas_runner.py     # RAGAS evaluation harness
│
├── pipelines/
│   ├── fastest_rag/                # Baseline + Redis semantic cache
│   │   └── src/fastest_rag/
│   │       ├── pipeline.py         # Naive RAG + binary quantized retrieval
│   │       ├── benchmark.py        # Latency/throughput benchmarking
│   │       └── cache_layer.py      # Redis semantic cache integration
│   │
│   ├── multimodal_rag/             # Text + image + table with provenance
│   │   └── src/multimodal_rag/
│   │       ├── pipeline.py
│   │       ├── vision_describer.py # Claude vision for tables+images
│   │       └── provenance.py       # Source attribution tracking
│   │
│   ├── corrective_rag/             # LangGraph CRAG
│   │   └── src/corrective_rag/
│   │       ├── graph.py            # LangGraph workflow
│   │       ├── relevance_grader.py
│   │       ├── query_rewriter.py
│   │       └── web_searcher.py     # Tavily fallback
│   │
│   ├── self_rag/                   # Agentic Self-RAG
│   │   └── src/self_rag/
│   │       ├── graph.py            # LangGraph stateful graph
│   │       ├── retrieval_grader.py
│   │       ├── hallucination_grader.py
│   │       ├── answer_grader.py
│   │       └── hyde.py             # HyDE query expansion
│   │
│   └── video_rag/                  # MCP Video RAG
│       └── src/video_rag/
│           ├── mcp_server.py       # FastMCP server exposing video tools
│           ├── video_indexer.py    # Whisper + CLIP + timestamp chunking
│           ├── segment_retriever.py
│           └── knowledge_graph.py  # Neo4j video-topic graph
│
├── api/                            # FastAPI pipeline router
│   └── src/api/
│       ├── main.py
│       ├── routers/
│       │   ├── query.py            # POST /query — routes to selected pipeline
│       │   ├── benchmark.py        # GET /benchmark — run perf tests
│       │   └── metrics.py          # GET /metrics — aggregated stats
│       └── middleware/
│           └── observability.py    # LangFuse tracing + cost tracking
│
└── ui/                             # Chainlit frontend
    ├── app.py
    ├── components/
    │   ├── pipeline_selector.py
    │   ├── provenance_viewer.py    # Highlights source chunks in answers
    │   └── video_player.py         # Timestamp-linked video segments
    └── pages/
        ├── compare.py              # A/B comparison of two pipelines
        ├── metrics_dashboard.py    # Live metrics dashboard
        └── video_search.py         # Multi-video Q&A
```

---

## 3. Technology Stack & Dependencies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.12 | All services |
| Package manager | `uv` with workspaces | Fast dependency management |
| API | FastAPI + Uvicorn | Pipeline router, ingestion endpoints |
| UI | Chainlit (primary) | Chat interface + dashboards |
| Orchestration | LangGraph | CRAG and Self-RAG stateful graphs |
| LLM | Claude API (Sonnet / Haiku) | Generation, grading, vision |
| Embeddings | OpenAI `text-embedding-3-large` | Document and query embedding |
| Vector store | pgvector (primary) + Qdrant (binary quantization) | Semantic retrieval |
| Cache | Redis | Semantic query cache (Fastest RAG) |
| Knowledge graph | Neo4j | Video topic graph (MCP Video RAG) |
| Observability | LangFuse | Full LLM trace visibility |
| Evaluation | RAGAS | Automated RAG quality metrics |
| Video ASR | OpenAI Whisper | Transcription for Video RAG |
| Video vision | CLIP (OpenAI) | Visual embeddings per frame |
| PDF parsing | PyMuPDF (fitz) | Text + image extraction |
| Web search | Tavily API | Corrective fallback in CRAG |
| Containerization | Docker + Docker Compose | Local development |
| Validation | Pydantic v2 | Data models and settings |

---

## 4. Implementation Phases

### Phase 0: Shared Infrastructure
Monorepo setup, shared data models, vector store abstraction, document ingestion pipeline, embedding service, evaluation harness, and GitHub Actions CI/CD (lint, type check, tests, integration tests, RAGAS eval).

### Phase 1: Fastest RAG Stack
Baseline naive RAG pipeline with binary quantization benchmarks. Redis semantic cache with configurable similarity threshold. Benchmark dashboard for latency/throughput comparison.

### Phase 2: Multimodal RAG
Extended ingestion pipeline with vision model descriptions for images and tables. Provenance tracking maps each answer sentence back to its source chunk. Per-type retrieval quotas: 50% text / 30% image / 20% table.

### Phase 3: Corrective RAG (CRAG)
LangGraph state machine: retrieve → grade → route. Three-way routing: RELEVANT → generate, IRRELEVANT → web search (Tavily), AMBIGUOUS → decompose. Claude Haiku for cost-efficient relevance grading.

### Phase 4: Self-RAG with LangGraph
Extended CRAG with four decision nodes: retrieve-or-not, relevance grading, hallucination check, answer quality verification. HyDE (Hypothetical Document Expansion) for ambiguous queries. Max 2 retry loops before returning best-effort answer.

### Phase 5: MCP-powered RAG over Videos
Video indexing pipeline (Whisper + CLIP), MCP server exposing Claude-callable tools, timestamp-based dual retrieval (text + visual), Neo4j topic graph, and video player UI component.

### Phase 6: Integration & Polish
Unified Chainlit UI with pipeline selector, A/B comparison mode, real-time metrics dashboard, cost tracking middleware, and comprehensive documentation.

---

## 5. LangGraph Graph Designs

### CRAG Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│  State: { query, documents, web_results, answer, grade, iterations } │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                         ┌─────▼─────┐
                         │  retrieve │  → vector store top-k
                         └─────┬─────┘
                               │
                    ┌──────────▼──────────┐
                    │  grade_documents    │  → Claude grades each doc
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │ RELEVANT          │ AMBIGUOUS          │ IRRELEVANT
           │                   │                    │
    ┌──────▼──────┐   ┌────────▼────────┐  ┌───────▼──────────┐
    │  generate   │   │  decompose_docs  │  │  rewrite_query   │
    └─────────────┘   └────────┬────────┘  └───────┬──────────┘
                               │                    │
                       ┌───────▼────┐      ┌────────▼─────────┐
                       │  generate  │      │  web_search      │
                       └────────────┘      └────────┬─────────┘
                                                     │
                                             ┌───────▼────┐
                                             │  generate  │
                                             └────────────┘
```

### Self-RAG Graph

```
┌────────────────────────────────────────────────────────────────────────┐
│  State: { query, retrieve_needed, documents, answer, grades, attempts } │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  retrieve_or_not    │  → NO → direct_generate
                     └──────────┬──────────┘
                                │ YES
                     ┌──────────▼──────────┐
                     │  retrieve           │  → vector store top-k
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  grade_relevance    │  → FAIL → hyde_expand → retrieve
                     └──────────┬──────────┘
                                │ PASS
                     ┌──────────▼──────────┐
                     │  generate           │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  grade_grounding    │  → FAIL → rewrite → retrieve (max 2x)
                     └──────────┬──────────┘
                                │ PASS
                     ┌──────────▼──────────┐
                     │  grade_answer       │  → FAIL → rewrite query → restart
                     └──────────┬──────────┘
                                │ PASS
                             ┌──▼──┐
                             │ END │
                             └─────┘
```

---

## 6. Data Models (Pydantic Schemas)

```python
# shared/src/shared/models/document.py

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime


class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE_DESCRIPTION = "image_description"
    TABLE = "table"
    VIDEO_TRANSCRIPT = "video_transcript"


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    embedding: Optional[list[float]] = None
    # Provenance metadata
    source_file: str
    page_number: Optional[int] = None
    bounding_box: Optional[dict] = None   # {x1, y1, x2, y2} for PDF images
    start_timestamp: Optional[float] = None  # seconds, for video chunks
    end_timestamp: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float
    retrieval_method: str   # "vector", "bm25", "hybrid", "web"


class QueryRequest(BaseModel):
    query: str
    pipeline: str = "self_rag"
    top_k: int = 5
    use_cache: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievalResult]
    pipeline_used: str
    token_cost_usd: float
    latency_ms: float
    trace_url: Optional[str] = None    # LangFuse trace link


class EvalResult(BaseModel):
    pipeline: str
    faithfulness: float         # 0-1: answer grounded in context
    answer_relevancy: float     # 0-1: answer relevant to question
    context_precision: float    # 0-1: retrieved context is precise
    context_recall: float       # 0-1: relevant context was retrieved
    num_queries: int
    avg_latency_ms: float
    avg_cost_usd: float
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
```

---

## 7. Evaluation Strategy (RAGAS)

### Test Dataset Composition

Each pipeline is evaluated against the same 50-query golden test set for apples-to-apples comparison:

- 10 factual retrieval queries (answer is a specific fact in the docs)
- 10 multi-hop reasoning queries (answer requires combining 2+ chunks)
- 10 table/chart queries (only answerable from structured data — tests multimodal retrieval)
- 10 adversarial queries (answer is NOT in the docs — tests hallucination resistance)
- 10 video-specific queries (timestamp questions — for Video RAG pipeline)

### Metrics Tracked

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Faithfulness | Answer is supported by retrieved context | >0.85 |
| Answer Relevancy | Answer addresses the question | >0.80 |
| Context Precision | Retrieved chunks are on-topic | >0.75 |
| Context Recall | Correct chunks were retrieved | >0.70 |
| Hallucination Rate | % answers with unsupported claims | <10% |

### Expected Results (Hypothesis)

| Pipeline | Faithfulness | Answer Relevancy | Context Recall | Hallucination Rate |
|----------|-------------|-----------------|----------------|--------------------|
| Naive RAG | ~0.70 | ~0.72 | ~0.60 | ~25% |
| Fastest RAG (BQ) | ~0.68 | ~0.72 | ~0.55 | ~27% |
| Multimodal RAG | ~0.75 | ~0.80 | ~0.80* | ~20% |
| CRAG | ~0.82 | ~0.78 | ~0.65 | ~12% |
| Self-RAG | ~0.88 | ~0.82 | ~0.70 | ~7% |

*Multimodal excels at context recall because it retrieves from images and tables, not just text.

---

## 8. API Design (FastAPI Endpoints)

```
POST   /api/query                   # Run a query through selected pipeline
POST   /api/ingest/document         # Upload PDF for ingestion
POST   /api/ingest/video            # Submit video URL for indexing
GET    /api/pipelines               # List available pipelines and their status
POST   /api/eval/run                # Run RAGAS evaluation on a test set
GET    /api/eval/results            # Get latest eval results per pipeline
GET    /api/benchmark/run           # Run latency benchmark (Fastest RAG)
GET    /api/benchmark/results       # Get benchmark results
GET    /api/metrics                 # Real-time metrics (token cost, latency, cache hit rate)
GET    /api/traces/{trace_id}       # Get LangFuse trace for a request
```

Unified query request/response — all pipelines use the same `QueryRequest` / `QueryResponse` schemas defined in Section 6.

---

## 9. Testing Strategy

### Test Pyramid

```
        /\
       /  \         E2E Tests (5): One per pipeline, via API
      /────\        — Submit query → assert answer + sources returned
     /      \
    /────────\      Integration Tests (20): Pipeline internals
   /          \     — CRAG graph paths (relevant / ambiguous / irrelevant)
  /────────────\    — Self-RAG grading nodes
 /              \   — Video retrieval with real Whisper output
/────────────────\
                    Unit Tests (50+): Individual components
                    — Graders (mock LLM calls)
                    — Chunkers
                    — ProvenanceTracker
                    — BenchmarkRunner
                    — Cache layer
```

### Testing Tools

- `pytest` + `pytest-asyncio` — async test support
- `pytest-mock` — mock LLM API calls (avoid real API costs in unit tests)
- `respx` — mock httpx calls (Tavily, OpenAI)
- `testcontainers` — spin up real pgvector and Redis for integration tests
- RAGAS for automated quality evaluation (runs separately via GitHub Actions)

---

## 10. Observability & Monitoring Setup

### LangFuse Integration

Every LLM call, every graph node, and every retrieval operation is traced in LangFuse.

**Trace structure per query:**
```
Trace: query_id
├── Span: retrieve (latency, top-k scores)
├── Span: grade_documents (LLM call, grades per doc)
│   ├── Generation: relevance_grade (prompt, response, tokens, cost)
├── Span: generate (LLM call)
│   ├── Generation: answer (prompt, response, tokens, cost)
└── Span: grade_hallucination (LLM call)
    └── Generation: grounding_check (prompt, response, tokens, cost)
```

### Custom Metrics Logged

- `pipeline_used`: which RAG strategy was selected
- `cache_hit`: whether Redis cache served the response
- `retrieval_method`: vector / hybrid / web
- `total_cost_usd`: full request cost
- `ragas_faithfulness`: if eval was run inline

### Cost Tracking

All pipelines log token usage (input + output) with model-tier pricing applied:
- Claude Haiku: $0.25/1M input, $1.25/1M output
- Claude Sonnet: $3/1M input, $15/1M output
- OpenAI `text-embedding-3-large`: $0.13/1M tokens

---

## 11. Key Design Decisions

### Why pgvector as primary, Qdrant as secondary?

pgvector lives alongside the application database (SQLAlchemy), reducing infrastructure complexity. Qdrant is added specifically for the binary quantization benchmark because it has mature BQ support and a dedicated benchmarking API. For production, either would work.

### Why Chainlit over Next.js for the UI?

Chainlit gives a production-quality chat UI out-of-the-box with streaming, file upload, and custom element support (for the provenance viewer and video player components). Next.js would require building all of this from scratch. Chainlit is the faster path to a polished demo. The FastAPI backend is fully decoupled, so migrating to Next.js later requires only a new frontend — no backend changes.

### Why not build each project as a separate repo?

The shared infrastructure (embedding service, vector store, ingestion pipeline, RAGAS harness) would need to be duplicated or published as a private package. A monorepo with uv workspaces gives the same isolation (each pipeline is an independent Python package) with shared dependencies and a single CI pipeline.

### Model selection for graders in CRAG/Self-RAG

Graders (relevance, hallucination, answer quality) use `claude-haiku-4-5` — they need a yes/no judgment, not deep reasoning. This reduces cost by ~10x versus using Sonnet for grading. The final generation step uses `claude-sonnet-4-6`. This mirrors production cost-optimization patterns.

### CLIP vs. other visual embedding models for Video RAG

CLIP (ViT-B/32 or ViT-L/14) is chosen for visual embeddings because it is open-weight, well-supported, and produces embeddings in the same semantic space as text — enabling cross-modal retrieval where a text query can retrieve visually similar frames. A future upgrade path is `nomic-embed-vision` for higher recall.
