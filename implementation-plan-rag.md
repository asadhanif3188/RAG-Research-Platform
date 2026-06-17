# Implementation Plan: RAG & Advanced Retrieval Projects

**Created:** May 26, 2026  
**Author:** Asad Hanif  
**Target:** Portfolio projects for AI Engineer / ML Engineer roles requiring RAG expertise  
**Roadmap context:** 43-week Applied AI Engineer program  
**Estimated build window:** Weeks 8-20 of the roadmap (early-to-mid program, builds foundational RAG skills before moving to agentic systems)

---

## Table of Contents

1. [Project Overview & Architecture Decision](#1-project-overview--architecture-decision)
2. [Repository Structure](#2-repository-structure)
3. [Technology Stack & Dependencies](#3-technology-stack--dependencies)
4. [Implementation Phases](#4-implementation-phases)
5. [Detailed Task Breakdown per Project](#5-detailed-task-breakdown-per-project)
6. [LangGraph Graph Designs](#6-langgraph-graph-designs)
7. [Data Models (Pydantic Schemas)](#7-data-models-pydantic-schemas)
8. [Evaluation Strategy (RAGAS)](#8-evaluation-strategy-ragas)
9. [API Design (FastAPI Endpoints)](#9-api-design-fastapi-endpoints)
10. [Testing Strategy](#10-testing-strategy)
11. [Observability & Monitoring Setup](#11-observability--monitoring-setup)
12. [Build Timeline Estimate](#12-build-timeline-estimate)

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
├── README.md                       # Root README linking to each pipeline
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint (ruff) + type check (mypy) + unit tests on every PR
│       ├── integration-tests.yml   # Integration tests with Docker services (on merge to main)
│       └── eval.yml                # RAGAS evaluation on demand (workflow_dispatch)
│
├── shared/                         # Shared Python package
│   ├── pyproject.toml
│   └── src/shared/
│       ├── __init__.py
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
│   ├── fastest_rag/               # Project 3: Fastest RAG Stack
│   │   ├── pyproject.toml
│   │   ├── README.md              # Standalone README: architecture, setup, demo, metrics
│   │   ├── Dockerfile             # Independently runnable container
│   │   ├── demo.py                # Quick Streamlit/FastAPI demo (runs standalone)
│   │   └── src/fastest_rag/
│   │       ├── pipeline.py        # Naive RAG + binary quantized retrieval
│   │       ├── benchmark.py       # Latency/throughput benchmarking
│   │       └── cache_layer.py     # Redis semantic cache integration
│   │
│   ├── multimodal_rag/            # Project 4: Multimodal RAG
│   │   ├── pyproject.toml
│   │   ├── README.md              # Standalone README: architecture, setup, demo, metrics
│   │   ├── Dockerfile             # Independently runnable container
│   │   ├── demo.py                # Quick Streamlit demo with provenance viewer
│   │   └── src/multimodal_rag/
│   │       ├── pipeline.py
│   │       ├── vision_describer.py # GPT-4o/Claude vision for tables+images
│   │       └── provenance.py       # Source attribution tracking
│   │
│   ├── corrective_rag/            # Project 2: CRAG
│   │   ├── pyproject.toml
│   │   ├── README.md              # Standalone README: architecture, setup, demo, metrics
│   │   ├── Dockerfile             # Independently runnable container
│   │   ├── demo.py                # Quick Streamlit demo showing CRAG graph decisions
│   │   └── src/corrective_rag/
│   │       ├── graph.py           # LangGraph workflow
│   │       ├── relevance_grader.py
│   │       ├── query_rewriter.py
│   │       └── web_searcher.py    # Tavily fallback
│   │
│   ├── self_rag/                  # Project 5: Self-RAG
│   │   ├── pyproject.toml
│   │   ├── README.md              # Standalone README: architecture, setup, demo, metrics
│   │   ├── Dockerfile             # Independently runnable container
│   │   ├── demo.py                # Quick Streamlit demo with graph trace visualization
│   │   └── src/self_rag/
│   │       ├── graph.py           # LangGraph stateful graph
│   │       ├── retrieval_grader.py
│   │       ├── hallucination_grader.py
│   │       ├── answer_grader.py
│   │       └── hyde.py            # HyDE query expansion
│   │
│   └── video_rag/                 # Project 1: MCP Video RAG
│       ├── pyproject.toml
│       ├── README.md              # Standalone README: architecture, setup, demo, metrics
│       ├── Dockerfile             # Independently runnable container
│       ├── demo.py                # Quick Streamlit demo with video player
│       └── src/video_rag/
│           ├── mcp_server.py      # FastMCP server exposing video tools
│           ├── video_indexer.py   # Whisper + CLIP + timestamp chunking
│           ├── segment_retriever.py
│           └── knowledge_graph.py # Neo4j video-topic graph
│
├── api/                           # FastAPI pipeline router
│   ├── pyproject.toml
│   └── src/api/
│       ├── main.py
│       ├── routers/
│       │   ├── query.py           # POST /query — routes to selected pipeline
│       │   ├── ingest.py          # POST /ingest — document/video upload
│       │   ├── benchmark.py       # GET /benchmark — run perf tests
│       │   └── eval.py            # POST /eval — run RAGAS evaluation
│       └── middleware/
│           └── observability.py   # LangFuse tracing middleware
│
└── ui/                            # Chainlit frontend
    ├── app.py                     # Main Chainlit app
    ├── components/
    │   ├── pipeline_selector.py
    │   ├── provenance_viewer.py   # Highlights source chunks in answers
    │   ├── video_player.py        # Timestamp-linked video segments
    │   └── metrics_dashboard.py
    └── pages/
        ├── chat.py
        ├── compare.py             # A/B comparison of two pipelines
        └── benchmark.py           # Live benchmark dashboard
```

---

## 3. Technology Stack & Dependencies

### Core Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.12 | All services |
| Package manager | `uv` with workspaces | Fast dependency management |
| API | FastAPI + Uvicorn | Pipeline router, ingestion endpoints |
| UI | Chainlit (primary), Streamlit (benchmark) | Chat interface + dashboards |
| Orchestration | LangGraph | CRAG and Self-RAG stateful graphs |
| LLM | Claude API (claude-sonnet-4-6 / claude-haiku-4-5) | Generation, grading, vision |
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

### Key Python Dependencies

```toml
# shared/pyproject.toml
[project]
dependencies = [
    "fastapi>=0.115",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "langchain>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-openai>=0.2",
    "langgraph>=0.2",
    "langfuse>=2.0",
    "pgvector>=0.3",
    "sqlalchemy>=2.0",
    "redis>=5.0",
    "neo4j>=5.0",
    "pymupdf>=1.24",           # PDF parsing
    "openai>=1.40",            # Embeddings + Whisper
    "anthropic>=0.34",         # Claude API
    "ragas>=0.2",              # Evaluation
    "tavily-python>=0.3",      # Web search fallback
    "httpx>=0.27",
    "asyncpg>=0.29",
    "chainlit>=1.3",
]
```

---

## 4. Implementation Phases

### Phase 0: Shared Infrastructure (Week 1-2)
Set up the monorepo, shared data models, vector store abstraction, document ingestion pipeline, embedding service, and evaluation harness. All subsequent projects depend on this. **Set up GitHub Actions CI/CD** (lint, type check, tests, integration tests, RAGAS eval).

### Phase 1: Fastest RAG Stack (Week 3-4)
Build the baseline naive RAG pipeline and add binary quantization. Establish benchmark baselines. Implement Redis semantic cache. Build the benchmark dashboard.

### Phase 2: Multimodal RAG (Week 5-7)
Extend the ingestion pipeline with vision model descriptions for images and tables. Build provenance tracking. Implement the provenance viewer UI. Run RAGAS evaluation on mixed-content PDFs. **Ship with standalone README, Dockerfile, and Streamlit demo.**

### Phase 3: Corrective RAG — CRAG (Week 8-9)
Build the LangGraph CRAG graph with relevance grading, document decomposition, query rewriting, and Tavily web search fallback. Compare RAGAS metrics against baseline naive RAG. **Ship with standalone README, Dockerfile, and Streamlit demo.**

### Phase 4: Self-RAG with LangGraph (Week 10-11)
Extend CRAG with three decision points: retrieve-or-not, relevance grading, hallucination grading, answer grading. Add HyDE query expansion. Visualize the decision graph execution trace.

### Phase 5: MCP-powered RAG over Videos (Week 12-15)
Build the video indexing pipeline (Whisper + CLIP), MCP server exposing video tools, timestamp-based retrieval, Neo4j topic graph, and clip viewer UI component.

### Phase 6: Integration & Polish (Week 16-17)
Connect all pipelines to the shared UI pipeline selector. Build A/B comparison mode. Finalize benchmark dashboard. Write comprehensive README. Record demo video.

---

## 5. Detailed Task Breakdown per Project

### Phase 0: Shared Infrastructure

```
[x] 0.1  Initialize uv monorepo with workspace pyproject.toml
[x] 0.2  Create docker-compose.yml with pgvector, Qdrant, Redis, Neo4j, LangFuse
[x] 0.3  Define shared Pydantic models: Document, Chunk, Query, RetrievalResult, EvalResult
[x] 0.4  Implement VectorStoreClient — pgvector + Qdrant abstraction with common interface
[x] 0.5  Implement EmbeddingService — wraps OpenAI text-embedding-3-large with batching
[x] 0.6  Implement DocumentIngestionPipeline — PDF → text chunks → embeddings → vector store
[x] 0.7  Implement ChunkingStrategies: fixed-size, semantic (sentence-transformers), sliding window
[x] 0.8  Implement RedisSemanticCache — embed query, check cache, return if hit
[x] 0.9  Implement RAGASRunner — faithfulness, answer relevancy, context precision, context recall
[x] 0.10 Write integration tests for vector store CRUD and embedding service
[x] 0.11 Write .env.example with all required environment variables
[x] 0.12 Set up GitHub Actions CI workflow (.github/workflows/ci.yml):
      - Lint with ruff, type check with mypy, run unit tests with pytest
      - Trigger on every PR and push to main
[x] 0.13 Set up GitHub Actions integration test workflow (.github/workflows/integration-tests.yml):
      - Spin up Docker services (pgvector, Redis) via docker-compose
      - Run integration tests
      - Trigger on merge to main
[x] 0.14 Set up GitHub Actions RAGAS eval workflow (.github/workflows/eval.yml):
      - Run RAGAS evaluation on demand (workflow_dispatch)
      - Upload eval results as artifacts
```

### Phase 1: Fastest RAG Stack

```
[x] 1.1  Implement NaiveRAGPipeline — baseline: embed query → top-k retrieval → generate
[x] 1.2  Configure Qdrant collection with binary quantization (BQ) encoding
[x] 1.3  Configure Qdrant collection with scalar quantization (SQ) for comparison
[x] 1.4  Implement BenchmarkRunner — measures latency (p50, p95, p99), throughput (QPS), recall@k
[x] 1.5  Load 1M+ sample vectors (use synthetic embeddings or public dataset)
[x] 1.6  Run benchmarks across: full precision, SQ, BQ — record results
[x] 1.7  Implement Redis semantic cache with configurable similarity threshold
[x] 1.8  Measure cache hit rate and effective latency reduction
[x] 1.9  Build Streamlit benchmark dashboard showing live latency charts and cache stats
[x] 1.10 Run RAGAS evaluation on BQ vs full-precision to show quality trade-off
[x] 1.11 Write unit tests for BenchmarkRunner and cache layer
[x] 1.12 Write standalone README for fastest_rag/ — architecture diagram, benchmark results table, cache stats, demo GIF
[x] 1.13 Write Dockerfile for fastest_rag/ — independently runnable with `docker run`
```

### Phase 2: Multimodal RAG

```
[x] 2.1  Extend DocumentIngestionPipeline with image extraction (PyMuPDF)
[x] 2.2  Implement VisionDescriber — sends images/tables to Claude vision, returns text descriptions
[x] 2.3  Implement TableExtractor — detect tables in PDFs, extract as markdown
[x] 2.4  Build MultimodalChunker — chunks text, image descriptions, and table markdown separately
[x] 2.5  Store chunks with metadata: source_file, page_number, chunk_type (text/image/table), bounding_box
[x] 2.6  Implement ProvenanceTracker — maps each answer sentence back to its source chunk(s)
[x] 2.7  Build MultimodalRAGPipeline — retrieves across text, image, and table chunk types
[x] 2.8  Implement ProvenanceViewer Chainlit component — highlights source page/section in answer
[x] 2.9  Create test dataset: 10 mixed-content PDFs (financial reports, scientific papers)
[x] 2.10 Run RAGAS evaluation on text-only vs multimodal retrieval — show recall improvement
[x] 2.11 Write tests for VisionDescriber (mock Claude API) and ProvenanceTracker
[x] 2.12 Write standalone README for multimodal_rag/ — architecture diagram, quick start, RAGAS results table, demo GIF
[x] 2.13 Build standalone demo.py (Streamlit) — upload PDF, query, see provenance highlighting, runs without full platform
[x] 2.14 Write Dockerfile for multimodal_rag/ — independently runnable with `docker run`
```

### Phase 3: Corrective RAG (CRAG)

```
[x] 3.1  Design LangGraph graph: retrieve → grade_documents → (relevant: generate) / (ambiguous: decompose) / (irrelevant: web_search) → generate
[x] 3.2  Implement RelevanceGrader — Claude grades each retrieved doc: RELEVANT / AMBIGUOUS / IRRELEVANT
[x] 3.3  Implement DocumentDecomposer — for AMBIGUOUS docs, extract only relevant sub-sections
[x] 3.4  Implement QueryRewriter — rewrites query to improve web search quality
[x] 3.5  Implement WebSearcher — Tavily API integration with result parsing and deduplication
[x] 3.6  Implement CRAGGraph — assemble full LangGraph stateful graph with conditional edges
[x] 3.7  Add LangFuse tracing to every node (inputs, outputs, latency, token cost)
[x] 3.8  Run RAGAS evaluation: naive RAG vs CRAG — show hallucination rate reduction
[x] 3.9  Write test dataset of 50 queries where naive RAG hallucinated (golden answers known)
[x] 3.10 Write unit tests for RelevanceGrader, DocumentDecomposer, QueryRewriter
[x] 3.11 Write integration test for full CRAG graph end-to-end
[x] 3.12 Write standalone README for corrective_rag/ — architecture diagram, graph flowchart, RAGAS comparison table, demo GIF
[x] 3.13 Build standalone demo.py (Streamlit) — enter query, see CRAG decision path (RELEVANT/AMBIGUOUS/IRRELEVANT), runs without full platform
[x] 3.14 Write Dockerfile for corrective_rag/ — independently runnable with `docker run`
```

### Phase 4: Self-RAG with LangGraph

```
[ ] 4.1  Extend CRAG graph with three decision point nodes: retrieve_or_not, relevance_grade, hallucination_grade, answer_grade
[ ] 4.2  Implement RetrieveOrNot — LLM decides if retrieval is needed for the given query
[ ] 4.3  Implement HallucinationGrader — checks if generated answer is grounded in retrieved docs
[ ] 4.4  Implement AnswerGrader — checks if answer actually addresses the question
[ ] 4.5  Implement HyDEQueryExpander — generate hypothetical document, use its embedding for retrieval
[ ] 4.6  Add adaptive retrieval depth: if grounding check fails, re-query with HyDE-expanded query
[ ] 4.7  Add graph execution trace visualization in Chainlit (show which nodes fired, why)
[ ] 4.8  Compare Self-RAG vs CRAG vs naive RAG on the same 50-query test set
[ ] 4.9  Write unit tests for all grader nodes
[ ] 4.10 Write integration test for full Self-RAG graph with all decision paths exercised
[ ] 4.11 Write standalone README for self_rag/ — architecture diagram, graph trace examples, RAGAS comparison table, demo GIF
[ ] 4.12 Build standalone demo.py (Streamlit) — enter query, see full decision graph trace, runs without full platform
[ ] 4.13 Write Dockerfile for self_rag/ — independently runnable with `docker run`
```

### Phase 5: MCP-powered RAG over Videos

```
[ ] 5.1  Implement VideoIndexer — downloads/accepts video, runs Whisper ASR, segments by sentence
[ ] 5.2  Implement SceneDetector — detect scene changes using frame difference (PyAV/OpenCV)
[ ] 5.3  Implement CLIPEmbedder — extract CLIP embeddings per scene keyframe
[ ] 5.4  Build hybrid video index: text embeddings (transcripts) + CLIP embeddings (frames)
[ ] 5.5  Implement TimestampChunker — each chunk has start_ts, end_ts, transcript, frame_embedding
[ ] 5.6  Implement SegmentRetriever — dual retrieval: text similarity + visual similarity, fused ranking
[ ] 5.7  Build Neo4j topic graph: videos → topics → segments (for multi-hop "find segments about X in videos tagged Y")
[ ] 5.8  Implement FastMCP server with tools: search_video, get_segment, list_videos, get_transcript
[ ] 5.9  Build Chainlit VideoPlayer component — displays retrieved video segment with seek to timestamp
[ ] 5.10 Build multi-video Q&A mode — query spans a library of indexed videos
[ ] 5.11 Test with 5 YouTube lecture videos (download via yt-dlp, index, query)
[ ] 5.12 Write unit tests for VideoIndexer, SegmentRetriever
[ ] 5.13 Write integration test for MCP server tools
[ ] 5.14 Write standalone README for video_rag/ — architecture diagram, MCP tool docs, demo GIF with video playback
[ ] 5.15 Build standalone demo.py (Streamlit) — upload/link video, query, see timestamped segment, runs without full platform
[ ] 5.16 Write Dockerfile for video_rag/ — independently runnable with `docker run`
```

### Phase 6: Integration & Polish

```
[ ] 6.1  Build PipelineSelector UI component — dropdown to select active RAG strategy
[ ] 6.2  Build A/B Comparison mode — run same query through two pipelines, show results side-by-side
[ ] 6.3  Build shared Metrics Dashboard — RAGAS scores, latency, cost per pipeline
[ ] 6.4  Connect all 5 pipelines to FastAPI router with unified request/response schema
[ ] 6.5  Add cost tracking middleware — log token usage and USD cost per request per pipeline
[ ] 6.6  Write end-to-end tests for each pipeline via the API
[ ] 6.7  Write comprehensive README with architecture diagram, setup instructions, demo GIFs
[ ] 6.8  Record 5-minute demo video showing all pipelines and the A/B comparison mode
[ ] 6.9  Deploy to Fly.io or Railway (or document Docker deployment steps)
```

---

## 5.5 Phase-by-Phase Implementation Prompts

Use these prompts with Claude Code to implement each phase systematically. Each prompt includes architecture context, tech stack, testing requirements, and success criteria.

### Phase 0: Shared Infrastructure Prompt

```
You are implementing Phase 0 (Shared Infrastructure) of the rag-research-platform monorepo.

ARCHITECTURE CONTEXT:
- Repository structure: monorepo using `uv` with workspaces
- All subsequent projects depend on this phase's output
- Docker Compose should provide pgvector, Qdrant, Redis, Neo4j, LangFuse locally
- Single shared Python package at ./shared/ contains all reusable code

WHAT TO BUILD:
1. Initialize uv monorepo root with pyproject.toml (workspace configuration)
2. Create docker-compose.yml with services:
   - PostgreSQL 16 + pgvector extension
   - Qdrant vector database
   - Redis for caching
   - Neo4j for video knowledge graphs
   - LangFuse for observability
3. Create ./shared/pyproject.toml with dependencies listed in section 3
4. Implement Pydantic models in ./shared/src/shared/models/:
   - DocumentChunk (with ChunkType enum: TEXT, IMAGE_DESCRIPTION, TABLE, VIDEO_TRANSCRIPT)
   - RetrievalResult
   - QueryRequest, QueryResponse
   - EvalResult
5. Implement storage abstraction:
   - VectorStoreClient with pgvector + Qdrant backends (common interface)
   - RedisSemanticCache with configurable similarity threshold
   - Neo4jClient for video topic graphs
6. Implement EmbeddingService:
   - Wrap OpenAI text-embedding-3-large
   - Batch API calls
   - Cache embeddings locally
7. Implement DocumentIngestionPipeline:
   - Accept PDFs as input
   - Parse text with PyMuPDF
   - Chunk using ChunkingStrategies: fixed-size, semantic, sliding-window
   - Embed chunks and store in vector DB
8. Implement RAGASRunner:
   - Compute faithfulness, answer_relevancy, context_precision, context_recall
   - Format results as EvalResult Pydantic model
9. Create .env.example with all required secrets (OpenAI key, Anthropic key, etc.)
10. Set up GitHub Actions CI/CD:
    - .github/workflows/ci.yml: lint (ruff), type check (mypy), unit tests (pytest) — runs on every PR and push to main
    - .github/workflows/integration-tests.yml: spin up Docker services, run integration tests — runs on merge to main
    - .github/workflows/eval.yml: RAGAS evaluation on demand (workflow_dispatch), upload results as artifacts
    - Add ruff.toml and mypy.ini configuration files

TECH STACK:
- Language: Python 3.12
- Package manager: uv with workspace support
- LLM: Anthropic Claude API
- Embeddings: OpenAI text-embedding-3-large
- Databases: PostgreSQL 16 + pgvector, Qdrant, Redis, Neo4j
- Validation: Pydantic v2
- Vector retrieval: sqlalchemy (pgvector), qdrant-client
- CI/CD: GitHub Actions (lint, type check, tests, integration tests, RAGAS eval)
- Linting: ruff
- Type checking: mypy

TESTING REQUIREMENTS:
- Write unit tests for VectorStoreClient (mock DB responses)
- Write integration tests using testcontainers for pgvector + Redis
- Test DocumentIngestionPipeline with a small PDF sample
- Test EmbeddingService batching logic
- Test RAGASRunner with mock LLM responses

OBSERVABILITY:
- All operations should log structured data (use Python logging)
- Add LangFuse tracing to vector store operations
- Track embedding costs (OpenAI API)

SUCCESS CRITERIA (Definition of Done):
✓ Docker Compose starts all services without errors
✓ All Pydantic models defined and validated
✓ VectorStoreClient CRUD operations work with pgvector (create/read/update/delete)
✓ EmbeddingService successfully embeds sample text chunks
✓ DocumentIngestionPipeline ingests a sample PDF and stores chunks + embeddings
✓ RAGASRunner computes metrics on sample queries/answers/contexts
✓ 10+ integration tests pass
✓ All imports in subsequent phases will find shared/ models and utilities
✓ README includes "docker-compose up" and setup instructions
✓ GitHub Actions CI workflow passes: ruff lint, mypy type check, pytest unit tests
✓ GitHub Actions integration test workflow runs with Docker services
✓ ruff.toml and mypy.ini configured for the monorepo

COMMIT MESSAGE:
feat(shared-infrastructure): Initialize monorepo, vector DB abstraction, ingestion pipeline, and evaluation harness
```

---

### Phase 1: Fastest RAG Stack Prompt

```
You are implementing Phase 1 (Fastest RAG Stack) of the rag-research-platform.

ARCHITECTURE CONTEXT:
- Depends on Phase 0 (shared infrastructure ready)
- This is the baseline RAG pipeline: embed query → top-k retrieval → generate answer
- Binary quantization is the optimization focus
- Redis semantic cache layer sits between query embedding and vector store

WHAT TO BUILD:
1. Implement NaiveRAGPipeline in ./pipelines/fastest_rag/src/fastest_rag/pipeline.py:
   - Accepts QueryRequest (query, top_k, pipeline="fastest_rag")
   - Embeds query using shared EmbeddingService
   - Retrieves top-k chunks from pgvector using cosine similarity
   - Generates answer using claude-sonnet-4-6 with retrieved chunks as context
   - Returns QueryResponse with answer, sources, latency_ms, token_cost_usd
2. Configure Qdrant collection for binary quantization (BQ):
   - Create separate collection with BQ encoding
   - Migrate 1M+ sample vectors (use synthetic embeddings or public dataset)
3. Configure Qdrant collection for scalar quantization (SQ) for comparison
4. Implement BenchmarkRunner in ./pipelines/fastest_rag/src/fastest_rag/benchmark.py:
   - Measures latency: p50, p95, p99 percentiles
   - Measures throughput: queries per second (QPS)
   - Measures recall@k: how many relevant docs in top-k
   - Compares full precision vs. BQ vs. SQ
   - Outputs benchmark results as JSON
5. Implement Redis semantic cache:
   - Cache key: embedding of query (as vector)
   - On cache hit: return cached answer directly
   - Configurable similarity threshold (e.g., cosine > 0.95)
   - Track cache hit rate and latency reduction
6. Build Streamlit benchmark dashboard:
   - Show latency comparison chart (full precision vs BQ vs SQ)
   - Show cache hit rate over time
   - Live throughput meter
   - Recall@k metrics table
7. Run RAGAS evaluation on BQ vs full-precision to quantify quality trade-off

TECH STACK:
- API: FastAPI in ./api/src/api/routers/query.py
- Orchestration: LangGraph (optional for this simple pipeline)
- LLM: claude-sonnet-4-6 for generation
- Vector store: pgvector + Qdrant
- Cache: Redis with semantic similarity
- Benchmarking: custom BenchmarkRunner (no external framework needed)
- Dashboard: Streamlit

TESTING REQUIREMENTS:
- Unit tests for NaiveRAGPipeline (mock vector store, mock LLM)
- Unit tests for BenchmarkRunner (synthetic latency measurements)
- Unit tests for cache layer (hit/miss logic)
- Integration test: end-to-end query through FastAPI endpoint
- Performance test: ensure p99 latency < 50ms with BQ

SUCCESS CRITERIA:
✓ NaiveRAGPipeline returns answers in < 100ms (end-to-end)
✓ Binary quantization shows 30%+ latency improvement vs. full precision
✓ BQ quality loss < 5% (measured by RAGAS faithfulness drop)
✓ Redis cache hit rate > 40% on repeated queries
✓ Benchmark dashboard visualizes all comparisons clearly
✓ 5+ unit tests pass
✓ 1 integration test passes

COMMIT MESSAGE:
feat(fastest-rag): Implement baseline RAG + binary quantization + semantic caching + benchmark dashboard
```

---

### Phase 2: Multimodal RAG Prompt

```
You are implementing Phase 2 (Multimodal RAG) of the rag-research-platform.

ARCHITECTURE CONTEXT:
- Depends on Phase 1 (Fastest RAG Stack works)
- Extends DocumentIngestionPipeline from shared/ to handle images and tables
- Adds vision model calls to describe visual content
- Adds provenance tracking to map answers back to source pages/sections
- ChunkType enum now includes IMAGE_DESCRIPTION and TABLE

WHAT TO BUILD:
1. Extend DocumentIngestionPipeline in ./shared/src/shared/ingestion/pdf_parser.py:
   - Extract images from PDFs (PyMuPDF)
   - Detect tables in PDFs using PDFPlumber or similar
2. Implement VisionDescriber in ./pipelines/multimodal_rag/src/multimodal_rag/vision_describer.py:
   - Send images to Claude vision (claude-sonnet-4-6)
   - Get text descriptions of what the image shows
   - Send tables to Claude vision for markdown extraction
3. Implement TableExtractor in ./pipelines/multimodal_rag/src/multimodal_rag/table_extractor.py:
   - Detect tables using heuristics or library
   - Extract table content as Markdown
4. Implement MultimodalChunker in ./shared/src/shared/ingestion/chunking.py:
   - Chunks text separately from image descriptions and tables
   - Preserves metadata: source_file, page_number, chunk_type, bounding_box (for images)
5. Implement ProvenanceTracker in ./pipelines/multimodal_rag/src/multimodal_rag/provenance.py:
   - Maps each sentence in generated answer back to source chunk(s)
   - Tracks: which source file, which page, which chunk_id
   - Returns source attribution list with confidence scores
6. Implement MultimodalRAGPipeline in ./pipelines/multimodal_rag/src/multimodal_rag/pipeline.py:
   - Accepts QueryRequest
   - Retrieves across text, image_description, and table chunk types
   - May use separate ranking for each chunk type and fuse results
   - Generates answer with multimodal context
   - Returns QueryResponse + provenance list
7. Build ProvenanceViewer Chainlit component in ./ui/components/provenance_viewer.py:
   - When answer is displayed, highlight source page/section
   - Show which chunk type contributed (text/image/table)
   - Link to PDF page or image
8. Create test dataset:
   - 10 mixed-content PDFs: financial reports, scientific papers with charts
9. Run RAGAS evaluation:
   - Compare text-only vs. multimodal retrieval
   - Show recall improvement (should be ~20-30% better for multimodal)
10. Write standalone README for multimodal_rag/:
    - Architecture diagram showing ingestion flow (PDF → images/tables → vision model → embeddings)
    - Quick start: `docker run` or `docker-compose up`
    - RAGAS comparison table (text-only vs multimodal)
    - Demo GIF showing provenance highlighting
11. Build standalone demo.py (Streamlit):
    - Upload PDF, query, see provenance highlighting with source pages
    - Runs independently without the full platform
12. Write Dockerfile for multimodal_rag/:
    - Independently runnable with `docker run`

TECH STACK:
- PDF parsing: PyMuPDF (fitz) + PDFPlumber for tables
- Vision model: Claude Sonnet for image/table descriptions
- UI component: Chainlit custom element (HTML/SVG based)
- Metadata storage: Extended DocumentChunk Pydantic model
- Ranking: hybrid fusion (BM25 for text + semantic for images)
- Demo: Streamlit standalone app
- CI: GitHub Actions (lint + type check + tests on every PR)

TESTING REQUIREMENTS:
- Unit tests for VisionDescriber (mock Claude API)
- Unit tests for TableExtractor (mock PDF with tables)
- Unit tests for ProvenanceTracker (mock retrieved chunks and answers)
- Integration test: ingest mixed-content PDF, query, verify provenance
- E2E test: submit query via API, check provenance in response

SUCCESS CRITERIA:
✓ VisionDescriber correctly describes sample images
✓ TableExtractor correctly extracts table markdown
✓ MultimodalRAGPipeline retrieves from all chunk types
✓ ProvenanceTracker maps answer sentences to sources
✓ Chainlit UI shows provenance highlighting
✓ RAGAS context_recall improves by >15% vs. text-only
✓ 8+ unit tests pass
✓ 1+ integration tests pass
✓ Standalone README with architecture diagram, RAGAS results table, and demo GIF
✓ Standalone Streamlit demo runs independently (`docker run` or `python demo.py`)
✓ Dockerfile builds and runs without errors
✓ GitHub Actions CI passes (lint + tests) on PR

COMMIT MESSAGE:
feat(multimodal-rag): Add vision model integration, table extraction, provenance tracking, and source attribution UI
```

---

### Phase 3: Corrective RAG (CRAG) Prompt

```
You are implementing Phase 3 (Corrective RAG — CRAG) of the rag-research-platform.

ARCHITECTURE CONTEXT:
- Depends on Phase 1 (Fastest RAG works)
- Uses LangGraph for stateful orchestration
- Adds self-assessment loop: retrieve → grade → (branch on grade)
- Three outcomes: RELEVANT (generate) / AMBIGUOUS (decompose) / IRRELEVANT (web search)
- See section 6 "LangGraph Graph Designs" for CRAG graph structure

WHAT TO BUILD:
1. Implement RelevanceGrader in ./pipelines/corrective_rag/src/corrective_rag/relevance_grader.py:
   - Accepts retrieved document and query
   - Uses claude-haiku-4-5 to grade: RELEVANT / AMBIGUOUS / IRRELEVANT
   - Returns grade + confidence score
2. Implement DocumentDecomposer in ./pipelines/corrective_rag/src/corrective_rag/document_decomposer.py:
   - For AMBIGUOUS docs, extract only the relevant sub-sections
   - Uses Claude to identify relevant parts
   - Returns extracted text
3. Implement QueryRewriter in ./pipelines/corrective_rag/src/corrective_rag/query_rewriter.py:
   - Rewrites query to improve web search quality
   - Example: "What is RAG?" → "Retrieval-Augmented Generation techniques and applications"
4. Implement WebSearcher in ./pipelines/corrective_rag/src/corrective_rag/web_searcher.py:
   - Tavily API integration
   - Searches when document grading returns IRRELEVANT
   - Parses results and deduplicates
5. Implement CRAGGraph in ./pipelines/corrective_rag/src/corrective_rag/graph.py:
   - LangGraph StateGraph with nodes:
     * retrieve (query vector store)
     * grade_documents (for each doc)
     * generate (when relevant)
     * decompose_docs (when ambiguous)
     * rewrite_query (when irrelevant)
     * web_search (when irrelevant)
   - Conditional edges based on grades
   - Full state tracking
6. Add LangFuse tracing to all nodes:
   - Log inputs, outputs, latency, token cost
   - Trace URL returned in QueryResponse
7. Create test dataset:
   - 50 queries where naive RAG hallucinated
   - Golden answers for each query
8. Run RAGAS evaluation:
   - Compare naive RAG vs. CRAG
   - Show hallucination rate drop (target: 25% → 12%)
9. Write standalone README for corrective_rag/:
   - Architecture diagram (CRAG graph flowchart)
   - Quick start: `docker run` or `docker-compose up`
   - RAGAS comparison table (naive RAG vs CRAG)
   - Demo GIF showing the decision path in action
10. Build standalone demo.py (Streamlit):
    - Enter query, see CRAG decision path live (RELEVANT/AMBIGUOUS/IRRELEVANT)
    - Runs independently without the full platform
11. Write Dockerfile for corrective_rag/:
    - Independently runnable with `docker run`
    - Includes docker-compose.override.yml for local dev

TECH STACK:
- Orchestration: LangGraph with StateGraph
- LLM (generation): claude-sonnet-4-6
- LLM (grading): claude-haiku-4-5 (cost optimization)
- Web search: Tavily API
- Observability: LangFuse
- Demo: Streamlit standalone app
- CI: GitHub Actions (lint + type check + tests on every PR)

TESTING REQUIREMENTS:
- Unit tests for RelevanceGrader (mock Claude API)
- Unit tests for DocumentDecomposer, QueryRewriter
- Unit tests for WebSearcher (mock Tavily API)
- Integration test: full CRAG graph execution with all branches
  * Test RELEVANT path (generate directly)
  * Test AMBIGUOUS path (decompose then generate)
  * Test IRRELEVANT path (web search then generate)
- E2E test: submit hallucination-prone query, verify CRAG improves answer

SUCCESS CRITERIA:
✓ CRAG graph compiles and executes without errors
✓ All three branches (RELEVANT/AMBIGUOUS/IRRELEVANT) tested
✓ RelevanceGrader correctly classifies documents
✓ WebSearcher retrieves relevant results on fallback
✓ RAGAS faithfulness improves from ~0.70 to ~0.82 vs. naive RAG
✓ Hallucination rate drops from ~25% to ~12%
✓ LangFuse trace shows all node executions
✓ 10+ unit tests pass
✓ 3 integration tests pass (one per branch)
✓ Standalone README with architecture diagram, RAGAS results table, and demo GIF
✓ Standalone Streamlit demo runs independently (`docker run` or `python demo.py`)
✓ Dockerfile builds and runs without errors
✓ GitHub Actions CI passes (lint + tests) on PR

COMMIT MESSAGE:
feat(corrective-rag): Implement CRAG with relevance grading, document decomposition, and web search fallback
```

---

### Phase 4: Self-RAG with LangGraph Prompt

```
You are implementing Phase 4 (Self-RAG with LangGraph) of the rag-research-platform.

ARCHITECTURE CONTEXT:
- Depends on Phase 3 (CRAG works)
- Extends CRAG graph with three additional decision nodes
- Five decision points total: retrieve-or-not, relevance-grade, hallucination-grade, answer-grade, + adaptive retry
- See section 6 "LangGraph Graph Designs" for Self-RAG graph structure
- Goal: more autonomous, fault-tolerant RAG with recovery loops

WHAT TO BUILD:
1. Implement RetrieveOrNot in ./pipelines/self_rag/src/self_rag/retrieve_or_not.py:
   - Uses claude-haiku-4-5 to decide: should we retrieve documents for this query?
   - Examples: "What is 2+2?" (no retrieve needed) vs "What is RAG?" (retrieve needed)
   - Returns boolean decision
2. Implement HallucinationGrader in ./pipelines/self_rag/src/self_rag/hallucination_grader.py:
   - Checks if generated answer is grounded in retrieved documents
   - Uses claude-haiku-4-5
   - Returns GROUNDED / NOT_GROUNDED with confidence
3. Implement AnswerGrader in ./pipelines/self_rag/src/self_rag/answer_grader.py:
   - Checks if answer addresses the original question
   - Uses claude-haiku-4-5
   - Returns ADDRESSES_QUESTION / DOES_NOT_ADDRESS with confidence
4. Implement HyDEQueryExpander in ./pipelines/self_rag/src/self_rag/hyde.py:
   - Hypothetical Document Embeddings
   - LLM generates a hypothetical document that would answer the query
   - Embed the hypothetical document
   - Use that embedding for retrieval (often better than original query)
5. Implement SelfRAGGraph in ./pipelines/self_rag/src/self_rag/graph.py:
   - Extend CRAG graph with new nodes:
     * retrieve_or_not (entry point)
     * retrieve (conditional: only if needed)
     * grade_relevance
     * generate
     * grade_grounding (if NOT grounded, branch)
     * hyde_expand (when grounding fails, expand query)
     * retrieve_again (retry with HyDE query, max 2x)
     * grade_answer (final quality check)
   - State tracking: query, documents, answer, grades, attempts
   - Conditional edges based on all decisions
6. Add graph execution trace visualization in Chainlit:
   - Show which nodes fired, in what order
   - Show decision values (e.g., "RetrieveOrNot: YES", "HallucinationGrader: NOT_GROUNDED")
   - Animated flow visualization
7. Test all decision paths:
   - Simple knowledge question (retrieve_or_not: NO)
   - Grounding failure + successful HyDE retry
   - Answer quality failure + rewrite + retry
8. Compare Self-RAG vs CRAG vs naive RAG on same 50-query test set

TECH STACK:
- Orchestration: LangGraph with conditional edges + state persistence
- LLM (generation): claude-sonnet-4-6
- LLM (all grading): claude-haiku-4-5
- Query expansion: HyDE (LLM-based)
- Visualization: Chainlit custom component (flow diagram)

TESTING REQUIREMENTS:
- Unit tests for RetrieveOrNot, HallucinationGrader, AnswerGrader (mock Claude)
- Unit tests for HyDEQueryExpander
- Integration test: execute each decision branch separately
  * retrieve_or_not: YES path
  * retrieve_or_not: NO path
  * grounding_failure: YES path (HyDE retry)
  * answer_quality_failure: YES path (rewrite + retry)
- Integration test: full Self-RAG graph with all paths exercised
- E2E test: submit query, verify graph trace visualization

SUCCESS CRITERIA:
✓ Self-RAG graph compiles and executes
✓ All decision nodes return correct judgments
✓ HyDE query expansion produces better embeddings
✓ Graph handles all branch conditions + adaptive retry
✓ Trace visualization shows clear flow diagram
✓ RAGAS faithfulness improves to ~0.88 (vs CRAG's 0.82)
✓ Hallucination rate drops to ~7% (vs CRAG's 12%)
✓ Answer relevancy improves to ~0.82 (vs CRAG's 0.78)
✓ 12+ unit tests pass
✓ 5+ integration tests pass

COMMIT MESSAGE:
feat(self-rag): Implement Self-RAG with adaptive decision nodes, HyDE expansion, hallucination detection, and graph trace visualization
```

---

### Phase 5: MCP-powered RAG over Videos Prompt

```
You are implementing Phase 5 (MCP-powered RAG over Videos) of the rag-research-platform.

ARCHITECTURE CONTEXT:
- Depends on Phase 2 (multimodal ingestion infrastructure)
- Most unique project: video indexing with timestamp-level retrieval
- MCP (Model Context Protocol) exposes video search as tools
- Two retrieval modalities: text (transcripts) + visual (CLIP frames)
- Neo4j knowledge graph enables multi-hop queries: "find segments about X in videos tagged Y"

WHAT TO BUILD:
1. Implement VideoIndexer in ./pipelines/video_rag/src/video_rag/video_indexer.py:
   - Accept video file or YouTube URL
   - Use Whisper to transcribe audio
   - Segment transcript by sentences (each sentence ≈ a few seconds)
   - Return text chunks with (start_ts, end_ts, transcript_text)
2. Implement SceneDetector in ./pipelines/video_rag/src/video_rag/scene_detector.py:
   - Extract keyframes at scene changes
   - Use frame difference (optical flow) or shot boundary detection
   - Generate one keyframe per scene
3. Implement CLIPEmbedder in ./pipelines/video_rag/src/video_rag/clip_embedder.py:
   - For each keyframe, compute CLIP embedding
   - Store frame_embedding alongside scene metadata
4. Build hybrid video index:
   - Text index: transcript sentence embeddings (OpenAI)
   - Visual index: keyframe embeddings (CLIP)
   - Both indexed in pgvector with separate columns
5. Implement TimestampChunker in ./pipelines/video_rag/src/video_rag/timestamp_chunker.py:
   - DocumentChunk extension with start_ts, end_ts
   - chunk_type = VIDEO_TRANSCRIPT
   - Stores transcript text + frame embedding
6. Implement SegmentRetriever in ./pipelines/video_rag/src/video_rag/segment_retriever.py:
   - Dual retrieval: text similarity (query embedding vs transcript) + visual similarity (query embedding vs frame CLIP)
   - Fused ranking: combine text score + visual score (e.g., 0.6 * text + 0.4 * visual)
   - Returns top-k segments with timestamps
7. Build Neo4j topic graph in ./pipelines/video_rag/src/video_rag/knowledge_graph.py:
   - Node types: Video, Topic, Segment
   - Relationships: video CONTAINS segment, video HAS_TOPIC topic, segment MENTIONS topic
   - Enable multi-hop queries: "find segments about machine_learning in videos tagged ai"
8. Implement FastMCP server in ./pipelines/video_rag/src/video_rag/mcp_server.py:
   - Export tools: search_video, get_segment, list_videos, get_transcript
   - Uses Claude-native Model Context Protocol
   - Claude can call these tools directly
9. Build Chainlit VideoPlayer component in ./ui/components/video_player.py:
   - Display retrieved video segment with HTML5 video player
   - Auto-seek to start_ts
   - Show transcript alongside video
   - Hover over transcript text to see CLIP similarity heatmap
10. Build multi-video Q&A mode in ./ui/pages/video_search.py:
    - Query spans entire video library
    - Returns segments from multiple videos
    - Group results by video
11. Test with real videos:
    - Index 5 YouTube lecture videos (use yt-dlp for download)
    - Run sample queries: timestamp questions, concept questions, cross-video questions

TECH STACK:
- Video processing: yt-dlp (download), ffmpeg (via Python subprocess)
- ASR: OpenAI Whisper
- Vision: CLIP (open-weight)
- Scene detection: frame difference + optical flow (OpenCV)
- MCP server: FastMCP (Anthropic's Python library)
- Knowledge graph: Neo4j
- Vector retrieval: pgvector (text) + CLIP embeddings
- UI: Chainlit + HTML5 video player

TESTING REQUIREMENTS:
- Unit tests for VideoIndexer (mock Whisper API)
- Unit tests for SceneDetector (synthetic video frames)
- Unit tests for CLIPEmbedder (mock CLIP model)
- Unit tests for SegmentRetriever (fused ranking logic)
- Integration test: ingest sample video, retrieve segments, verify timestamps
- Integration test: MCP server responds to tool calls
- E2E test: query via Claude + MCP tools, video player shows segment

SUCCESS CRITERIA:
✓ VideoIndexer successfully transcribes and segments sample video
✓ SceneDetector extracts keyframes at scene changes
✓ CLIPEmbedder computes visual embeddings
✓ SegmentRetriever retrieves with correct timestamps (within 1 second)
✓ MCP server exposes all 4 tools without errors
✓ Chainlit VideoPlayer displays video with auto-seek to timestamp
✓ Multi-video queries return results from multiple videos
✓ Neo4j graph supports topic-based multi-hop queries
✓ 10+ unit tests pass
✓ 3+ integration tests pass

COMMIT MESSAGE:
feat(video-rag): Implement MCP video search with Whisper transcription, CLIP embeddings, dual retrieval, Neo4j knowledge graph, and video player UI
```

---

### Phase 6: Integration & Polish Prompt

```
You are implementing Phase 6 (Integration & Polish) of the rag-research-platform.

ARCHITECTURE CONTEXT:
- All 5 pipeline implementations are complete (Phase 0-5 done)
- This phase unifies them under one UI with A/B comparison and metrics
- Portfolio readiness: comprehensive README, demo video, deployment docs
- Single entry point: Chainlit UI routes queries to selected pipeline

WHAT TO BUILD:
1. Build PipelineSelector UI component in ./ui/components/pipeline_selector.py:
   - Dropdown: Naive RAG | Fastest RAG | Multimodal RAG | CRAG | Self-RAG | Video RAG
   - Shows status of each pipeline (ready / loading / error)
   - Stores selected pipeline in Chainlit session
2. Build A/B Comparison mode in ./ui/pages/compare.py:
   - User selects two pipelines
   - Enters query
   - Runs query through both pipelines in parallel
   - Displays results side-by-side:
     * Left: Pipeline A answer + sources + cost + latency
     * Right: Pipeline B answer + sources + cost + latency
     * Bottom: Metrics comparison (RAGAS faithfulness, answer_relevancy, hallucination rate if available)
3. Build Metrics Dashboard in ./ui/pages/metrics_dashboard.py:
   - Metrics per pipeline (updated in real-time via API polling):
     * RAGAS scores: faithfulness, answer_relevancy, context_precision, context_recall
     * Performance: p50/p95/p99 latency, QPS, cache hit rate (for Fastest RAG)
     * Cost: average USD per query
     * Benchmark results: BQ vs SQ vs full-precision comparison (Fastest RAG specific)
   - Line charts over time
   - Export metrics as CSV
4. Create FastAPI pipeline router in ./api/src/api/main.py:
   - Single POST /api/query endpoint accepts QueryRequest with pipeline field
   - Router logic: routes to correct pipeline implementation based on pipeline name
   - Unified QueryResponse returned
5. Implement cost tracking middleware in ./api/src/api/middleware/observability.py:
   - Track tokens used per LLM call
   - Calculate USD cost based on model rates (Haiku/Sonnet pricing)
   - Log to LangFuse with custom "total_cost_usd" field
   - Include in QueryResponse
6. Write end-to-end tests for each pipeline via API:
   - Test POST /api/query with pipeline="fastest_rag" → verify response
   - Test POST /api/query with pipeline="crag" → verify web search fallback
   - Test POST /api/query with pipeline="video_rag" → verify timestamp in response
   - All should return consistent QueryResponse schema
7. Write comprehensive README:
   - Architecture diagram (ASCII or SVG showing all 5 pipelines)
   - Quick start: "docker-compose up", "pip install -e .", "python -m chainlit run ui/app.py"
   - Explanation of each pipeline: when to use, what it optimizes for
   - Benchmark results: comparison table of all 5 pipelines on 50-query test set
   - Metrics visualization: screenshots of A/B comparison and dashboard
   - API documentation: curl examples for /api/query endpoint
   - Cost estimate: example query costs for each pipeline
   - Future work: mention Nomic embeddings upgrade, fine-tuning, etc.
8. Record 5-minute demo video:
   - Show Chainlit UI startup
   - Query 1: Naive RAG (fast but less accurate)
   - Query 2: CRAG (web search fallback visible)
   - Query 3: Self-RAG (graph trace shows decision nodes)
   - Query 4: Multimodal RAG (provenance highlighting source page)
   - Query 5: Video RAG (video player with segment playback)
   - A/B comparison: side-by-side of Fastest RAG vs Self-RAG
   - Final: metrics dashboard showing all comparisons
   - Narration explaining trade-offs and use cases
9. Deployment documentation:
   - Docker build + push to container registry
   - Fly.io or Railway deployment guide
   - Environment variable setup
   - Scaling notes (horizontal scaling with API, Chainlit as stateless frontend)
10. Code quality & final checks:
    - Run all tests (unit + integration + E2E)
    - Code review: check for docstrings, type hints (optional but encouraged)
    - Performance profiling: ensure no pipeline exceeds 100ms p99 latency
    - Cost audit: estimate monthly spend at scale (e.g., 1000 queries/day)

TECH STACK:
- Frontend: Chainlit UI with custom components
- API: FastAPI
- Middleware: observability (LangFuse), cost tracking
- Testing: pytest + respx (HTTP mocking)
- Deployment: Docker, Fly.io / Railway

TESTING REQUIREMENTS:
- E2E tests for all 5 pipelines via API (5 tests)
- Integration test for A/B comparison (2 queries, 2 pipelines)
- Integration test for metrics dashboard (fetch and render metrics)
- Performance test: all pipelines p99 < 100ms
- Cost audit test: validate pricing math

SUCCESS CRITERIA:
✓ Chainlit UI launches without errors
✓ Pipeline selector shows all 5 pipelines ready
✓ A/B comparison runs two pipelines and displays results side-by-side
✓ Metrics dashboard fetches and displays metrics correctly
✓ FastAPI /api/query endpoint works with all pipeline names
✓ Cost tracking logs correct USD amounts to LangFuse
✓ README is comprehensive and clear
✓ Demo video is 4-6 minutes, shows all pipelines
✓ All 5 E2E tests pass
✓ Deployment guide is complete
✓ Code is production-ready (tested, documented, no warnings)

COMMIT MESSAGE:
feat(platform-integration): Unify 5 RAG pipelines under shared UI, build A/B comparison + metrics dashboard, add cost tracking, write comprehensive README and demo
```

---

## 6. LangGraph Graph Designs

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

## 7. Data Models (Pydantic Schemas)

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
    pipeline: str = "self_rag"  # fastest_rag | multimodal_rag | crag | self_rag | video_rag
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

## 8. Evaluation Strategy (RAGAS)

### Test Dataset Strategy

Each pipeline is evaluated against the same 50-query golden test set. This enables apples-to-apples comparison across all 5 pipelines.

**Test set composition:**
- 10 factual retrieval queries (answer is a specific fact in the docs)
- 10 multi-hop reasoning queries (answer requires combining 2+ chunks)
- 10 table/chart queries (only answerable from structured data — tests multimodal retrieval)
- 10 adversarial queries (answer is NOT in the docs — tests hallucination resistance)
- 10 video-specific queries (timestamp questions — for Video RAG pipeline)

**RAGAS metrics tracked per pipeline:**

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Faithfulness | Answer is supported by retrieved context | >0.85 |
| Answer Relevancy | Answer addresses the question | >0.80 |
| Context Precision | Retrieved chunks are on-topic | >0.75 |
| Context Recall | Correct chunks were retrieved | >0.70 |
| Hallucination Rate | % answers with unsupported claims | <10% |

**Expected results hypothesis (to validate empirically):**

| Pipeline | Faithfulness | Answer Relevancy | Context Recall | Hallucination Rate |
|----------|-------------|-----------------|----------------|--------------------|
| Naive RAG | ~0.70 | ~0.72 | ~0.60 | ~25% |
| Fastest RAG (BQ) | ~0.68 | ~0.72 | ~0.55 | ~27% |
| Multimodal RAG | ~0.75 | ~0.80 | ~0.80* | ~20% |
| CRAG | ~0.82 | ~0.78 | ~0.65 | ~12% |
| Self-RAG | ~0.88 | ~0.82 | ~0.70 | ~7% |

*Multimodal excels at context recall because it retrieves from images and tables, not just text.

---

## 9. API Design (FastAPI Endpoints)

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

**Unified query request/response (all pipelines):**

```python
# POST /api/query
# Request: QueryRequest (see Data Models above)
# Response: QueryResponse (see Data Models above)
```

---

## 10. Testing Strategy

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
- RAGAS for automated quality evaluation (not in unit test suite — runs separately)

---

## 11. Observability & Monitoring Setup

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

**Custom metrics logged to LangFuse:**
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

A weekly cost report is auto-generated from LangFuse data.

---

## 12. Build Timeline Estimate

| Phase | Content | Duration |
|-------|---------|----------|
| Phase 0 | Shared infrastructure | 2 weeks |
| Phase 1 | Fastest RAG Stack | 2 weeks |
| Phase 2 | Multimodal RAG | 3 weeks |
| Phase 3 | Corrective RAG (CRAG) | 2 weeks |
| Phase 4 | Self-RAG with LangGraph | 2 weeks |
| Phase 5 | MCP Video RAG | 4 weeks |
| Phase 6 | Integration & Polish | 2 weeks |
| **Total** | | **17 weeks** |

### Milestone Checkpoints

| Milestone | Target Week | Definition of Done |
|-----------|------------|-------------------|
| M0: Infrastructure ready | Week 2 | Docker Compose running, ingestion pipeline tested, RAGAS eval runs |
| M1: Fastest RAG live | Week 4 | Benchmark dashboard shows BQ vs full-precision latency comparison |
| M2: Multimodal RAG live | Week 7 | Mixed-content PDF queries answered with provenance highlighted |
| M3: CRAG live | Week 9 | RAGAS shows hallucination rate drop vs naive RAG |
| M4: Self-RAG live | Week 11 | All three decision nodes exercised, graph trace visible in UI |
| M5: Video RAG live | Week 15 | Video query returns timestamped segment with playback |
| M6: Platform complete | Week 17 | A/B comparison, metrics dashboard, README, demo video published |

---

## Appendix A: Key Design Decisions

### A1. Why pgvector as primary, Qdrant as secondary?

pgvector lives alongside the application database (SQLAlchemy), reducing infrastructure complexity. Qdrant is added specifically for the binary quantization benchmark because it has mature BQ support and a dedicated benchmarking API. For production, either would work.

### A2. Why Chainlit over Next.js for the UI?

Chainlit gives a production-quality chat UI out-of-the-box with streaming, file upload, and custom element support (for the provenance viewer and video player components). Next.js would require building all of this from scratch. Chainlit is the faster path to a polished demo. The FastAPI backend is fully decoupled, so migrating to Next.js later requires only a new frontend — no backend changes.

### A3. Why not build each project as a separate repo?

The shared infrastructure (embedding service, vector store, ingestion pipeline, RAGAS harness) would need to be duplicated or published as a private package. A monorepo with uv workspaces gives the same isolation (each pipeline is an independent Python package) with shared dependencies and a single CI pipeline.

### A4. Model selection for graders in CRAG/Self-RAG

Graders (relevance, hallucination, answer quality) use `claude-haiku-4-5` — they need a yes/no judgment, not deep reasoning. This reduces cost by ~10x versus using Sonnet for grading. The final generation step uses `claude-sonnet-4-6`. This mirrors production cost-optimization patterns.

### A5. CLIP vs. other visual embedding models for Video RAG

CLIP (ViT-B/32 or ViT-L/14) is chosen for visual embeddings because it is open-weight, well-supported, and produces embeddings in the same semantic space as text — enabling cross-modal retrieval where a text query can retrieve visually similar frames. A future upgrade path is `nomic-embed-vision` for higher recall.
