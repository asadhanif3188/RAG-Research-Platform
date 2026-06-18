# RAG Research Platform — Demo Guide

**Author:** Asad Hanif  
**Platform:** RAG Research Platform (5 unified pipelines)  
**Estimated Demo Time:** 15-20 minutes (full walkthrough) or 5-7 minutes (highlights only)

---

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Pre-Demo Setup](#1-pre-demo-setup)
3. [Demo Flow Overview](#2-demo-flow-overview)
4. [Step 1: Infrastructure Startup](#step-1-infrastructure-startup)
5. [Step 2: API Server Launch](#step-2-api-server-launch)
6. [Step 3: Fastest RAG — Baseline Pipeline](#step-3-fastest-rag--baseline-pipeline)
7. [Step 4: Multimodal RAG — Vision + Tables](#step-4-multimodal-rag--vision--tables)
8. [Step 5: Corrective RAG (CRAG) — Self-Assessment](#step-5-corrective-rag-crag--self-assessment)
9. [Step 6: Self-RAG — Agentic Decision Graph](#step-6-self-rag--agentic-decision-graph)
10. [Step 7: Video RAG — MCP + Timestamp Retrieval](#step-7-video-rag--mcp--timestamp-retrieval)
11. [Step 8: A/B Pipeline Comparison](#step-8-ab-pipeline-comparison)
12. [Step 9: Metrics Dashboard](#step-9-metrics-dashboard)
13. [Step 10: Code & Architecture Walkthrough](#step-10-code--architecture-walkthrough)
14. [Talking Points & Q&A Prep](#talking-points--qa-prep)
15. [Troubleshooting](#troubleshooting)

---

## Deployment Options

The platform supports three deployment models. Choose based on your demo environment:

### Option 1: Infrastructure in Docker + Python Services (Recommended for Demo)

**Best for:** Live demos, interactive development, quick iteration

```bash
# Terminal 1: Start infrastructure (PostgreSQL, Qdrant, Redis, Neo4j, LangFuse)
docker-compose up -d

# Terminal 2: Start FastAPI backend
uv run uvicorn api.src.api.main:app --reload --port 8000

# Terminal 3: Start Chainlit UI
uv run chainlit run ui/app.py -w

# Terminal 4 (Optional): Run individual pipeline Streamlit demos
cd pipelines/multimodal_rag && uv run streamlit run demo.py
```

**Advantages:**
- Infrastructure isolated in Docker (clean environment)
- Python services reload on code changes (fast iteration)
- Easy to debug with terminal output
- Can run multiple pipeline demos in separate terminals

### Option 2: Full Docker Compose (All Services in Containers)

**Best for:** Production deployment, cloud environments, complete isolation

**Step 1:** Extend `docker-compose.yml` with pipeline services:

```yaml
services:
  # ... existing infrastructure services ...
  
  fastest_rag:
    build:
      context: .
      dockerfile: pipelines/fastest_rag/Dockerfile
    ports:
      - "8501:8501"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - rag_network

  multimodal_rag:
    build:
      context: .
      dockerfile: pipelines/multimodal_rag/Dockerfile
    ports:
      - "8502:8501"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - rag_network

  corrective_rag:
    build:
      context: .
      dockerfile: pipelines/corrective_rag/Dockerfile
    ports:
      - "8503:8501"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      TAVILY_API_KEY: ${TAVILY_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - rag_network

  self_rag:
    build:
      context: .
      dockerfile: pipelines/self_rag/Dockerfile
    ports:
      - "8504:8501"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - rag_network

  video_rag:
    build:
      context: .
      dockerfile: pipelines/video_rag/Dockerfile
    ports:
      - "8505:8501"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      neo4j:
        condition: service_healthy
    networks:
      - rag_network

  api:
    build:
      context: .
      dockerfile: api/Dockerfile  # Create if not exists
    ports:
      - "8000:8000"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - rag_network

  ui:
    build:
      context: .
      dockerfile: ui/Dockerfile  # Create if not exists
    ports:
      - "8080:8000"
    environment:
      API_BASE_URL: http://api:8000
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    depends_on:
      - api
    networks:
      - rag_network
```

**Step 2:** Run everything:

```bash
docker-compose up
```

**Advantages:**
- Single command to start entire system
- Reproducible across machines
- Easy to scale horizontally
- Cleaner for CI/CD pipelines

### Option 3: Individual Docker Images (Manual Orchestration)

**Best for:** Kubernetes deployment, microservice architecture, selective scaling

```bash
# Build all images
docker build -t fastest-rag -f pipelines/fastest_rag/Dockerfile .
docker build -t multimodal-rag -f pipelines/multimodal_rag/Dockerfile .
docker build -t corrective-rag -f pipelines/corrective_rag/Dockerfile .
docker build -t self-rag -f pipelines/self_rag/Dockerfile .
docker build -t video-rag -f pipelines/video_rag/Dockerfile .
docker build -t rag-api -f api/Dockerfile .
docker build -t rag-ui -f ui/Dockerfile .

# Start infrastructure
docker-compose up -d

# Run pipelines individually (each on a different port)
docker run --network rag_network -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-... \
  -e OPENAI_API_KEY=sk-... \
  fastest-rag

docker run --network rag_network -p 8502:8501 \
  -e ANTHROPIC_API_KEY=sk-... \
  -e OPENAI_API_KEY=sk-... \
  multimodal-rag

# ... repeat for other pipelines on ports 8503-8505

docker run --network rag_network -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-... \
  rag-api

docker run --network rag_network -p 8080:8000 \
  -e API_BASE_URL=http://host.docker.internal:8000 \
  rag-ui
```

**Advantages:**
- Fine-grained control over each service
- Easy to restart individual components
- Ideal for Kubernetes + Helm charts
- Better resource isolation

---

## Recommended for Demo: Option 1

For the live demo in this guide, we recommend **Option 1** (Docker infrastructure + Python services). It provides:
- **Fast iteration:** Code reloads without rebuilding Docker images
- **Clear logs:** Each service's output in its own terminal
- **Easy debugging:** Breakpoints and print statements work directly
- **Clean environment:** Docker handles infrastructure deterministically

---

## 1. Pre-Demo Setup

Complete these steps **before** the demo starts.

### 1.1 Environment Setup

```bash
# Clone the repository
git clone <repo-url> && cd rag-research-platform

# Copy environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...
```

### 1.2 Install Dependencies

```bash
# Install uv (if not installed)
pip install uv

# Install all workspace packages
uv sync --all-packages
```

### 1.3 Start Infrastructure Services

```bash
# Start PostgreSQL (pgvector), Qdrant, Redis, Neo4j, LangFuse
docker-compose up -d

# Wait for health checks to pass (~30 seconds)
docker-compose ps   # all services should show "healthy"
```

### 1.4 Initialize Database

```bash
# pgvector extension and schema are auto-initialized via infra/init-pgvector.sql
# Verify:
docker exec -it rag-postgres psql -U rag_user -d rag_platform -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

### 1.5 Pre-Ingest Sample Data (Recommended)

```bash
# Ingest a sample PDF for text-based pipelines
curl -X POST http://localhost:8000/api/ingest/document \
  -F "file=@samples/sample-research-paper.pdf"

# For Video RAG, index a sample video
curl -X POST http://localhost:8000/api/ingest/video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=EXAMPLE", "title": "AI Lecture"}'
```

### 1.6 Pre-Demo Checklist

- [ ] All Docker services running and healthy
- [ ] `.env` file has valid API keys (Anthropic, OpenAI, Tavily)
- [ ] Dependencies installed (`uv sync --all-packages`)
- [ ] Sample documents ingested
- [ ] Terminal windows ready (API server, UI, curl commands)
- [ ] Browser open for LangFuse dashboard (http://localhost:3000)

---

## 2. Demo Flow Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        DEMO FLOW                                 │
│                                                                  │
│  Start → Infrastructure → API → Fastest RAG → Multimodal RAG   │
│                                                                  │
│  → CRAG (web search fallback) → Self-RAG (decision graph)       │
│                                                                  │
│  → Video RAG (timestamps) → A/B Comparison → Metrics Dashboard  │
│                                                                  │
│  → Code Walkthrough → Q&A                                       │
└──────────────────────────────────────────────────────────────────┘
```

**Key narrative:** Each pipeline solves a different RAG limitation. The platform lets you compare them side-by-side to pick the right strategy for your use case.

---

## Step 1: Infrastructure Startup

**Time:** 1-2 minutes  
**Goal:** Show the system architecture and services

### What to Show

1. **docker-compose.yml** — Open the file and highlight the 7 services:
   - PostgreSQL 16 + pgvector (vector storage)
   - Qdrant (binary quantization benchmarks)
   - Redis (semantic caching)
   - Neo4j (video knowledge graphs)
   - LangFuse (LLM observability)

2. **Verify services are running:**

```bash
docker-compose ps
```

Expected output: All services `Up (healthy)`

3. **Show LangFuse dashboard** (optional):
   - Open http://localhost:3000
   - Show empty traces panel (will populate during demo)

### Talking Point

> "This platform runs 7 infrastructure services. Every pipeline shares the same vector store, embedding service, and evaluation harness — but each implements a different retrieval strategy."

---

## Step 2: API Server Launch

**Time:** 1 minute  
**Goal:** Start the FastAPI server and show endpoints

### Start the Server

```bash
uv run uvicorn api.src.api.main:app --reload --port 8000
```

### Show Available Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List available pipelines (if endpoint exists)
curl http://localhost:8000/api/pipelines
```

### Show API Documentation

- Open http://localhost:8000/docs (Swagger UI)
- Highlight key endpoints:
  - `POST /query` — unified query endpoint for all pipelines
  - `GET /metrics/summary` — per-pipeline metrics
  - `POST /benchmark/run` — performance benchmarking

### Talking Point

> "One unified API serves all 5 pipelines. The `pipeline` field in the request determines which RAG strategy handles the query. Same input schema, same output schema — different retrieval logic."

---

## Step 3: Fastest RAG — Baseline Pipeline

**Time:** 2-3 minutes  
**Goal:** Show the baseline naive RAG and caching

### 3.1 Run a Basic Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is retrieval-augmented generation?",
    "pipeline": "fastest_rag",
    "top_k": 5,
    "use_cache": true
  }'
```

**Point out in the response:**
- `answer` — generated answer from Claude
- `sources` — retrieved chunks with scores
- `latency_ms` — end-to-end latency
- `token_cost_usd` — cost of this query

### 3.2 Demonstrate Caching

```bash
# Run the SAME query again
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is retrieval-augmented generation?",
    "pipeline": "fastest_rag",
    "top_k": 5,
    "use_cache": true
  }'
```

**Point out:**
- `latency_ms` should be significantly lower (cache hit)
- `token_cost_usd` should be $0.00 (no LLM call)

### 3.3 Show Benchmark Results (Optional)

```bash
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"pipeline": "fastest_rag", "num_queries": 20}'
```

### Talking Point

> "Fastest RAG is the baseline — embed, retrieve, generate. Redis semantic caching means repeated or similar queries skip the LLM entirely. Binary quantization in Qdrant reduces vector search latency by 30%+ with less than 5% quality loss."

---

## Step 4: Multimodal RAG — Vision + Tables

**Time:** 2-3 minutes  
**Goal:** Show retrieval across text, images, and tables

### 4.1 Run the Standalone Demo (Streamlit)

```bash
cd pipelines/multimodal_rag
uv run streamlit run demo.py
```

**In the Streamlit UI:**
1. Upload a mixed-content PDF (financial report or research paper with charts)
2. Enter a query about a chart or table in the PDF
3. Show the answer with **provenance highlighting** — which page, which chunk type (text/image/table) contributed

### 4.2 Or Use the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the revenue chart show for Q3?",
    "pipeline": "multimodal_rag",
    "top_k": 5
  }'
```

**Point out in the response:**
- `sources` array includes chunks with different `chunk_type` values: `text`, `image_description`, `table`
- Provenance metadata shows `page_number` and `source_file`

### Talking Point

> "Most RAG systems only index text. Multimodal RAG also indexes images and tables by sending them through Claude's vision model to generate text descriptions. The provenance tracker maps every sentence in the answer back to its source — you can see exactly which page and which chart contributed."

---

## Step 5: Corrective RAG (CRAG) — Self-Assessment

**Time:** 2-3 minutes  
**Goal:** Show the 3-way branching logic (RELEVANT / AMBIGUOUS / IRRELEVANT)

### 5.1 Run the Standalone Demo

```bash
cd pipelines/corrective_rag
uv run streamlit run demo.py
```

### 5.2 Demo Query 1 — RELEVANT Path

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the transformer attention mechanism",
    "pipeline": "crag",
    "top_k": 5
  }'
```

**Show:** Retrieved docs are relevant → generates answer directly

### 5.3 Demo Query 2 — IRRELEVANT Path (Web Search Fallback)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in quantum computing in 2026?",
    "pipeline": "crag",
    "top_k": 5
  }'
```

**Show:** Retrieved docs are irrelevant → query is rewritten → Tavily web search → generates answer from web results

### 5.4 Show LangFuse Trace

- Open http://localhost:3000 (LangFuse)
- Find the latest trace
- Show the node execution path: `retrieve → grade_documents → rewrite_query → web_search → generate`
- Show token costs per node

### Talking Point

> "CRAG adds a self-assessment loop. After retrieving documents, Claude Haiku grades each one as RELEVANT, AMBIGUOUS, or IRRELEVANT. If the docs don't answer the question, it rewrites the query and falls back to web search via Tavily. This dropped hallucination rate from 25% to 12% in our RAGAS evaluation."

---

## Step 6: Self-RAG — Agentic Decision Graph

**Time:** 2-3 minutes  
**Goal:** Show the full decision graph with 5 checkpoints

### 6.1 Run the Standalone Demo

```bash
cd pipelines/self_rag
uv run streamlit run demo.py
```

### 6.2 Demo Query — Full Graph Execution

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare BERT and GPT architectures for text classification",
    "pipeline": "self_rag",
    "top_k": 5
  }'
```

**Show in the response:**
- The graph trace (if returned) showing which nodes fired
- Decision points: retrieve_or_not → retrieve → grade_relevance → generate → grade_grounding → grade_answer

### 6.3 Demo Query — No Retrieval Needed

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is 2 + 2?",
    "pipeline": "self_rag",
    "top_k": 5
  }'
```

**Show:** `retrieve_or_not` decides NO retrieval needed → generates directly (saves latency and cost)

### 6.4 Demo Query — Hallucination Recovery

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What specific results did the authors report in Table 3?",
    "pipeline": "self_rag",
    "top_k": 5
  }'
```

**Show:** If grounding check fails → HyDE expansion generates a hypothetical document → re-retrieves with better embedding → retries (max 2x)

### Talking Point

> "Self-RAG extends CRAG with five decision checkpoints. It first decides whether to retrieve at all, then grades relevance, checks for hallucinations, and verifies the answer addresses the question. If any check fails, it retries with HyDE query expansion. This achieved 0.88 faithfulness — a 26% improvement over naive RAG."

---

## Step 7: Video RAG — MCP + Timestamp Retrieval

**Time:** 2-3 minutes  
**Goal:** Show video indexing, dual retrieval (text + visual), and MCP tools

### 7.1 Show Video Indexing Pipeline

```bash
# Index a video (if not pre-ingested)
curl -X POST http://localhost:8000/api/ingest/video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=EXAMPLE", "title": "AI Lecture"}'
```

**Explain the pipeline:**
1. Whisper transcribes audio → timestamped sentences
2. OpenCV detects scene changes → keyframes extracted
3. CLIP embeds keyframes → visual embeddings
4. Neo4j stores video → topic → segment relationships

### 7.2 Query a Video

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When does the speaker explain backpropagation?",
    "pipeline": "video_rag",
    "top_k": 3
  }'
```

**Point out:**
- `sources` include `start_timestamp` and `end_timestamp` (in seconds)
- Dual retrieval: text similarity (transcript) + visual similarity (CLIP)
- Fused ranking: 60% text score + 40% visual score

### 7.3 Show MCP Server (Code Walkthrough)

Open `pipelines/video_rag/src/video_rag/mcp_server.py` and show the 4 MCP tools:
- `search_video` — semantic search across video library
- `get_segment` — retrieve specific segment by ID
- `list_videos` — list all indexed videos
- `get_transcript` — get full transcript for a video

### 7.4 Show Neo4j Knowledge Graph (Optional)

- Open Neo4j Browser (http://localhost:7474)
- Run: `MATCH (v:Video)-[:CONTAINS]->(s:Segment)-[:MENTIONS]->(t:Topic) RETURN v, s, t LIMIT 25`
- Show the graph visualization: videos → segments → topics

### Talking Point

> "Video RAG is the most unique pipeline. It combines Whisper transcription with CLIP visual embeddings for dual-modality retrieval. The MCP server exposes video search as tools that Claude can call directly. Neo4j enables multi-hop queries like 'find segments about machine learning in videos tagged AI.'"

---

## Step 8: A/B Pipeline Comparison

**Time:** 2-3 minutes  
**Goal:** Show side-by-side comparison of two pipelines

### 8.1 Run A/B Comparison via API

```bash
# Compare Fastest RAG vs Self-RAG on the same query
# Pipeline A
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the benefits of retrieval-augmented generation over fine-tuning",
    "pipeline": "fastest_rag"
  }'

# Pipeline B (same query)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the benefits of retrieval-augmented generation over fine-tuning",
    "pipeline": "self_rag"
  }'
```

### 8.2 Compare Results Side-by-Side

Create a comparison table from the two responses:

| Metric | Fastest RAG | Self-RAG |
|--------|-------------|----------|
| Latency | ~X ms | ~Y ms |
| Cost | $0.00X | $0.00Y |
| Sources | N chunks | M chunks |
| Answer Quality | Baseline | More thorough |

### 8.3 Show the UI Comparison Page (if UI is running)

The `ui/pages/compare.py` component runs both pipelines in parallel and displays results side-by-side with metrics comparison at the bottom.

### Talking Point

> "The A/B comparison mode lets you run the same query through any two pipelines and see the trade-offs immediately — latency vs. accuracy, cost vs. quality. Fastest RAG is 3x cheaper but Self-RAG catches hallucinations. The right choice depends on your use case."

---

## Step 9: Metrics Dashboard

**Time:** 1-2 minutes  
**Goal:** Show aggregated metrics across all pipelines

### 9.1 Fetch Metrics via API

```bash
# Get summary metrics for all pipelines
curl http://localhost:8000/metrics/summary

# Get historical metrics
curl "http://localhost:8000/metrics/history?hours=24"
```

### 9.2 Key Metrics to Highlight

| Pipeline | Faithfulness | Hallucination Rate | Avg Latency | Avg Cost/Query |
|----------|-------------|-------------------|-------------|----------------|
| Fastest RAG | ~0.70 | ~25% | ~80ms | $0.003 |
| Multimodal RAG | ~0.75 | ~20% | ~150ms | $0.008 |
| CRAG | ~0.82 | ~12% | ~200ms | $0.012 |
| Self-RAG | ~0.88 | ~7% | ~350ms | $0.018 |
| Video RAG | ~0.75 | ~15% | ~250ms | $0.015 |

### 9.3 Cost Estimate

> "At 1,000 queries/day, the platform costs approximately $74/month. Using Haiku for grading nodes instead of Sonnet reduces this by 10x."

### Talking Point

> "The metrics dashboard tracks RAGAS scores, latency percentiles, and USD cost per query across all pipelines in real-time. This lets you make data-driven decisions about which pipeline to deploy for production."

---

## Step 10: Code & Architecture Walkthrough

**Time:** 3-5 minutes  
**Goal:** Show code quality, architecture decisions, and testing

### 10.1 Architecture Overview

Open `README.md` and show the architecture diagram:
- Shared UI → Pipeline Router (FastAPI) → 5 Pipeline Strategies → Shared Infrastructure

### 10.2 Key Files to Show

| File | What to Highlight |
|------|------------------|
| `shared/src/shared/models/document.py` | Pydantic v2 data contracts shared by all pipelines |
| `shared/src/shared/storage/vector_store.py` | pgvector + Qdrant abstraction with common interface |
| `pipelines/corrective_rag/src/corrective_rag/graph.py` | LangGraph StateGraph with conditional edges |
| `pipelines/self_rag/src/self_rag/hallucination_grader.py` | Claude Haiku grading node |
| `pipelines/video_rag/src/video_rag/mcp_server.py` | FastMCP tool definitions |
| `api/src/api/routers/query.py` | Unified query router |
| `api/src/api/middleware/observability.py` | Cost tracking middleware |

### 10.3 Testing

```bash
# Run all unit tests
uv run pytest -m unit -v

# Run integration tests (requires Docker services)
uv run pytest -m integration -v

# Show test count
uv run pytest --collect-only | tail -5
```

**Highlight:** 41 test files, CI/CD runs lint + type check + tests on every PR

### 10.4 Code Quality

```bash
# Linting
uv run ruff check .

# Type checking
uv run mypy shared/src api/src
```

### Talking Point

> "The monorepo uses uv workspaces so each pipeline is an independent Python package with shared infrastructure. Ruff enforces strict linting, mypy checks types, and GitHub Actions runs everything on every PR. The test pyramid covers unit tests, integration tests with real Docker services, and E2E API tests."

---

## Talking Points & Q&A Prep

### Why a Monorepo?

> "Shared embedding service, vector store, and RAGAS evaluation would be duplicated across 5 repos. The monorepo with uv workspaces gives package isolation with shared CI/CD. Portfolio reviewers clone one repo and see the full system."

### Why Claude for Grading?

> "Grading nodes (relevance, hallucination, answer quality) use Claude Haiku at $0.25/1M input tokens. It's a binary yes/no judgment — you don't need Sonnet-level reasoning. This mirrors production cost-optimization patterns."

### How Does HyDE Work?

> "Hypothetical Document Embeddings: instead of embedding the query directly, the LLM generates a hypothetical document that would answer the query, then we embed *that* document. The hypothesis embedding is often closer in vector space to the actual relevant documents than the original query."

### What's the Cost at Scale?

> "At 1,000 queries/day: ~$74/month using the Self-RAG pipeline. Fastest RAG with caching drops to ~$12/month. The cost tracking middleware logs every token and every dollar to LangFuse."

### Common Questions

| Question | Answer |
|----------|--------|
| "Can it handle production traffic?" | "FastAPI + async throughout. Horizontal scaling via multiple API instances behind a load balancer." |
| "What about data privacy?" | "All vector storage is self-hosted (pgvector). LangFuse runs locally. Only LLM calls go to external APIs." |
| "Why not use LangChain for everything?" | "LangGraph for orchestration, but raw Anthropic SDK for simple pipelines. LangChain adds overhead where it's not needed." |
| "How do you handle stale cache?" | "Redis TTL + semantic similarity threshold (cosine > 0.95). Similar but different queries bypass cache." |

---

## Troubleshooting

### Docker Services Won't Start

```bash
# Check for port conflicts
docker-compose logs postgres
docker-compose logs qdrant

# Reset volumes if corrupt
docker-compose down -v && docker-compose up -d
```

### API Key Errors

```bash
# Verify .env is loaded
cat .env | grep API_KEY

# Test Anthropic key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-4-5-20251001","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}'
```

### pgvector Extension Missing

```bash
docker exec -it rag-postgres psql -U rag_user -d rag_platform -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Slow First Query

First query is slow because:
1. Embedding model loads on first call
2. pgvector creates index on first search
3. Redis cache is cold

Subsequent queries will be much faster.

### LangFuse Not Showing Traces

- Verify `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` in `.env`
- Check LangFuse is running: http://localhost:3000
- Traces may take 5-10 seconds to appear

---

## Quick Reference: All Demo Commands

```bash
# === SETUP ===
docker-compose up -d
uv sync --all-packages
uv run uvicorn api.src.api.main:app --reload --port 8000

# === QUERIES ===
# Fastest RAG
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "pipeline": "fastest_rag"}' | python -m json.tool

# Multimodal RAG
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "What does the chart show?", "pipeline": "multimodal_rag"}' | python -m json.tool

# CRAG
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "Latest quantum computing breakthroughs?", "pipeline": "crag"}' | python -m json.tool

# Self-RAG
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "Compare BERT vs GPT", "pipeline": "self_rag"}' | python -m json.tool

# Video RAG
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "When is backpropagation explained?", "pipeline": "video_rag"}' | python -m json.tool

# === METRICS ===
curl -s http://localhost:8000/metrics/summary | python -m json.tool

# === STANDALONE DEMOS ===
cd pipelines/multimodal_rag && uv run streamlit run demo.py
cd pipelines/corrective_rag && uv run streamlit run demo.py
cd pipelines/self_rag && uv run streamlit run demo.py

# === TESTS ===
uv run pytest -m unit -v
uv run pytest -m integration -v
uv run ruff check .
```
