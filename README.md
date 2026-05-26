# RAG Research Platform

A unified monorepo showcasing five production-grade RAG pipeline strategies — built as a single platform with shared infrastructure, an A/B comparison UI, and RAGAS-based evaluation.

## Architecture

```
rag-research-platform/
├── shared/          ← Core infrastructure: models, storage, embeddings, ingestion, eval
├── pipelines/       ← Five RAG strategies (built in later phases)
│   ├── fastest_rag/
│   ├── multimodal_rag/
│   ├── corrective_rag/
│   ├── self_rag/
│   └── video_rag/
├── api/             ← FastAPI pipeline router
├── ui/              ← Chainlit chat + metrics dashboard
├── infra/           ← SQL init scripts
└── docker-compose.yml
```

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)

### 2. Clone and configure

```bash
git clone https://github.com/your-username/rag-research-platform
cd rag-research-platform
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and ANTHROPIC_API_KEY
```

### 3. Start all services

```bash
docker-compose up -d
```

This starts:

| Service | URL | Purpose |
|---------|-----|---------|
| PostgreSQL + pgvector | `localhost:5432` | Primary vector store |
| Qdrant | `localhost:6333` | Binary-quantized vector store |
| Redis | `localhost:6379` | Semantic query cache |
| Neo4j | `localhost:7474` | Video topic knowledge graph |
| LangFuse | `localhost:3000` | LLM observability dashboard |

Verify all services are healthy:

```bash
docker-compose ps
```

### 4. Install Python dependencies

```bash
uv sync
```

### 5. Run unit tests

```bash
uv run pytest shared/tests/unit/ -v
```

### 6. Run integration tests (requires Docker)

```bash
uv run pytest shared/tests/integration/ -v -m integration
```

---

## Shared Infrastructure (`shared/`)

The `shared` package is the foundation that all pipeline packages depend on.

### Key modules

| Module | Description |
|--------|-------------|
| `shared.config` | Pydantic Settings — all config from env vars |
| `shared.models` | `DocumentChunk`, `QueryRequest`, `QueryResponse`, `RetrievalResult`, `EvalResult` |
| `shared.storage.vector_store` | `PgVectorClient` + `QdrantVectorClient` behind `VectorStoreClient` ABC |
| `shared.storage.cache` | `RedisSemanticCache` — cosine-similarity cache with TTL |
| `shared.storage.neo4j_client` | `Neo4jClient` — video topic graph CRUD |
| `shared.embeddings.service` | `EmbeddingService` — batched OpenAI embeddings with caching + cost tracking |
| `shared.ingestion.chunking` | `FixedSizeChunker`, `SlidingWindowChunker`, `SemanticChunker` |
| `shared.ingestion.pdf_parser` | `PDFParser` — text + table extraction via PyMuPDF |
| `shared.ingestion.pipeline` | `DocumentIngestionPipeline` — PDF → chunks → embeddings → vector store |
| `shared.eval.ragas_runner` | `RAGASRunner` — faithfulness, answer relevancy, context precision/recall |

### Chunking strategies

```python
from shared.ingestion.chunking import ChunkingStrategies, ChunkingStrategy

chunker = ChunkingStrategies.get(ChunkingStrategy.SEMANTIC, chunk_size=512, overlap=64)
chunks = chunker.chunk(text, document_id="my-doc")
```

### Ingest a PDF

```python
from shared.ingestion.pipeline import DocumentIngestionPipeline
from shared.ingestion.chunking import ChunkingStrategy
from shared.storage.vector_store import PgVectorClient
from shared.embeddings.service import EmbeddingService

vector_store = PgVectorClient(database_url=settings.database_url)
await vector_store.connect()

embedding_svc = EmbeddingService(api_key=settings.openai_api_key)
embedding_svc.connect()

pipeline = DocumentIngestionPipeline(
    vector_store=vector_store,
    embedding_service=embedding_svc,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
)
result = await pipeline.ingest_pdf("paper.pdf")
print(f"Stored {result.total_chunks} chunks in {result.elapsed_seconds:.2f}s")
```

### Evaluate with RAGAS

```python
from shared.eval.ragas_runner import RAGASRunner

runner = RAGASRunner(
    anthropic_api_key=settings.anthropic_api_key,
    openai_api_key=settings.openai_api_key,
)
eval_result = await runner.evaluate(
    query="What is RAG?",
    answer="RAG stands for Retrieval-Augmented Generation...",
    contexts=["<retrieved chunk 1>", "<retrieved chunk 2>"],
    ground_truth="Retrieval-Augmented Generation combines...",
)
print(f"Faithfulness: {eval_result.faithfulness:.3f}")
print(f"Aggregate: {eval_result.aggregate_score():.3f}")
```

---

## Environment Variables

See [.env.example](.env.example) for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | For embeddings (text-embedding-3-large) |
| `ANTHROPIC_API_KEY` | For generation (Claude models) |
| `DATABASE_URL` | PostgreSQL connection string |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant connection |
| `REDIS_URL` | Redis connection |
| `NEO4J_URI` | Neo4j Bolt URI |
| `LANGFUSE_SECRET_KEY` | LangFuse observability |
| `SEMANTIC_CACHE_THRESHOLD` | Cosine similarity threshold for cache hits (default: 0.92) |

---

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Shared Infrastructure | **Done** | Vector store, ingestion, eval harness |
| 1 — Fastest RAG Stack | Pending | Baseline + binary quantization + Redis cache |
| 2 — Multimodal RAG | Pending | Vision descriptions for images/tables |
| 3 — Corrective RAG (CRAG) | Pending | LangGraph with relevance grading + Tavily fallback |
| 4 — Self-RAG | Pending | Agentic decision graph with HyDE |
| 5 — MCP Video RAG | Pending | Whisper + CLIP + Neo4j topic graph |
| 6 — Integration & Polish | Pending | Unified UI, A/B comparison, benchmark dashboard |

---

## Tech Stack

Python 3.12 · uv workspaces · FastAPI · Chainlit · LangGraph · Claude API · OpenAI Embeddings · pgvector · Qdrant · Redis · Neo4j · LangFuse · RAGAS · PyMuPDF · Pydantic v2
