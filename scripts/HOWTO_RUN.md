# How to Run & Test the RAG Pipelines

Two pipelines are fully implemented:

| Pipeline | Description |
|---|---|
| **Fastest RAG** | Baseline: embed → pgvector search → Claude generate |
| **Multimodal RAG** | Extends above with image descriptions, table chunks, and provenance |

---

## Quick Start (No Docker, No API Keys)

Mock mode simulates all external services in memory.

```bash
# From repo root
python scripts/run_demo.py
```

You will see both pipelines run, a side-by-side comparison table, and
provenance attribution for the Multimodal RAG response.

**Options:**

```bash
# Run only one pipeline
python scripts/run_demo.py --pipeline fastest_rag
python scripts/run_demo.py --pipeline multimodal_rag

# Custom query
python scripts/run_demo.py --query "How does binary quantization reduce memory?"

# Show full source chunk content
python scripts/run_demo.py --verbose

# Dump raw JSON (useful for scripting)
python scripts/run_demo.py --json
```

---

## Full Live Mode (Docker + API Keys)

### 1. Prerequisites

- Docker Desktop running
- `uv` installed (`pip install uv`)
- Anthropic API key (`ANTHROPIC_API_KEY`)
- OpenAI API key (`OPENAI_API_KEY`)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
```

### 3. Start infrastructure

```bash
docker-compose up -d
```

This starts: PostgreSQL + pgvector, Qdrant, Redis, Neo4j, LangFuse.

Wait ~15 seconds for all services to be healthy:

```bash
docker-compose ps        # all should show "healthy"
```

### 4. Install Python packages

```bash
uv sync
```

### 5. Run live demo

```bash
python scripts/run_demo.py --live
```

---

## Running Tests

### Unit tests only (no Docker required)

```bash
# All unit tests across shared + both pipelines
uv run pytest shared/tests/unit pipelines/fastest_rag/tests/unit pipelines/multimodal_rag/tests/unit -v
```

### Fastest RAG tests only

```bash
uv run pytest pipelines/fastest_rag/tests/ -v
```

### Multimodal RAG tests only

```bash
uv run pytest pipelines/multimodal_rag/tests/ -v
```

### Integration tests (requires Docker)

```bash
# Start infrastructure first (see step 3 above)
uv run pytest shared/tests/integration pipelines/fastest_rag/tests/integration pipelines/multimodal_rag/tests/integration -v -m integration
```

### Full test suite with coverage

```bash
uv run pytest --cov=shared/src --cov=pipelines/fastest_rag/src --cov=pipelines/multimodal_rag/src --cov-report=term-missing
```

---

## Running the API Server

```bash
# Start Docker stack first (see step 3 above)
uv run uvicorn api.main:app --reload --port 8000
```

Then call via curl or the Swagger UI at http://localhost:8000/docs.

### Example: Fastest RAG

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "pipeline": "fastest_rag",
    "top_k": 3,
    "use_cache": true
  }' | python -m json.tool
```

### Example: Multimodal RAG

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What do the charts show about retrieval performance?",
    "pipeline": "multimodal_rag",
    "top_k": 5,
    "use_cache": false
  }' | python -m json.tool
```

### Health check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/query/health
```

---

## Ingesting Documents (Live Mode)

To query real documents, ingest PDFs first using the pipeline:

```python
# ingest_sample.py — run from repo root: python ingest_sample.py
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def main():
    from shared.config import get_settings
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import PgVectorClient
    from shared.ingestion.pipeline import DocumentIngestionPipeline

    settings = get_settings()

    emb = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )
    emb.connect()

    store = PgVectorClient(database_url=settings.database_url)
    await store.connect()

    ingest = DocumentIngestionPipeline(vector_store=store, embedding_service=emb)

    for pdf in Path("data/").glob("*.pdf"):
        result = await ingest.ingest_pdf(pdf)
        print(f"Ingested {result.document_id}: {result.total_chunks} chunks "
              f"in {result.elapsed_seconds:.1f}s")

asyncio.run(main())
```

For **Multimodal** ingestion (images described via Claude vision):

```python
from multimodal_rag.vision_describer import VisionDescriber

vision = VisionDescriber(api_key=settings.anthropic_api_key)
vision.connect()

async def describe(img_bytes, media_type, meta):
    return await vision.describe_image(img_bytes, media_type)

ingest = DocumentIngestionPipeline(
    vector_store=store,
    embedding_service=emb,
    vision_describer=describe,   # <-- enables image chunk extraction
)
```

---

## Viewing Provenance in Python

```python
from ui.components.provenance_viewer import ProvenanceViewer

# response = result from MultimodalRAGPipeline.run(...)
viewer = ProvenanceViewer(
    answer=response.answer,
    provenance=response.metadata.get("provenance", []),
)

# Print Markdown attribution to terminal
print(viewer.to_markdown())

# Or send as a Chainlit message (inside a Chainlit app)
# await viewer.send()
```

---

## Project Structure (relevant paths)

```
scripts/
  run_demo.py              ← this demo runner
  HOWTO_RUN.md             ← this file

pipelines/
  fastest_rag/src/         ← NaiveRAGPipeline, CacheLayer, benchmark
  multimodal_rag/src/      ← MultimodalRAGPipeline, VisionDescriber,
                              TableExtractor, ProvenanceTracker

shared/src/shared/
  ingestion/               ← DocumentIngestionPipeline, PDFParser, chunkers
  storage/                 ← PgVectorClient, RedisSemanticCache
  embeddings/              ← EmbeddingService (OpenAI)
  models/                  ← Pydantic schemas

api/src/api/
  main.py                  ← FastAPI app
  routers/query.py         ← POST /query router
  dependencies.py          ← DI singletons

ui/components/
  provenance_viewer.py     ← Chainlit provenance HTML component
```
