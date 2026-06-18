# Deployment Guide

## Local Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose

### Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd rag-research-platform

# 2. Copy environment config
cp .env.example .env
# Edit .env with your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

# 3. Start infrastructure services
docker-compose up -d

# 4. Install dependencies
uv sync --all-packages

# 5. Run the API server
uv run uvicorn api.src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Run the Chainlit UI (separate terminal)
uv run chainlit run ui/app.py --port 8501
```

### Infrastructure Services

| Service    | Port  | Purpose                           |
|-----------|-------|-----------------------------------|
| PostgreSQL | 5432  | Vector store (pgvector extension) |
| Qdrant     | 6333  | Alternative vector DB             |
| Redis      | 6379  | Semantic cache layer              |
| Neo4j      | 7474/7687 | Knowledge graph (Video RAG)   |
| LangFuse   | 3000  | Observability & tracing           |

### Environment Variables

| Variable              | Required | Description                        |
|----------------------|----------|------------------------------------|
| `ANTHROPIC_API_KEY`  | Yes      | Claude API key for LLM calls       |
| `OPENAI_API_KEY`     | Yes      | OpenAI key for embeddings          |
| `TAVILY_API_KEY`     | Yes      | Tavily key for CRAG web search     |
| `DATABASE_URL`       | No       | PostgreSQL connection string       |
| `REDIS_URL`          | No       | Redis connection string            |
| `NEO4J_URI`          | No       | Neo4j bolt URI                     |
| `NEO4J_PASSWORD`     | No       | Neo4j password                     |
| `LANGFUSE_SECRET_KEY`| No       | LangFuse secret for observability  |
| `LANGFUSE_PUBLIC_KEY`| No       | LangFuse public key                |
| `LANGFUSE_HOST`      | No       | LangFuse server URL                |

## Docker Deployment

### Build the API Image

```bash
docker build -t rag-platform-api:latest -f Dockerfile.api .
```

### Build the UI Image

```bash
docker build -t rag-platform-ui:latest -f Dockerfile.ui .
```

### Run with Docker Compose (Production)

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Cloud Deployment

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login and create app
fly auth login
fly apps create rag-platform-api

# Set secrets
fly secrets set ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... TAVILY_API_KEY=tvly-...

# Deploy API
fly deploy --config fly.api.toml

# Deploy UI
fly apps create rag-platform-ui
fly deploy --config fly.ui.toml
```

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and init
railway login
railway init

# Link to project
railway link

# Deploy
railway up
```

## Scaling Notes

### API (FastAPI)

- **Stateless**: Each API instance is independent — scale horizontally
- Run multiple uvicorn workers: `uvicorn api.main:app --workers 4`
- Behind a load balancer (Fly.io, Railway, or nginx)
- Connection pooling: SQLAlchemy async with pool_size=10, max_overflow=20

### UI (Chainlit)

- **Stateless frontend**: Session data stored in browser
- Scale independently from API
- Single instance sufficient for moderate traffic

### Database Scaling

- **PostgreSQL**: Use connection pooler (PgBouncer) for >50 concurrent connections
- **Redis**: Single instance handles ~100K ops/sec; cluster for higher throughput
- **Neo4j**: Single instance for dev; Aura for production workloads
- **Qdrant**: Horizontal scaling with sharding for large vector collections

### Cost Estimation (1000 queries/day)

| Component           | Estimated Monthly Cost |
|--------------------|----------------------|
| Claude Sonnet API  | ~$45 (avg 500 tokens/query) |
| OpenAI Embeddings  | ~$4 (3072-dim embeddings) |
| Tavily Web Search  | ~$10 (CRAG fallback ~20% of queries) |
| Fly.io API (shared-cpu-1x) | ~$5 |
| Fly.io DB (1GB)    | ~$7 |
| Redis (256MB)      | ~$3 |
| **Total**          | **~$74/month** |

## Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m unit

# Integration tests (requires Docker services)
uv run pytest -m integration

# E2E API tests
uv run pytest api/tests/e2e/

# With coverage
uv run pytest --cov=shared --cov=api --cov-report=term-missing
```

## Linting & Formatting

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy .
```
