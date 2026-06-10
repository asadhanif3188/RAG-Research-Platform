# Multimodal RAG — standalone Streamlit demo
#
# Build:  docker build -t multimodal-rag .
# Run:    docker run --rm -p 8501:8501 \
#           -e ANTHROPIC_API_KEY=sk-ant-... \
#           -e OPENAI_API_KEY=sk-... \
#           multimodal-rag
#
# With infrastructure (pgvector + Redis on rag_network):
#   docker run --rm -p 8501:8501 --network rag_network \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -e OPENAI_API_KEY=sk-... \
#     -e DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@postgres:5432/ragdb \
#     -e REDIS_URL=redis://redis:6379/0 \
#     multimodal-rag

FROM python:3.12-slim AS base

# System deps for PyMuPDF and pdfplumber
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libmupdf-dev \
        libfreetype6-dev \
        libjpeg62-turbo-dev \
        libharfbuzz-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace layout for uv resolution
COPY pyproject.toml /app/pyproject.toml
COPY shared/ /app/shared/
COPY pipelines/fastest_rag/ /app/pipelines/fastest_rag/
COPY pipelines/multimodal_rag/ /app/pipelines/multimodal_rag/

# Install all dependencies
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Install streamlit (not in pyproject.toml — demo-only dependency)
RUN uv pip install streamlit numpy

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run the Streamlit demo
ENTRYPOINT ["uv", "run", "streamlit", "run", \
    "pipelines/multimodal_rag/demo.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
