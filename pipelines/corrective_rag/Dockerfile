FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace root config
COPY pyproject.toml ./pyproject.toml

# Copy shared package
COPY shared/ ./shared/

# Copy corrective_rag package
COPY pipelines/corrective_rag/ ./pipelines/corrective_rag/

# Install dependencies
RUN uv sync --no-dev --frozen 2>/dev/null || uv pip install --system \
    ./shared \
    ./pipelines/corrective_rag

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import corrective_rag; print('ok')"

# Run standalone demo
CMD ["streamlit", "run", "pipelines/corrective_rag/demo.py", \
     "--server.port=8501", "--server.headless=true"]
