FROM python:3.12-slim AS base

# System dependencies for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace config
COPY pyproject.toml uv.lock* ./
COPY shared/ shared/
COPY pipelines/video_rag/ pipelines/video_rag/

# Install dependencies
RUN uv sync --package video-rag --frozen --no-dev

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime

COPY --from=base /app /app
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Default: run the MCP server
CMD ["uv", "run", "fastmcp", "run", "pipelines/video_rag/src/video_rag/mcp_server.py"]
