# Contributing

Thanks for your interest in the RAG Research Platform!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/asadhanif3188/rag-research-platform
cd rag-research-platform

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker-compose up -d

# Install dependencies
uv sync --all-packages

# Run tests
uv run pytest

# Lint and format
uv run ruff format .
uv run ruff check .
```

## Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new pipeline strategy
fix: correct cache hit rate calculation
docs: update API reference
test: add integration tests for CRAG graph
refactor: simplify vector store abstraction
```

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
