"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from shared.models.document import ChunkType, DocumentChunk


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    return DocumentChunk(
        id="test-chunk-001",
        document_id="doc-001",
        chunk_index=0,
        chunk_type=ChunkType.TEXT,
        content="Retrieval-Augmented Generation combines dense retrieval with language models.",
        metadata={"page_number": 1},
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    texts = [
        "RAG is a technique that retrieves relevant documents before generating an answer.",
        "Vector databases store embeddings for fast nearest-neighbour search.",
        "pgvector is a PostgreSQL extension that adds vector similarity search.",
        "Qdrant supports binary quantization for 32x memory reduction.",
        "Redis can cache semantic search results to reduce latency.",
    ]
    return [
        DocumentChunk(
            id=f"chunk-{i:03d}",
            document_id="doc-001",
            chunk_index=i,
            chunk_type=ChunkType.TEXT,
            content=text,
            metadata={"page_number": i + 1},
            embedding=[0.1 * (i + 1)] * 3072,
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def fake_embedding() -> list[float]:
    """Deterministic 3072-dim unit vector for tests."""
    import math
    vec = [1.0 / math.sqrt(3072)] * 3072
    return vec
