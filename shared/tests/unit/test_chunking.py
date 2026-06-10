"""Unit tests for all chunking strategies."""

import pytest

from shared.ingestion.chunking import (
    ChunkingStrategies,
    ChunkingStrategy,
    FixedSizeChunker,
    SemanticChunker,
    SlidingWindowChunker,
)
from shared.models.document import ChunkType

LONG_TEXT = " ".join([f"word{i}" for i in range(1200)])  # 1200-word document

PARAGRAPH_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines retrieval systems with generative
language models. It was first introduced by Lewis et al. in 2020.

The core idea is simple: instead of relying solely on the parametric knowledge stored in model
weights, the model can look up relevant documents at inference time.

This approach has several advantages. First, it allows the model to access up-to-date information
without retraining. Second, it provides provenance — the model can cite its sources.

However, RAG also has limitations. Retrieval quality directly impacts generation quality. If the
retrieved documents are irrelevant, the model may hallucinate or provide incorrect answers.
""".strip()


class TestFixedSizeChunker:
    def test_basic_split(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(LONG_TEXT, document_id="doc-1")
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(LONG_TEXT, document_id="doc-1")
        for c in chunks:
            assert c.content.strip()

    def test_chunk_indices_sequential(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(LONG_TEXT, document_id="doc-1")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_start_index_offset(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(LONG_TEXT, document_id="doc-1", start_index=5)
        assert chunks[0].chunk_index == 5

    def test_single_short_text(self):
        chunker = FixedSizeChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk("Hello world", document_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"

    def test_chunk_type_propagated(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(
            LONG_TEXT, document_id="doc-1", chunk_type=ChunkType.VIDEO_TRANSCRIPT
        )
        for c in chunks:
            assert c.chunk_type == ChunkType.VIDEO_TRANSCRIPT

    def test_document_id_propagated(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(LONG_TEXT, document_id="my-doc")
        for c in chunks:
            assert c.document_id == "my-doc"


class TestSlidingWindowChunker:
    def test_produces_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=100, step_size=50)
        chunks = chunker.chunk(LONG_TEXT, document_id="doc-1")
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=100, step_size=50)
        chunks = chunker.chunk(LONG_TEXT, document_id="doc-1")
        for c in chunks:
            assert c.content.strip()


class TestSemanticChunker:
    def test_paragraph_boundaries_respected(self):
        chunker = SemanticChunker(chunk_size=100)
        chunks = chunker.chunk(PARAGRAPH_TEXT, document_id="doc-1")
        assert len(chunks) >= 1

    def test_no_empty_chunks(self):
        chunker = SemanticChunker(chunk_size=100)
        chunks = chunker.chunk(PARAGRAPH_TEXT, document_id="doc-1")
        for c in chunks:
            assert c.content.strip()

    def test_oversized_paragraph_handled(self):
        # A single paragraph that exceeds chunk_size should be sub-chunked
        huge_para = " ".join([f"word{i}" for i in range(300)])
        chunker = SemanticChunker(chunk_size=100)
        chunks = chunker.chunk(huge_para, document_id="doc-1")
        assert len(chunks) >= 2


class TestChunkingStrategiesFactory:
    def test_fixed_size(self):
        chunker = ChunkingStrategies.get(ChunkingStrategy.FIXED_SIZE)
        assert isinstance(chunker, FixedSizeChunker)

    def test_sliding_window(self):
        chunker = ChunkingStrategies.get(ChunkingStrategy.SLIDING_WINDOW)
        assert isinstance(chunker, SlidingWindowChunker)

    def test_semantic(self):
        chunker = ChunkingStrategies.get(ChunkingStrategy.SEMANTIC)
        assert isinstance(chunker, SemanticChunker)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            ChunkingStrategies.get("nonexistent")  # type: ignore[arg-type]
