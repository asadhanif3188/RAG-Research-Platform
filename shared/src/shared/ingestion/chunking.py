"""Chunking strategies: fixed-size, semantic, and sliding-window."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from shared.models.document import ChunkType, DocumentChunk

logger = logging.getLogger(__name__)


class ChunkingStrategy(StrEnum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


class BaseChunker(ABC):
    @abstractmethod
    def chunk(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """Split text into DocumentChunk objects."""


# ── Fixed-size chunker ────────────────────────────────────────────────────────


class FixedSizeChunker(BaseChunker):
    """Split text into chunks of approximately `chunk_size` tokens with `overlap` token overlap.

    Uses a simple whitespace tokenisation for speed (not tiktoken).
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        words = text.split()
        chunks: list[DocumentChunk] = []
        step = max(1, self._chunk_size - self._overlap)
        i = 0
        chunk_idx = start_index

        while i < len(words):
            chunk_words = words[i : i + self._chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        chunk_type=chunk_type,
                        content=chunk_text,
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
            i += step

        logger.debug("FixedSizeChunker: %d chunks from %d words", len(chunks), len(words))
        return chunks


# ── Sliding-window chunker ────────────────────────────────────────────────────


class SlidingWindowChunker(BaseChunker):
    """Sentence-aware sliding window — preserves sentence boundaries.

    Fills a window up to `chunk_size` words, slides by `step_size` words.
    """

    def __init__(self, chunk_size: int = 512, step_size: int = 256) -> None:
        self._chunk_size = chunk_size
        self._step_size = step_size

    def chunk(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[DocumentChunk] = []
        current_words: list[str] = []
        chunk_idx = start_index

        def flush() -> None:
            nonlocal chunk_idx
            content = " ".join(current_words).strip()
            if content:
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        chunk_type=chunk_type,
                        content=content,
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1

        for sentence in sentences:
            words = sentence.split()
            if len(current_words) + len(words) > self._chunk_size:
                flush()
                # Slide: keep last step_size words as context
                current_words = current_words[-self._step_size :] + words
                # If a single sentence is larger than chunk_size, split it into multiple chunks
                while len(current_words) > self._chunk_size:
                    chunk_words = current_words[: self._chunk_size]
                    chunks.append(
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=chunk_idx,
                            chunk_type=chunk_type,
                            content=" ".join(chunk_words),
                            metadata=metadata or {},
                        )
                    )
                    chunk_idx += 1
                    current_words = current_words[self._step_size :]
            else:
                current_words.extend(words)

        flush()
        logger.debug("SlidingWindowChunker: %d chunks", len(chunks))
        return chunks


# ── Semantic chunker ─────────────────────────────────────────────────────────


class SemanticChunker(BaseChunker):
    """Paragraph-boundary chunker that groups paragraphs until a size threshold.

    Uses double-newline splits as paragraph boundaries, avoiding mid-paragraph cuts.
    Falls back to FixedSizeChunker for very long single paragraphs.
    """

    def __init__(self, chunk_size: int = 512, max_tokens_per_paragraph: int = 200) -> None:
        self._chunk_size = chunk_size
        self._fallback = FixedSizeChunker(chunk_size=chunk_size, overlap=64)

    def chunk(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: list[DocumentChunk] = []
        current_parts: list[str] = []
        current_word_count = 0
        chunk_idx = start_index

        def flush_current() -> None:
            nonlocal chunk_idx, current_word_count
            if current_parts:
                content = "\n\n".join(current_parts).strip()
                if content:
                    chunks.append(
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=chunk_idx,
                            chunk_type=chunk_type,
                            content=content,
                            metadata=metadata or {},
                        )
                    )
                    chunk_idx += 1
                current_parts.clear()
                current_word_count = 0

        for para in paragraphs:
            word_count = len(para.split())
            if word_count > self._chunk_size:
                # Oversized paragraph — flush current, then sub-chunk
                flush_current()
                sub_chunks = self._fallback.chunk(
                    para, document_id, chunk_type, metadata, chunk_idx
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                continue

            if current_word_count + word_count > self._chunk_size:
                flush_current()

            current_parts.append(para)
            current_word_count += word_count

        flush_current()
        logger.debug("SemanticChunker: %d chunks from %d paragraphs", len(chunks), len(paragraphs))
        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────


class ChunkingStrategies:
    """Factory for chunking strategy instances."""

    @staticmethod
    def get(
        strategy: ChunkingStrategy,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> BaseChunker:
        match strategy:
            case ChunkingStrategy.FIXED_SIZE:
                return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
            case ChunkingStrategy.SLIDING_WINDOW:
                return SlidingWindowChunker(chunk_size=chunk_size, step_size=overlap)
            case ChunkingStrategy.SEMANTIC:
                return SemanticChunker(chunk_size=chunk_size)
            case _:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
