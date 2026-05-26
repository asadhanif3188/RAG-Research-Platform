"""DocumentIngestionPipeline — orchestrates parse → chunk → embed → store."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.embeddings.service import EmbeddingService
from shared.ingestion.chunking import ChunkingStrategies, ChunkingStrategy
from shared.ingestion.pdf_parser import PDFParser
from shared.models.document import ChunkType, DocumentChunk
from shared.storage.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    document_id: str
    total_chunks: int
    stored_chunk_ids: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentIngestionPipeline:
    """End-to-end pipeline: PDF → parsed pages → chunks → embeddings → vector store.

    Usage:
        pipeline = DocumentIngestionPipeline(
            vector_store=pgvector_client,
            embedding_service=embedding_svc,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
        )
        result = await pipeline.ingest_pdf("paper.pdf")
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedding_service: EmbeddingService,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        extract_images: bool = False,
        batch_embed_size: int = 50,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._chunking_strategy = chunking_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._extract_images = extract_images
        self._batch_embed_size = batch_embed_size

        self._pdf_parser = PDFParser(extract_images=extract_images)
        self._chunker = ChunkingStrategies.get(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

    async def ingest_pdf(
        self,
        pdf_path: str | Path,
        document_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Full ingestion of a single PDF. Returns an IngestionResult."""
        start = time.perf_counter()
        path = Path(pdf_path)

        # 1. Parse
        parsed = self._pdf_parser.parse(path)
        doc_id = document_id or parsed.document_id
        base_metadata = {**parsed.metadata, **(extra_metadata or {})}

        # 2. Chunk — text and tables
        all_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in parsed.pages:
            page_meta = {**base_metadata, "page_number": page.page_number}

            # Text chunks
            if page.text.strip():
                text_chunks = self._chunker.chunk(
                    text=page.text,
                    document_id=doc_id,
                    chunk_type=ChunkType.TEXT,
                    metadata=page_meta,
                    start_index=chunk_index,
                )
                all_chunks.extend(text_chunks)
                chunk_index += len(text_chunks)

            # Table chunks (stored as-is — no further chunking)
            for table_text in page.tables:
                all_chunks.append(
                    DocumentChunk(
                        document_id=doc_id,
                        chunk_index=chunk_index,
                        chunk_type=ChunkType.TABLE,
                        content=table_text,
                        metadata=page_meta,
                    )
                )
                chunk_index += 1

        logger.info(
            "Ingestion '%s': %d chunks from %d pages",
            path.name,
            len(all_chunks),
            parsed.total_pages,
        )

        # 3. Embed in batches
        all_chunks = await self._embed_chunks(all_chunks)

        # 4. Store in vector DB
        stored_ids = await self._vector_store.upsert(all_chunks)

        elapsed = time.perf_counter() - start
        logger.info(
            "Stored %d chunks for document '%s' in %.2fs",
            len(stored_ids),
            doc_id,
            elapsed,
        )

        return IngestionResult(
            document_id=doc_id,
            total_chunks=len(all_chunks),
            stored_chunk_ids=stored_ids,
            elapsed_seconds=elapsed,
            metadata={
                "chunking_strategy": self._chunking_strategy.value,
                "embedding_tokens": self._embedding_service.total_tokens_used,
                "embedding_cost_usd": self._embedding_service.total_cost_usd,
            },
        )

    async def ingest_text(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest raw text (no PDF parsing). Useful for programmatic ingestion."""
        start = time.perf_counter()

        chunks = self._chunker.chunk(
            text=text,
            document_id=document_id,
            chunk_type=chunk_type,
            metadata=extra_metadata or {},
        )
        chunks = await self._embed_chunks(chunks)
        stored_ids = await self._vector_store.upsert(chunks)

        elapsed = time.perf_counter() - start
        return IngestionResult(
            document_id=document_id,
            total_chunks=len(chunks),
            stored_chunk_ids=stored_ids,
            elapsed_seconds=elapsed,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Embed all chunks in batches, mutating the embedding field in-place."""
        texts = [c.content for c in chunks]
        embeddings = await self._embedding_service.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks
