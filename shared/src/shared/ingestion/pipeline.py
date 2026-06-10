"""DocumentIngestionPipeline — orchestrates parse → chunk → embed → store.

Optionally accepts a *vision_describer* callable to convert extracted image bytes
into text descriptions stored as IMAGE_DESCRIPTION chunks. When provided, the
pipeline will call `await vision_describer(image_bytes, media_type, metadata)`
for each image on every page, expecting a string description in return.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.ingestion.chunking import ChunkingStrategies, ChunkingStrategy
from shared.ingestion.pdf_parser import PDFParser
from shared.models.document import ChunkType, DocumentChunk

if TYPE_CHECKING:
    from shared.embeddings.service import EmbeddingService
    from shared.storage.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)

# Type alias for the optional vision description callable
VisionDescriberFn = Callable[[bytes, str, dict[str, Any]], Awaitable[str]]


@dataclass
class IngestionResult:
    document_id: str
    total_chunks: int
    stored_chunk_ids: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentIngestionPipeline:
    """End-to-end pipeline: PDF → parsed pages → chunks → embeddings → vector store.

    Usage (text-only):
        pipeline = DocumentIngestionPipeline(
            vector_store=pgvector_client,
            embedding_service=embedding_svc,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
        )
        result = await pipeline.ingest_pdf("paper.pdf")

    Usage (multimodal — with vision descriptions):
        async def describe(img_bytes, media_type, meta):
            return await vision_describer.describe_image(img_bytes, media_type)

        pipeline = DocumentIngestionPipeline(..., vision_describer=describe)
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
        vision_describer: VisionDescriberFn | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._chunking_strategy = chunking_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._extract_images = extract_images or (vision_describer is not None)
        self._batch_embed_size = batch_embed_size
        self._vision_describer = vision_describer

        self._pdf_parser = PDFParser(extract_images=self._extract_images)
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

            # Image description chunks (requires vision_describer)
            if self._vision_describer and page.images:
                image_chunks = await self._describe_page_images(
                    pdf_path=path,
                    page_images=page.images,
                    document_id=doc_id,
                    page_meta=page_meta,
                    start_index=chunk_index,
                )
                all_chunks.extend(image_chunks)
                chunk_index += len(image_chunks)

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

    async def _describe_page_images(
        self,
        pdf_path: Path,
        page_images: list[dict[str, Any]],
        document_id: str,
        page_meta: dict[str, Any],
        start_index: int,
    ) -> list[DocumentChunk]:
        """Describe each image on a page via the vision_describer callable."""
        chunks: list[DocumentChunk] = []
        chunk_idx = start_index

        for img_ref in page_images:
            xref = img_ref.get("xref")
            if xref is None:
                continue
            try:
                img_bytes, media_type = self._pdf_parser.extract_image_bytes(pdf_path, xref)
                img_meta = {**page_meta, "xref": xref, "media_type": media_type}
                description = await self._vision_describer(img_bytes, media_type, img_meta)  # type: ignore[misc]
                if description and description.strip():
                    chunks.append(
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=chunk_idx,
                            chunk_type=ChunkType.IMAGE_DESCRIPTION,
                            content=description,
                            metadata=img_meta,
                        )
                    )
                    chunk_idx += 1
            except Exception as exc:
                logger.warning("Image description failed (xref=%s): %s", xref, exc)

        return chunks

    async def _embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Embed all chunks in batches, mutating the embedding field in-place."""
        texts = [c.content for c in chunks]
        embeddings = await self._embedding_service.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk.embedding = embedding

        return chunks
