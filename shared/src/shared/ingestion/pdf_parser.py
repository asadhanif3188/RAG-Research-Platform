"""PDF parser using PyMuPDF (fitz).

Extracts:
- Text blocks per page → TEXT chunks
- Image descriptions (via vision model, optional) → IMAGE_DESCRIPTION chunks
- Table text (heuristic detection) → TABLE chunks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    images: list[dict[str, Any]] = field(default_factory=list)  # {bbox, xref, page}
    tables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    document_id: str
    title: str
    total_pages: int
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class PDFParser:
    """Parse PDF files into structured text, image refs, and table text."""

    def __init__(self, extract_images: bool = False, min_text_length: int = 10) -> None:
        self._extract_images = extract_images
        self._min_text_length = min_text_length

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF file synchronously. Returns a ParsedDocument."""
        import fitz  # PyMuPDF

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc = fitz.open(str(path))
        document_id = path.stem

        metadata = {
            "source_path": str(path),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
        }
        # Extract PDF metadata
        pdf_meta = doc.metadata or {}
        metadata.update({k: v for k, v in pdf_meta.items() if v})

        title = pdf_meta.get("title") or path.stem
        pages: list[ParsedPage] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            parsed_page = self._parse_page(page, page_num + 1)
            pages.append(parsed_page)

        doc.close()
        logger.info("Parsed PDF '%s': %d pages", path.name, len(pages))

        return ParsedDocument(
            document_id=document_id,
            title=title,
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
        )

    def _parse_page(self, page: Any, page_number: int) -> ParsedPage:
        import fitz

        # Extract text blocks
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_parts: list[str] = []
        tables: list[str] = []
        images: list[dict[str, Any]] = []

        for block in blocks:
            block_type = block[6]  # 0 = text, 1 = image
            if block_type == 0:
                text = block[4].strip()
                if len(text) >= self._min_text_length:
                    if self._looks_like_table(text):
                        tables.append(text)
                    else:
                        text_parts.append(text)
            elif block_type == 1 and self._extract_images:
                images.append(
                    {
                        "bbox": block[:4],
                        "page": page_number,
                        "xref": block[5],
                    }
                )

        full_text = "\n".join(text_parts)
        return ParsedPage(
            page_number=page_number,
            text=full_text,
            images=images,
            tables=tables,
            metadata={"page_number": page_number},
        )

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """Heuristic: if >30% of lines contain tab/pipe chars, treat as table."""
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False
        table_lines = sum(1 for l in lines if "\t" in l or "|" in l or "  " * 3 in l)
        return table_lines / len(lines) > 0.3
