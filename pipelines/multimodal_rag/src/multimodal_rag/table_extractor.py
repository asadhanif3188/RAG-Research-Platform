"""TableExtractor — detect and extract tables from PDFs as Markdown using pdfplumber.

Primary strategy: pdfplumber structured extraction (accurate, handles complex layouts).
Fallback: heuristic text-line detection from PyMuPDF block text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    """A single table extracted from a PDF page."""

    page_number: int
    markdown: str
    raw_data: list[list[str]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.markdown.strip():
            raise ValueError("ExtractedTable requires non-empty markdown")


class TableExtractor:
    """Extract tables from PDF files as Markdown using pdfplumber.

    Args:
        min_rows: Minimum number of rows for a region to be treated as a table.
        min_cols: Minimum number of columns for a region to be treated as a table.
    """

    def __init__(self, min_rows: int = 2, min_cols: int = 2) -> None:
        self._min_rows = min_rows
        self._min_cols = min_cols

    def extract_from_pdf(self, pdf_path: str | Path) -> list[ExtractedTable]:
        """Extract all tables from a PDF file.

        Returns an empty list if pdfplumber is unavailable or extraction fails.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            return self._extract_with_pdfplumber(path)
        except ImportError:
            logger.warning("pdfplumber not installed — table extraction disabled")
            return []
        except Exception as exc:
            logger.warning("pdfplumber extraction failed for '%s': %s", path.name, exc)
            return []

    def extract_from_page(self, pdf_path: str | Path, page_number: int) -> list[ExtractedTable]:
        """Extract tables from a specific page (1-indexed).

        Returns an empty list on any error.
        """
        path = Path(pdf_path)
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return []
                page = pdf.pages[page_number - 1]
                return self._extract_tables_from_page(page, page_number)
        except Exception as exc:
            logger.warning("Table extraction failed for page %d: %s", page_number, exc)
            return []

    def text_to_markdown(self, table_data: list[list[str | None]]) -> str:
        """Convert a 2D list of cell values to a Markdown table string.

        Args:
            table_data: Rows × columns of cell text (None treated as empty).

        Returns:
            Markdown table string, or empty string if data is too small.
        """
        if not table_data or len(table_data) < self._min_rows:
            return ""

        # Normalise: replace None, strip whitespace
        rows: list[list[str]] = [
            [str(cell).strip() if cell is not None else "" for cell in row] for row in table_data
        ]

        col_count = max(len(row) for row in rows)
        if col_count < self._min_cols:
            return ""

        # Pad rows to uniform width
        rows = [row + [""] * (col_count - len(row)) for row in rows]

        col_widths = [
            max(max(len(rows[r][c]) for r in range(len(rows))), 3) for c in range(col_count)
        ]

        def fmt_row(row: list[str]) -> str:
            return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

        lines = [fmt_row(rows[0])]
        lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
        for row in rows[1:]:
            lines.append(fmt_row(row))

        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_with_pdfplumber(self, path: Path) -> list[ExtractedTable]:
        import pdfplumber

        tables: list[ExtractedTable] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_tables = self._extract_tables_from_page(page, page.page_number)
                tables.extend(page_tables)

        logger.info("Extracted %d tables from '%s'", len(tables), path.name)
        return tables

    def _extract_tables_from_page(self, page: Any, page_number: int) -> list[ExtractedTable]:
        extracted: list[ExtractedTable] = []
        try:
            raw_tables = page.extract_tables()
            for table_data in raw_tables:
                if not table_data:
                    continue
                non_empty = [
                    row for row in table_data if any(c for c in row if c and str(c).strip())
                ]
                if len(non_empty) < self._min_rows:
                    continue
                if not table_data[0] or len(table_data[0]) < self._min_cols:
                    continue

                markdown = self.text_to_markdown(table_data)
                if markdown:
                    extracted.append(
                        ExtractedTable(
                            page_number=page_number,
                            markdown=markdown,
                            raw_data=table_data,
                        )
                    )
        except Exception as exc:
            logger.debug("Table parse error on page %d: %s", page_number, exc)

        return extracted
