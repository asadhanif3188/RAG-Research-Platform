"""Unit tests for TableExtractor — text_to_markdown and edge-case handling."""

from __future__ import annotations

import pytest

from multimodal_rag.table_extractor import ExtractedTable, TableExtractor


class TestTableExtractorTextToMarkdown:
    def setup_method(self):
        self.extractor = TableExtractor(min_rows=2, min_cols=2)

    def test_basic_table(self):
        data = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md = self.extractor.text_to_markdown(data)
        assert "| Name" in md
        assert "| Age" in md
        assert "| Alice" in md
        assert "|---|" in md

    def test_header_separator_present(self):
        data = [["A", "B"], ["1", "2"]]
        md = self.extractor.text_to_markdown(data)
        lines = md.splitlines()
        assert any("---" in line for line in lines)

    def test_none_cells_treated_as_empty(self):
        data = [["Col1", None], ["val", "x"]]
        md = self.extractor.text_to_markdown(data)
        assert md  # should not raise

    def test_returns_empty_for_single_row(self):
        data = [["Header1", "Header2"]]
        md = self.extractor.text_to_markdown(data)
        assert md == ""

    def test_returns_empty_for_single_column(self):
        data = [["A"], ["1"], ["2"]]
        md = self.extractor.text_to_markdown(data)
        assert md == ""

    def test_returns_empty_for_empty_input(self):
        assert self.extractor.text_to_markdown([]) == ""

    def test_uneven_rows_padded(self):
        data = [["A", "B", "C"], ["1", "2"]]
        md = self.extractor.text_to_markdown(data)
        assert md  # should handle without error

    def test_column_alignment_consistent(self):
        data = [["Short", "A very long column header"], ["x", "y"]]
        md = self.extractor.text_to_markdown(data)
        lines = md.splitlines()
        col_counts = [line.count("|") for line in lines if "|" in line]
        assert len(set(col_counts)) == 1, "All rows should have same number of | characters"


class TestExtractedTable:
    def test_valid_creation(self):
        t = ExtractedTable(page_number=1, markdown="| A | B |\n|---|---|\n| 1 | 2 |")
        assert t.page_number == 1
        assert "A" in t.markdown

    def test_empty_markdown_raises(self):
        with pytest.raises(ValueError):
            ExtractedTable(page_number=1, markdown="   ")


class TestTableExtractorPdfExtraction:
    def test_extract_from_nonexistent_pdf(self):
        extractor = TableExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_pdf("/nonexistent/path/doc.pdf")

    def test_extract_from_nonexistent_page(self, tmp_path):
        extractor = TableExtractor()
        # Create a minimal placeholder file; pdfplumber will fail gracefully
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 not a real pdf")
        result = extractor.extract_from_page(str(fake_pdf), page_number=99)
        assert result == []

    def test_extract_missing_pdfplumber_returns_empty(self, tmp_path, monkeypatch):
        extractor = TableExtractor()
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"dummy")

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pdfplumber":
                raise ImportError("pdfplumber not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = extractor.extract_from_pdf(str(fake_pdf))
        assert result == []
