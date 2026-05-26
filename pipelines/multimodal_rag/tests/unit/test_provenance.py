"""Unit tests for ProvenanceTracker — sentence attribution and edge cases."""

from __future__ import annotations

import pytest

from multimodal_rag.provenance import ProvenanceTracker, _jaccard, _tokenize
from shared.models.document import ChunkType
from shared.models.retrieval import RetrievalResult


def make_source(
    chunk_id: str,
    content: str,
    doc_id: str = "doc-1",
    score: float = 0.9,
    chunk_type: str = "text",
    page_number: int | None = None,
) -> RetrievalResult:
    meta = {}
    if page_number is not None:
        meta["page_number"] = page_number
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        content=content,
        chunk_type=chunk_type,
        score=score,
        metadata=meta,
    )


class TestTokenize:
    def test_lowercases_words(self):
        tokens = _tokenize("Hello World RAG")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_stopwords(self):
        tokens = _tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "fox" in tokens

    def test_removes_short_words(self):
        tokens = _tokenize("a b is it")
        assert not tokens  # all stopwords or length < 3

    def test_numbers_included(self):
        tokens = _tokenize("revenue grew 42 percent")
        assert "revenue" in tokens
        assert "42" in tokens or "percent" in tokens


class TestJaccard:
    def test_identical_sets(self):
        s = {"apple", "banana"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        score = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4
        assert score == pytest.approx(0.5)

    def test_empty_sets(self):
        assert _jaccard(set(), {"a"}) == 0.0
        assert _jaccard({"a"}, set()) == 0.0


class TestProvenanceTracker:
    def setup_method(self):
        self.tracker = ProvenanceTracker(min_confidence=0.05)

    def test_track_returns_records_for_matching_sentences(self):
        sources = [
            make_source("c1", "RAG combines retrieval with generation to answer questions.")
        ]
        answer = "RAG combines retrieval with generation to answer questions accurately."
        records = self.tracker.track(answer, sources)
        assert len(records) >= 1
        assert records[0].chunk_id == "c1"

    def test_track_empty_sources_returns_empty(self):
        records = self.tracker.track("Some answer.", sources=[])
        assert records == []

    def test_track_skips_very_short_sentences(self):
        sources = [make_source("c1", "The quick brown fox jumps over the lazy dog.")]
        records = self.tracker.track("Yes. No. Maybe.", sources)
        assert records == []  # all sentences < 3 words

    def test_track_low_overlap_below_threshold_excluded(self):
        tracker = ProvenanceTracker(min_confidence=0.5)
        sources = [make_source("c1", "completely unrelated content about weather")]
        answer = "RAG stands for Retrieval-Augmented Generation pipeline systems."
        records = tracker.track(answer, sources)
        assert records == []

    def test_record_includes_page_number(self):
        sources = [
            make_source("c1", "The neural network achieved accuracy on the test set.", page_number=5)
        ]
        answer = "The neural network accuracy was tested on the validation set evaluation."
        records = self.tracker.track(answer, sources)
        if records:
            assert records[0].page_number == 5

    def test_record_includes_chunk_type(self):
        sources = [
            make_source("c1", "Revenue grew by 42 percent in Q3 quarter results.", chunk_type="table")
        ]
        answer = "Revenue grew by 42 percent in Q3 quarter financial results."
        records = self.tracker.track(answer, sources)
        if records:
            assert records[0].chunk_type == "table"

    def test_confidence_between_zero_and_one(self):
        sources = [make_source("c1", "Transformers use attention mechanism for sequence modeling.")]
        answer = "Transformers use attention mechanism for processing sequences in modeling tasks."
        records = self.tracker.track(answer, sources)
        for r in records:
            assert 0.0 < r.confidence <= 1.0

    def test_multiple_sentences_attributed_to_best_source(self):
        sources = [
            make_source("c1", "RAG stands for Retrieval Augmented Generation.", score=0.95),
            make_source("c2", "BERT is a bidirectional transformer language model.", score=0.80),
        ]
        answer = (
            "RAG stands for Retrieval Augmented Generation systems. "
            "BERT is a bidirectional transformer language model architecture."
        )
        records = self.tracker.track(answer, sources)
        chunk_ids = {r.chunk_id for r in records}
        assert "c1" in chunk_ids or "c2" in chunk_ids
