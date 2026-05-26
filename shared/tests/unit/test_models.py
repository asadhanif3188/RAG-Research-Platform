"""Unit tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from shared.models.document import ChunkType, DocumentChunk
from shared.models.query import PipelineStrategy, QueryRequest, QueryResponse
from shared.models.retrieval import EvalResult, RetrievalResult


class TestDocumentChunk:
    def test_default_id_is_uuid(self):
        chunk = DocumentChunk(document_id="doc-1", chunk_index=0, content="hello")
        assert len(chunk.id) == 36  # UUID4 format

    def test_chunk_type_default_text(self):
        chunk = DocumentChunk(document_id="doc-1", chunk_index=0, content="hello")
        assert chunk.chunk_type == ChunkType.TEXT

    def test_whitespace_only_content_rejected(self):
        with pytest.raises(ValidationError):
            DocumentChunk(document_id="doc-1", chunk_index=0, content="   ")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            DocumentChunk(document_id="doc-1", chunk_index=0, content="")

    def test_negative_chunk_index_rejected(self):
        with pytest.raises(ValidationError):
            DocumentChunk(document_id="doc-1", chunk_index=-1, content="hello")

    def test_token_count_estimate(self):
        chunk = DocumentChunk(
            document_id="doc-1",
            chunk_index=0,
            content="one two three four five",  # 5 words
        )
        assert chunk.token_count() == int(5 * 1.3)  # == 6

    def test_all_chunk_types_valid(self):
        for ct in ChunkType:
            chunk = DocumentChunk(document_id="d", chunk_index=0, content="x", chunk_type=ct)
            assert chunk.chunk_type == ct


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(query="What is RAG?")
        assert req.top_k == 5
        assert req.use_cache is True
        assert req.pipeline == PipelineStrategy.FASTEST_RAG

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="q", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="q", top_k=51)


class TestRetrievalResult:
    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            RetrievalResult(
                chunk_id="c1", document_id="d1", content="x", score=1.5
            )
        with pytest.raises(ValidationError):
            RetrievalResult(
                chunk_id="c1", document_id="d1", content="x", score=-0.1
            )

    def test_valid_result(self):
        r = RetrievalResult(
            chunk_id="c1", document_id="d1", content="hello", score=0.92
        )
        assert r.score == 0.92


class TestEvalResult:
    def test_aggregate_score_all_set(self):
        er = EvalResult(
            query="q",
            answer="a",
            contexts=["c1"],
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.6,
        )
        assert abs(er.aggregate_score() - 0.75) < 1e-6

    def test_aggregate_score_partial(self):
        er = EvalResult(
            query="q",
            answer="a",
            contexts=["c1"],
            faithfulness=0.8,
            answer_relevancy=0.9,
        )
        assert abs(er.aggregate_score() - 0.85) < 1e-6

    def test_aggregate_score_none_when_no_metrics(self):
        er = EvalResult(query="q", answer="a", contexts=[])
        assert er.aggregate_score() is None
