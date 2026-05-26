"""ProvenanceTracker — maps generated answer sentences back to source chunks.

Uses Jaccard word-overlap scoring to attribute each answer sentence to the
most relevant retrieved chunk. Lightweight and dependency-free.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from shared.models.document import ChunkType
from shared.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

# Common English stopwords to exclude from overlap scoring
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "with", "this", "that", "are",
        "was", "were", "be", "been", "has", "have", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "its", "their", "they", "we", "you", "he", "she", "as", "by",
        "from", "which", "who", "what", "when", "where", "how", "if",
        "also", "than", "then", "so", "up", "out", "no", "just", "into",
        "over", "after", "such", "these", "those", "both", "each",
    }
)


@dataclass
class ProvenanceRecord:
    """Attribution of one answer sentence to a source chunk."""

    sentence: str
    chunk_id: str
    document_id: str
    page_number: int | None
    chunk_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ProvenanceTracker:
    """Attribute answer sentences to the retrieved chunks that support them.

    For each sentence in the answer, computes Jaccard similarity against every
    source chunk and records the best-matching chunk above *min_confidence*.

    Args:
        min_confidence: Minimum Jaccard score (0–1) to include a record.
    """

    def __init__(self, min_confidence: float = 0.05) -> None:
        self._min_confidence = min_confidence

    def track(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> list[ProvenanceRecord]:
        """Map each answer sentence to its best matching source chunk.

        Args:
            answer: The generated answer text.
            sources: Retrieved chunks used as context during generation.

        Returns:
            List of ProvenanceRecord (one per sentence that exceeds threshold).
        """
        if not sources:
            return []

        sentences = self._split_sentences(answer)
        records: list[ProvenanceRecord] = []

        for sentence in sentences:
            if len(sentence.split()) < 3:
                continue
            record = self._best_match(sentence, sources)
            if record is not None:
                records.append(record)

        logger.debug(
            "Provenance: %d/%d sentences attributed, %d sources",
            len(records),
            len(sentences),
            len(sources),
        )
        return records

    def _best_match(
        self,
        sentence: str,
        sources: list[RetrievalResult],
    ) -> ProvenanceRecord | None:
        sentence_words = _tokenize(sentence)
        best_score = self._min_confidence
        best_source: RetrievalResult | None = None

        for source in sources:
            score = _jaccard(sentence_words, _tokenize(source.content))
            if score > best_score:
                best_score = score
                best_source = source

        if best_source is None:
            return None

        return ProvenanceRecord(
            sentence=sentence,
            chunk_id=best_source.chunk_id,
            document_id=best_source.document_id,
            page_number=best_source.metadata.get("page_number"),
            chunk_type=best_source.chunk_type,
            confidence=round(best_score, 4),
            metadata={
                "retrieval_score": best_source.score,
                **{k: v for k, v in best_source.metadata.items() if k != "page_number"},
            },
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in parts if s.strip()]


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-z][a-z0-9]*\b", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(set1: set[str], set2: set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)
