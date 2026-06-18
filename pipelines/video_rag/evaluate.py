"""Evaluation script for Video RAG pipeline.

Measures retrieval accuracy (timestamp precision) and answer quality
against a test dataset of video queries.

Usage:
    uv run python pipelines/video_rag/evaluate.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single evaluation case."""

    query: str
    expected_video_id: str
    expected_start_ts: float
    expected_end_ts: float
    expected_answer_contains: list[str]


@dataclass
class EvalResult:
    """Evaluation result for a single case."""

    query: str
    timestamp_error_s: float
    answer_contains_match: bool
    retrieved_video_correct: bool


def load_eval_dataset(path: str | None = None) -> list[EvalCase]:
    """Load evaluation cases from JSON."""
    if path:
        with open(path) as f:
            data = json.load(f)
        return [EvalCase(**case) for case in data]

    # Default synthetic test cases
    return [
        EvalCase(
            query="What is machine learning?",
            expected_video_id="ai-lecture-101",
            expected_start_ts=120.0,
            expected_end_ts=180.0,
            expected_answer_contains=["machine learning", "data"],
        ),
        EvalCase(
            query="How does backpropagation work?",
            expected_video_id="ai-lecture-101",
            expected_start_ts=600.0,
            expected_end_ts=720.0,
            expected_answer_contains=["gradient", "loss"],
        ),
        EvalCase(
            query="What are convolutional neural networks?",
            expected_video_id="cv-workshop",
            expected_start_ts=300.0,
            expected_end_ts=420.0,
            expected_answer_contains=["convolution", "filter"],
        ),
    ]


def evaluate_timestamp_accuracy(
    retrieved_start: float,
    retrieved_end: float,
    expected_start: float,
    expected_end: float,
) -> float:
    """Compute timestamp error in seconds (overlap-based)."""
    overlap_start = max(retrieved_start, expected_start)
    overlap_end = min(retrieved_end, expected_end)
    overlap = max(0.0, overlap_end - overlap_start)

    if overlap > 0:
        return 0.0  # Timestamps overlap — within tolerance

    # Minimum distance between the two intervals
    if retrieved_end < expected_start:
        return expected_start - retrieved_end
    return retrieved_start - expected_end


def evaluate_answer_quality(answer: str, expected_contains: list[str]) -> bool:
    """Check if the answer contains expected key terms."""
    answer_lower = answer.lower()
    return all(term.lower() in answer_lower for term in expected_contains)


def main() -> None:
    """Run evaluation (prints results — full pipeline integration required)."""
    cases = load_eval_dataset()

    print("=" * 60)
    print("Video RAG Evaluation")
    print("=" * 60)
    print(f"\nLoaded {len(cases)} evaluation cases\n")

    for i, case in enumerate(cases, 1):
        print(f"Case {i}: {case.query}")
        print(f"  Expected video: {case.expected_video_id}")
        print(f"  Expected timestamp: {case.expected_start_ts:.0f}s - {case.expected_end_ts:.0f}s")
        print(f"  Expected terms: {case.expected_answer_contains}")
        print()

    print("To run full evaluation, integrate with the pipeline:")
    print("  1. Index videos into pgvector + Neo4j")
    print("  2. Run queries through VideoRAGPipeline")
    print("  3. Compare retrieved timestamps and answers")
    print("\nTimestamp accuracy metric: overlap-based error (0 = within range)")
    print("Answer quality metric: all expected terms present in answer")


if __name__ == "__main__":
    main()
