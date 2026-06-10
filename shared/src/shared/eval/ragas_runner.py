"""RAGAS evaluation harness.

Computes four metrics for a set of query/answer/context triples:
  - faithfulness         (is the answer grounded in the retrieved context?)
  - answer_relevancy     (is the answer relevant to the question?)
  - context_precision    (are the retrieved contexts relevant to the question?)
  - context_recall       (do the contexts cover the ground truth answer?)

Requires `ground_truth` only for context_recall.
"""

from __future__ import annotations

import logging
from typing import Any

from shared.models.retrieval import EvalResult

logger = logging.getLogger(__name__)


class RAGASRunner:
    """Run RAGAS metrics using the ragas library with a LangChain LLM backend."""

    def __init__(
        self,
        llm: Any | None = None,
        embeddings: Any | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        """
        Args:
            llm: A LangChain-compatible LLM (e.g. ChatAnthropic). If None, uses claude-haiku.
            embeddings: A LangChain Embeddings instance for context_recall. If None, uses OpenAI.
            anthropic_api_key: Passed to ChatAnthropic when `llm` is None.
            openai_api_key: Passed to OpenAIEmbeddings when `embeddings` is None.
        """
        self._llm = llm
        self._embeddings = embeddings
        self._anthropic_api_key = anthropic_api_key
        self._openai_api_key = openai_api_key
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if self._llm is None:
            from langchain_anthropic import ChatAnthropic
            from pydantic import SecretStr

            self._llm = ChatAnthropic(  # type: ignore[call-arg]
                model_name="claude-haiku-4-5-20251001",
                anthropic_api_key=SecretStr(self._anthropic_api_key or ""),
            )

        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            from pydantic import SecretStr

            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=SecretStr(self._openai_api_key or ""),
            )

        self._initialized = True

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        pipeline: str | None = None,
    ) -> EvalResult:
        """Evaluate a single QA pair. Returns an EvalResult with metric scores."""
        self._ensure_initialized()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        metrics = [faithfulness, answer_relevancy, context_precision]

        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,
            )
            scores = result.to_pandas().iloc[0].to_dict()  # type: ignore[union-attr]
            logger.info(
                "RAGAS scores — faithfulness=%.3f, answer_relevancy=%.3f, "
                "context_precision=%.3f, context_recall=%s",
                scores.get("faithfulness", -1),
                scores.get("answer_relevancy", -1),
                scores.get("context_precision", -1),
                f"{scores['context_recall']:.3f}" if "context_recall" in scores else "N/A",
            )
        except Exception:
            logger.exception("RAGAS evaluation failed; returning empty scores")
            scores = {}

        return EvalResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
            pipeline=pipeline,
        )

    async def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
        pipeline: str | None = None,
    ) -> list[EvalResult]:
        """Evaluate a batch of samples.

        Each sample dict must have keys: query, answer, contexts.
        Optionally: ground_truth.
        """
        self._ensure_initialized()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        has_gt = all("ground_truth" in s for s in samples)

        data: dict[str, list[Any]] = {
            "question": [s["query"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
        }
        if has_gt:
            data["ground_truth"] = [s["ground_truth"] for s in samples]

        metrics = [faithfulness, answer_relevancy, context_precision]
        if has_gt:
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,
            )
            df = result.to_pandas()  # type: ignore[union-attr]
        except Exception:
            logger.exception("RAGAS batch evaluation failed; returning empty results")
            return [
                EvalResult(
                    query=s["query"],
                    answer=s["answer"],
                    contexts=s["contexts"],
                    ground_truth=s.get("ground_truth"),
                    pipeline=pipeline,
                )
                for s in samples
            ]

        results: list[EvalResult] = []
        for i, sample in enumerate(samples):
            row = df.iloc[i].to_dict()
            results.append(
                EvalResult(
                    query=sample["query"],
                    answer=sample["answer"],
                    contexts=sample["contexts"],
                    ground_truth=sample.get("ground_truth"),
                    faithfulness=row.get("faithfulness"),
                    answer_relevancy=row.get("answer_relevancy"),
                    context_precision=row.get("context_precision"),
                    context_recall=row.get("context_recall"),
                    pipeline=pipeline,
                )
            )

        return results
