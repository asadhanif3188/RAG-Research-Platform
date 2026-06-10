"""Qdrant quantization setup — create BQ and SQ collections, seed with vectors.

Three collection variants:
  - full_precision : no quantization, float32 vectors
  - binary_quantized (BQ): 1-bit per dimension — fastest, ~32x smaller
  - scalar_quantized (SQ): int8 per dimension — good quality/speed balance

Usage:
    from fastest_rag.quantization import QuantizationSetup, CollectionVariant

    setup = QuantizationSetup(host="localhost", port=6333)
    await setup.connect()
    await setup.create_all_collections(vector_size=3072)
    await setup.seed_synthetic(n_vectors=100_000, vector_size=3072)
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000  # vectors per upsert call


class CollectionVariant(StrEnum):
    FULL_PRECISION = "rag_full_precision"
    BINARY_QUANTIZED = "rag_binary_quantized"
    SCALAR_QUANTIZED = "rag_scalar_quantized"


@dataclass(frozen=True)
class SeedResult:
    variant: CollectionVariant
    n_vectors: int
    elapsed_seconds: float
    vectors_per_second: float


class QuantizationSetup:
    """Create and manage Qdrant collections with different quantization settings."""

    def __init__(self, host: str = "localhost", port: int = 6333, api_key: str = "") -> None:
        self._host = host
        self._port = port
        self._api_key = api_key or None
        self._client: Any = None

    async def connect(self) -> None:
        from qdrant_client import AsyncQdrantClient

        self._client = AsyncQdrantClient(host=self._host, port=self._port, api_key=self._api_key)
        logger.info("QuantizationSetup connected to Qdrant at %s:%d", self._host, self._port)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    # ── Collection creation ───────────────────────────────────────────────────

    async def create_all_collections(
        self, vector_size: int = 3072, recreate: bool = False
    ) -> dict[CollectionVariant, bool]:
        """Create all three collection variants. Returns {variant: created}."""
        results: dict[CollectionVariant, bool] = {}
        for variant in CollectionVariant:
            created = await self.create_collection(variant, vector_size, recreate)
            results[variant] = created
        return results

    async def create_collection(
        self,
        variant: CollectionVariant,
        vector_size: int = 3072,
        recreate: bool = False,
    ) -> bool:
        """Create a single collection. Returns True if created, False if already exists."""
        from qdrant_client.models import (
            BinaryQuantization,
            BinaryQuantizationConfig,
            Distance,
            ScalarQuantization,
            ScalarQuantizationConfig,
            ScalarType,
            VectorParams,
        )

        existing = {c.name for c in (await self._client.get_collections()).collections}

        if variant.value in existing:
            if not recreate:
                logger.info("Collection '%s' already exists, skipping", variant.value)
                return False
            await self._client.delete_collection(variant.value)

        vector_params = VectorParams(size=vector_size, distance=Distance.COSINE)

        if variant == CollectionVariant.BINARY_QUANTIZED:
            quantization_config = BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            )
        elif variant == CollectionVariant.SCALAR_QUANTIZED:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        else:
            quantization_config = None

        await self._client.create_collection(
            collection_name=variant.value,
            vectors_config=vector_params,
            quantization_config=quantization_config,
        )
        logger.info("Created Qdrant collection '%s' (vector_size=%d)", variant.value, vector_size)
        return True

    # ── Synthetic data seeding ────────────────────────────────────────────────

    async def seed_synthetic(
        self,
        n_vectors: int = 100_000,
        vector_size: int = 3072,
        variants: list[CollectionVariant] | None = None,
        seed: int = 42,
    ) -> list[SeedResult]:
        """Seed collections with random unit vectors for benchmarking.

        Args:
            n_vectors: Number of vectors to insert per collection.
            vector_size: Dimensionality of each vector.
            variants: Which collections to seed. Defaults to all three.
            seed: Random seed for reproducibility.
        """
        if variants is None:
            variants = list(CollectionVariant)

        random.seed(seed)
        results: list[SeedResult] = []

        for variant in variants:
            logger.info("Seeding '%s' with %d vectors...", variant.value, n_vectors)
            start = time.perf_counter()

            for batch_start in range(0, n_vectors, _BATCH_SIZE):
                batch_size = min(_BATCH_SIZE, n_vectors - batch_start)
                points = self._make_points(batch_start, batch_size, vector_size)
                await self._client.upsert(collection_name=variant.value, points=points, wait=False)

            elapsed = time.perf_counter() - start
            vps = n_vectors / elapsed

            logger.info(
                "Seeded '%s': %d vectors in %.2fs (%.0f vec/s)",
                variant.value,
                n_vectors,
                elapsed,
                vps,
            )
            results.append(
                SeedResult(
                    variant=variant,
                    n_vectors=n_vectors,
                    elapsed_seconds=elapsed,
                    vectors_per_second=vps,
                )
            )

        return results

    @staticmethod
    def _make_points(start_id: int, count: int, vector_size: int) -> list[Any]:
        """Generate random normalised vectors as PointStruct objects."""
        from qdrant_client.models import PointStruct

        points: list[PointStruct] = []
        for i in range(count):
            vec = [random.gauss(0, 1) for _ in range(vector_size)]
            norm = math.sqrt(sum(x * x for x in vec))
            vec = [x / (norm + 1e-9) for x in vec]
            points.append(
                PointStruct(
                    id=start_id + i,
                    vector=vec,
                    payload={"synthetic": True, "batch_id": start_id // _BATCH_SIZE},
                )
            )
        return points

    async def collection_info(self, variant: CollectionVariant) -> dict[str, Any]:
        """Return point count and config summary for a collection."""
        info = await self._client.get_collection(variant.value)
        return {
            "name": variant.value,
            "points_count": info.points_count,
            "vectors_size": info.config.params.vectors.size,
            "quantization": (
                type(info.config.quantization_config).__name__
                if info.config.quantization_config
                else "none"
            ),
        }
