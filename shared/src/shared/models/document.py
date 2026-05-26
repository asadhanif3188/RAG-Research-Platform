"""Document and chunk Pydantic models."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChunkType(StrEnum):
    TEXT = "text"
    IMAGE_DESCRIPTION = "image_description"
    TABLE = "table"
    VIDEO_TRANSCRIPT = "video_transcript"


class DocumentChunk(BaseModel):
    """A single chunk extracted from a source document.

    Stored in both the vector DB (embedding) and relational DB (full text).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int = Field(ge=0)
    chunk_type: ChunkType = ChunkType.TEXT
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("content")
    @classmethod
    def content_not_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be whitespace-only")
        return v

    def token_count(self) -> int:
        """Rough token estimate (words × 1.3 heuristic — no tiktoken dependency needed here)."""
        return int(len(self.content.split()) * 1.3)

    model_config = {"validate_assignment": True}
