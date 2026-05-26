"""Pydantic Settings — loads all configuration from environment variables / .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── LLM API Keys ─────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", repr=False)
    openai_api_key: str = Field(default="", repr=False)

    # ── Tavily ───────────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", repr=False)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ragdb"
    postgres_user: str = "raguser"
    postgres_password: str = Field(default="ragpassword", repr=False)
    database_url: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = Field(default="", repr=False)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    semantic_cache_threshold: float = 0.92

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="neo4jpassword", repr=False)

    # ── LangFuse ─────────────────────────────────────────────────────────────
    langfuse_secret_key: str = Field(default="", repr=False)
    langfuse_public_key: str = Field(default="", repr=False)
    langfuse_host: str = "http://localhost:3000"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 100

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
