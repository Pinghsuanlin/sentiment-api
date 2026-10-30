"""Serving configuration from environment variables (prefix ``SENTIMENT_``) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the API and the prediction layer."""

    model_config = SettingsConfigDict(
        env_prefix="SENTIMENT_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    predictor: Literal["distilbert", "baseline"] = "distilbert"
    model_path: Path = Path("models/distilbert-imdb/final")
    baseline_path: Path = Path("models/baseline.joblib")
    model_version: str = "dev"
    max_len: int = Field(128, ge=16, le=512)
    max_text_chars: int = Field(10_000, ge=1)
    max_batch_size: int = Field(32, ge=1, le=256)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
