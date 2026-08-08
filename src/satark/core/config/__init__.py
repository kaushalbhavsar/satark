"""Configuration loading for SATARK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SatarkSettings(BaseSettings):
    """Application settings loaded from environment and optional config files."""

    model_config = SettingsConfigDict(
        env_prefix="SATARK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    enable_ai: bool = False
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    plugins: list[str] = Field(default_factory=list)
    knowledge_version: str = "latest"

    def ensure_data_dir(self) -> Path:
        """Create the data directory if missing and return it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


def load_settings(**overrides: Any) -> SatarkSettings:
    """Load settings with optional keyword overrides."""
    return SatarkSettings(**overrides)
