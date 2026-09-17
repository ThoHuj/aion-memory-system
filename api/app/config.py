"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated
from urllib.parse import urlparse, urlunparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def redact_database_url(database_url: str) -> str:
    """Return ``database_url`` with the password masked for safe logging."""
    parsed = urlparse(database_url)
    if not parsed.password:
        return database_url

    username = parsed.username or ""
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{username}:***@{hostname}{port}"
    return urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


class Settings(BaseSettings):
    """Runtime configuration for the API service."""

    model_config = SettingsConfigDict(
        env_file=("docker/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = ""
    cors_origins: Annotated[list[str], NoDecode] = []
    log_level: str = "info"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("database_url")
    @classmethod
    def require_database_url(cls, value: str) -> str:
        """Reject startup when DATABASE_URL is missing from the environment."""
        if not value.strip():
            raise ValueError("DATABASE_URL environment variable is required")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        """Parse comma-separated CORS origins from an environment variable."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return [str(origin) for origin in value]
        raise TypeError("CORS_ORIGINS must be a comma-separated string or list")

    @property
    def database_url_redacted(self) -> str:
        """Database URL with credentials masked, suitable for log output."""
        return redact_database_url(self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings loaded from the environment."""
    return Settings()
