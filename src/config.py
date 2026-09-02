"""Configuration loaded from environment (.env in dev, real env in prod-ish).

Strict validation at startup — a missing required key raises before Streamlit
serves a request. Model IDs are overridable via env.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key (Claude).")
    voyage_api_key: SecretStr = Field(..., description="Voyage AI API key (embeddings).")
    cohere_api_key: SecretStr = Field(..., description="Cohere API key (rerank).")

    database_url: str = Field(
        default="postgresql://lexgo:lexgo@localhost:5432/lexgo",
        description="Postgres+pgvector connection string; matches docker-compose defaults.",
    )

    # TODO(#2): verify these IDs against the provider SDKs at first API call.
    chat_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-sonnet-4-6"
    embedding_model: str = "voyage-3-large"
    rerank_model: str = "rerank-english-v3.0"

    log_level: str = "INFO"
    log_dir: Path = REPO_ROOT / "logs"
    llm_call_log: Path = REPO_ROOT / "logs" / "llm_calls.jsonl"
    evals_dir: Path = REPO_ROOT / "evals" / "runs"
    prompts_dir: Path = REPO_ROOT / "prompts"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Import and call — do not instantiate Settings directly."""
    return Settings()
