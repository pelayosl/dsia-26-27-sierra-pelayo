"""Application configuration"""

from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).parents[1]
SQLITE_PREFIX = "sqlite:///"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"  # D13


class Settings(BaseSettings):
    """Immutable configuration.
    Secrets (`NVIDIA_API_KEY`) only come from environment or `.env`, never in code.
    """
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env", env_file_encoding="utf-8", frozen=True, extra="ignore"
    )

    dataset_path: str = "data/energy_weather_es_hourly.csv"
    database_url: str = "sqlite:///data/energy.db"
    model_path: Path = Path("models/model.joblib")
    reports_dir: Path = Path("reports")
    ai_provider: Literal["mock", "nvidia"] = "mock"
    # `SecretStr`: the key is shown as '**********' in reprs, logs and errors; `.get_secret_value()` reads it
    nvidia_api_key: SecretStr | None = None
    nvidia_model: str = DEFAULT_NVIDIA_MODEL
    log_level: str = "INFO"

    # Field validators run before the model is frozen, so they can return a transformed value.
    @field_validator("model_path", "reports_dir")
    @classmethod
    def resolve_path(cls, path: Path) -> Path:
        """Relative paths are resolved against `PROJECT_DIR`; absolute paths are kept."""
        return path if path.is_absolute() else PROJECT_DIR / path

    @field_validator("dataset_path")
    @classmethod
    def resolve_dataset_path(cls, location: str) -> str:
        """Same as `resolve_path`, but URLs are kept as they are."""
        if location.startswith(("http://", "https://")) or Path(location).is_absolute():
            return location
        return str(PROJECT_DIR / location)

    @field_validator("database_url")
    @classmethod
    def resolve_sqlite_path(cls, url: str) -> str:
        """Only relative SQLite URLs are rewritten; other databases (e.g. Postgres) are kept."""
        if not url.startswith(SQLITE_PREFIX):
            return url
        db_path = Path(url.removeprefix(SQLITE_PREFIX))
        if db_path.is_absolute():
            return url
        return f"{SQLITE_PREFIX}{(PROJECT_DIR / db_path).as_posix()}"

    # Mode "after" receives the already validated instance (`self`)
    @model_validator(mode="after")
    def check_nvidia_config(self) -> Self:
        """The real NVIDIA provider (D13) needs an API key and a model id."""
        if self.ai_provider == "nvidia":
            if self.nvidia_api_key is None or not self.nvidia_api_key.get_secret_value().strip():
                raise ValueError("NVIDIA_API_KEY is required when ai_provider is 'nvidia'")
            if not self.nvidia_model.strip():
                raise ValueError("NVIDIA_MODEL is required when ai_provider is 'nvidia'")
        return self


def load_settings() -> Settings:
    """Returns the application's configuration (single source of truth)."""
    return Settings()
