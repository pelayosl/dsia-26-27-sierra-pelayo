"""Tests for `config` (pydantic-settings)."""

import pytest
from pydantic import ValidationError

from energy_forecast.config import DEFAULT_NVIDIA_MODEL, PROJECT_DIR, Settings, load_settings

ENV_VARS = [
    "DATASET_PATH",
    "DATABASE_URL",
    "MODEL_PATH",
    "REPORTS_DIR",
    "AI_PROVIDER",
    "NVIDIA_API_KEY",
    "NVIDIA_MODEL",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Removes the project's variables from the environment so the machine's setup doesn't leak into the tests."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_use_mock_provider():
    # Arrange / Act
    settings = Settings(_env_file=None)

    # Assert
    assert settings.ai_provider == "mock"
    assert settings.nvidia_api_key is None
    assert settings.nvidia_model == DEFAULT_NVIDIA_MODEL
    assert settings.log_level == "INFO"


def test_reads_values_from_environment(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "owner/model")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.ai_provider == "nvidia"
    assert settings.nvidia_api_key.get_secret_value() == "nvapi-test-key"
    assert settings.nvidia_model == "owner/model"
    assert settings.log_level == "DEBUG"


def test_api_key_is_hidden_in_repr_and_errors():
    settings = Settings(_env_file=None, ai_provider="nvidia", nvidia_api_key="nvapi-secret-value")

    assert "nvapi-secret-value" not in repr(settings)
    assert "nvapi-secret-value" not in str(settings.model_dump())


def test_unknown_ai_provider_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_provider="openai")


def test_hf_provider_is_no_longer_accepted():
    # D13: the real provider is NVIDIA; Hugging Face Inference Providers is not used
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_provider="hf")


@pytest.mark.parametrize(
    ("api_key", "model", "message"),
    [(None, "owner/model", "NVIDIA_API_KEY"), ("  ", "owner/model", "NVIDIA_API_KEY"), ("nvapi-x", " ", "NVIDIA_MODEL")],
    ids=["no-key", "blank-key", "blank-model"],
)
def test_nvidia_provider_requires_key_and_model(api_key, model, message):
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None, ai_provider="nvidia", nvidia_api_key=api_key, nvidia_model=model)


def test_relative_paths_are_resolved_against_project_dir():
    settings = Settings(_env_file=None)

    assert settings.model_path == PROJECT_DIR / "models" / "model.joblib"
    assert settings.reports_dir == PROJECT_DIR / "reports"
    assert settings.model_path.is_absolute()


def test_absolute_paths_are_kept(tmp_path):
    settings = Settings(_env_file=None, model_path=tmp_path / "model.joblib", reports_dir=tmp_path)

    assert settings.model_path == tmp_path / "model.joblib"
    assert settings.reports_dir == tmp_path


def test_dataset_url_is_not_treated_as_a_path():
    url = "https://example.org/energy_weather_es_hourly.csv"

    settings = Settings(_env_file=None, dataset_path=url)

    assert settings.dataset_path == url


def test_relative_sqlite_url_points_inside_project_dir():
    settings = Settings(_env_file=None, database_url="sqlite:///data/energy.db")

    assert settings.database_url.startswith("sqlite:///")
    assert settings.database_url.endswith("energy.db")
    assert PROJECT_DIR.as_posix() in settings.database_url.replace("\\", "/")


def test_non_sqlite_database_url_is_kept():
    url = "postgresql://user:password@localhost:5432/energy"

    settings = Settings(_env_file=None, database_url=url)

    assert settings.database_url == url


def test_settings_are_immutable():
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.ai_provider = "nvidia"


def test_load_settings_returns_settings(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "mock")

    assert isinstance(load_settings(), Settings)


def test_mock_settings_fixture_points_to_tmp_path(mock_settings, tmp_path):
    assert mock_settings.ai_provider == "mock"
    assert mock_settings.model_path.is_relative_to(tmp_path)
    assert mock_settings.reports_dir.is_relative_to(tmp_path)
    assert tmp_path.as_posix() in mock_settings.database_url.replace("\\", "/")
