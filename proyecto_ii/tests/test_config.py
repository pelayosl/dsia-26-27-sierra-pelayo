"""Tests for `config` (pydantic-settings)."""

import pytest
from pydantic import ValidationError

from energy_forecast.config import PROJECT_DIR, Settings, load_settings

ENV_VARS = [
    "DATASET_PATH",
    "DATABASE_URL",
    "MODEL_PATH",
    "REPORTS_DIR",
    "AI_PROVIDER",
    "HF_TOKEN",
    "HF_MODEL",
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
    assert settings.hf_token is None
    assert settings.log_level == "INFO"


def test_reads_values_from_environment(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "hf")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("HF_MODEL", "owner/model")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.ai_provider == "hf"
    assert settings.hf_token == "hf_test_token"
    assert settings.hf_model == "owner/model"
    assert settings.log_level == "DEBUG"


def test_unknown_ai_provider_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_provider="openai")


@pytest.mark.parametrize(
    "hf_token, hf_model",
    [(None, "owner/model"), ("hf_test_token", None), (None, None)],
)
def test_hf_provider_requires_token_and_model(hf_token, hf_model):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_provider="hf", hf_token=hf_token, hf_model=hf_model)


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
        settings.ai_provider = "hf"


def test_load_settings_returns_settings(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "mock")

    assert isinstance(load_settings(), Settings)


def test_mock_settings_fixture_points_to_tmp_path(mock_settings, tmp_path):
    assert mock_settings.ai_provider == "mock"
    assert mock_settings.model_path.is_relative_to(tmp_path)
    assert mock_settings.reports_dir.is_relative_to(tmp_path)
    assert tmp_path.as_posix() in mock_settings.database_url.replace("\\", "/")
