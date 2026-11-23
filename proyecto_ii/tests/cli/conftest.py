"""Fixtures shared by the CLI tests (`test_cli.py`, `test_pipeline.py`)."""

import pytest

from energy_forecast import bootstrap
from energy_forecast.domain import modeling
from tests.fakes import FakeWeatherForecastClient


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Points the whole configuration to `tmp_path` through environment variables (they override `.env`)."""
    monkeypatch.setenv("DATASET_PATH", str(tmp_path / "energy_weather_es_hourly.csv"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'energy.db').as_posix()}")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "models" / "model.joblib"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_PROVIDER", "mock")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    return tmp_path


@pytest.fixture
def fast_hgb(monkeypatch):
    """HGB with few trees, so the real training runs in under a second."""
    monkeypatch.setitem(modeling.HGB_PARAMS, "max_iter", 20)


@pytest.fixture
def fake_weather(monkeypatch, weather_forecast_24h):
    """Replaces Open-Meteo in the composition root: the CLI tests never use the network."""
    fake = FakeWeatherForecastClient(weather_forecast_24h)
    monkeypatch.setattr(bootstrap, "OpenMeteoForecastClient", lambda: fake)
    return fake
