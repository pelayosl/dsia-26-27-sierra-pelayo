"""Shared fixtures. Everything in memory or in `tmp_path`: no network or absolute paths."""

import pytest

from energy_forecast.config import Settings


@pytest.fixture
def hourly_observations():
    """Synthetic DataFrame with ≥ `MIN_HISTORY_HOURS` + 48 consecutive hours and all `REQUIRED_COLUMNS`."""
    # TODO: UTC hourly grid; `load_mw` with a daily pattern (e.g. sine) so results are predictable.
    raise NotImplementedError


@pytest.fixture
def invalid_observations(hourly_observations):
    """Copy with invalid cases: duplicated `time_utc`, null and out-of-range `load_mw`, flag = 2."""
    # TODO: build from `hourly_observations`.
    raise NotImplementedError


@pytest.fixture
def weather_forecast_24h():
    """24-hour weather forecast with fixed values (`time_utc` + `WEATHER_COLUMNS`)."""
    # TODO: known constant values.
    raise NotImplementedError


@pytest.fixture
def small_trained_model(hourly_observations):
    """Small HGB model trained on `hourly_observations` (for loading and inference tests)."""
    # TODO: `build_training_frame` + `build_hgb_pipeline` with low `max_iter`.
    raise NotImplementedError


@pytest.fixture
def mock_settings(tmp_path):
    """`Settings` pointing to `tmp_path` (SQLite DB, model, reports) and `ai_provider="mock"`.

    Explicit arguments take precedence over the environment and `_env_file=None` ignores `.env`,
    so the machine's local configuration does not affect the tests.
    """
    return Settings(
        _env_file=None,
        dataset_path=str(tmp_path / "energy_weather_es_hourly.csv"),
        database_url=f"sqlite:///{(tmp_path / 'energy.db').as_posix()}",
        model_path=tmp_path / "models" / "model.joblib",
        reports_dir=tmp_path / "reports",
        ai_provider="mock",
        hf_token=None,
        hf_model=None,
    )
