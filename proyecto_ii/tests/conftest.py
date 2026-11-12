"""Shared fixtures. Everything in memory or in `tmp_path`: no network or absolute paths."""

import logging

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import Settings
from energy_forecast.domain.features import build_training_frame
from energy_forecast.domain.modeling import build_hgb_pipeline
from energy_forecast.domain.validator import validate_observations
from energy_forecast.domain.schema import FLAG_COLUMNS, HORIZON_HOURS, LOCAL_TIMEZONE, MIN_HISTORY_HOURS

OBSERVATIONS_START = "2025-01-06T00:00:00+00:00"  # a Monday, outside any DST change
OBSERVATIONS_HOURS = MIN_HISTORY_HOURS + 48
# Last hour of `hourly_observations`: origin of a forecast right after the observations
FORECAST_ORIGIN = pd.Timestamp(OBSERVATIONS_START) + pd.Timedelta(hours=OBSERVATIONS_HOURS - 1)
BASE_LOAD_MW = 30_000.0
DAILY_AMPLITUDE_MW = 5_000.0

# Row positions with invalid values in `invalid_observations`
DUPLICATED_ROWS = (9, 10)
NULL_LOAD_ROW = 20
OUT_OF_RANGE_LOAD_ROW = 30
INVALID_FLAG_ROW = 40


def make_raw_observations(start: str, hours: int, noise_mw: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic observations in the dsia-dataset CSV format (dates as ISO 8601 strings, all `REQUIRED_COLUMNS`).

    `load_mw` follows a daily sine pattern, plus Gaussian noise of `noise_mw` if given.
    """
    time_utc = pd.date_range(start, periods=hours, freq="h")
    hour = time_utc.hour.to_numpy()
    noise = np.random.default_rng(seed).normal(0.0, noise_mw, hours) if noise_mw else 0.0
    frame = pd.DataFrame(
        {
            "time_utc": time_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "time_local": time_utc.tz_convert(LOCAL_TIMEZONE).map(lambda t: t.isoformat()),
            "load_mw": BASE_LOAD_MW + DAILY_AMPLITUDE_MW * np.sin(2 * np.pi * hour / 24) + noise,
            "temperature_2m": 10.0 + 5.0 * np.sin(2 * np.pi * (hour - 9) / 24),
            "relative_humidity_2m": 60.0,
            "wind_speed_10m": 10.0,
            "shortwave_radiation": np.where((hour >= 7) & (hour <= 17), 300.0, 0.0),
            "precipitation": 0.0,
        }
    )
    for column in FLAG_COLUMNS:
        frame[column] = 0
    return frame


# Observations covering the three sets of `temporal_split`: the end of 2024, all of 2025 and the start of 2026
TRAINING_SPAN_START = "2024-11-01T00:00:00+00:00"
TRAINING_SPAN_HOURS = (pd.Timestamp("2026-01-15T00:00:00+00:00") - pd.Timestamp(TRAINING_SPAN_START)) // pd.Timedelta(hours=1)


@pytest.fixture
def hourly_observations() -> pd.DataFrame:
    """`MIN_HISTORY_HOURS` + 48 consecutive hours from `OBSERVATIONS_START`, without noise (predictable results)."""
    return make_raw_observations(OBSERVATIONS_START, OBSERVATIONS_HOURS)


@pytest.fixture
def training_span_observations() -> pd.DataFrame:
    """Raw observations from November 2024 to mid-January 2026 (with noise): enough for train, validation and test."""
    return make_raw_observations(TRAINING_SPAN_START, TRAINING_SPAN_HOURS, noise_mw=300.0)


@pytest.fixture
def invalid_observations(hourly_observations: pd.DataFrame) -> pd.DataFrame:
    """Copy with invalid cases: duplicated `time_utc`, null and out-of-range `load_mw`, flag = 2.

    Expected after validation: 3 rejected rows (`DUPLICATED_ROWS` + `INVALID_FLAG_ROW`) and
    `load_mw` = NaN in `NULL_LOAD_ROW` and `OUT_OF_RANGE_LOAD_ROW`.
    """
    frame = hourly_observations.copy()
    first, second = DUPLICATED_ROWS
    frame.loc[second, "time_utc"] = frame.loc[first, "time_utc"]
    frame.loc[NULL_LOAD_ROW, "load_mw"] = np.nan
    frame.loc[OUT_OF_RANGE_LOAD_ROW, "load_mw"] = 60_000.0
    frame.loc[INVALID_FLAG_ROW, "is_holiday"] = 2
    return frame


FORECAST_TEMPERATURE_C = 12.0


@pytest.fixture
def weather_forecast_24h() -> pd.DataFrame:
    """Weather forecast for the 24 hours after the last hour of `hourly_observations` (constant values).

    Its origin is `FORECAST_ORIGIN`; same columns as `WeatherForecastClient.next_hours` (`time_utc` + `WEATHER_COLUMNS`).
    """
    return pd.DataFrame(
        {
            "time_utc": pd.date_range(FORECAST_ORIGIN + pd.Timedelta(hours=1), periods=HORIZON_HOURS, freq="h"),
            "temperature_2m": FORECAST_TEMPERATURE_C,
            "relative_humidity_2m": 55.0,
            "wind_speed_10m": 12.0,
            "shortwave_radiation": 0.0,
            "precipitation": 0.0,
        }
    )


@pytest.fixture
def small_trained_model(hourly_observations):
    """Small HGB pipeline trained on `hourly_observations` (for loading and inference tests)."""
    observations, _ = validate_observations(hourly_observations)
    X, y = build_training_frame(observations)
    return build_hgb_pipeline(max_iter=20, early_stopping=False).fit(X, y)


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
        nvidia_api_key=None,
    )


@pytest.fixture
def restore_root_logger():
    """`setup_logging` uses `force=True` and replaces the root handlers (including pytest's): restore them afterwards.

    Use it in any test that calls `setup_logging`, directly or through a CLI's `main`.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in saved_handlers:
            handler.close()  # releases the log file (needed on Windows to delete tmp_path)
    root.handlers = saved_handlers
    root.setLevel(saved_level)
