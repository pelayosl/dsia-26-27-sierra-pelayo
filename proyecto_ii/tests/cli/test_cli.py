"""Smoke tests of the CLIs (exit codes), with `tmp_path` and `AI_PROVIDER=mock`."""

import json
from datetime import datetime, timezone

import pytest

from energy_forecast import bootstrap
from energy_forecast.cli import EXIT_FAILURE, EXIT_INPUT_ERROR, EXIT_OK
from energy_forecast.cli import forecast, ingest, train
from energy_forecast.config import load_settings
from energy_forecast.domain import modeling
from energy_forecast.domain.errors import ExternalServiceError
from energy_forecast.domain.schema import HORIZON_HOURS
from energy_forecast.infrastructure.db import create_db_engine
from energy_forecast.infrastructure.repositories import SqlObservationRepository, SqlPredictionRepository
from tests.fakes import FakeWeatherForecastClient

pytestmark = pytest.mark.usefixtures("restore_root_logger")


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


def stored_observations():
    engine = create_db_engine(load_settings().database_url)
    try:
        return SqlObservationRepository(engine).range()
    finally:
        engine.dispose()


# ---------------------------------------------------------------- ingest


def test_ingest_parse_args_defaults():
    args = ingest.parse_args([])

    assert args.input is None
    assert args.log_file is None
    assert args.verbose is False


def test_ingest_happy_path_is_idempotent(cli_env, hourly_observations):
    hourly_observations.to_csv(cli_env / "energy_weather_es_hourly.csv", index=False)

    assert ingest.main([]) == EXIT_OK
    assert ingest.main([]) == EXIT_OK

    assert len(stored_observations()) == len(hourly_observations)


def test_ingest_input_argument_overrides_dataset_path(cli_env, hourly_observations):
    other = cli_env / "other.csv"
    hourly_observations.head(50).to_csv(other, index=False)

    assert ingest.main(["--input", str(other)]) == EXIT_OK

    assert len(stored_observations()) == 50


def test_ingest_writes_log_file(cli_env, hourly_observations):
    hourly_observations.to_csv(cli_env / "energy_weather_es_hourly.csv", index=False)
    log_file = cli_env / "logs" / "ingest.log"

    assert ingest.main(["--log-file", str(log_file)]) == EXIT_OK

    assert "Ingestion finished" in log_file.read_text(encoding="utf-8")


def test_ingest_missing_file_returns_exit_2(cli_env):
    assert ingest.main(["--input", str(cli_env / "missing.csv")]) == EXIT_INPUT_ERROR


def test_ingest_schema_error_returns_exit_2(cli_env, hourly_observations):
    hourly_observations.drop(columns=["load_mw"]).to_csv(cli_env / "energy_weather_es_hourly.csv", index=False)

    assert ingest.main([]) == EXIT_INPUT_ERROR


def test_ingest_invalid_configuration_returns_exit_2(cli_env, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "nvidia")  # without NVIDIA_API_KEY
    monkeypatch.setenv("NVIDIA_API_KEY", "")

    assert ingest.main([]) == EXIT_INPUT_ERROR


def test_ingest_unexpected_error_returns_exit_1(cli_env, monkeypatch):
    class BrokenService:
        def run(self):
            raise RuntimeError("database is locked")

    monkeypatch.setattr(ingest, "build_ingestion_service", lambda settings: BrokenService())

    assert ingest.main([]) == EXIT_FAILURE


# ---------------------------------------------------------------- train


@pytest.fixture
def fast_hgb(monkeypatch):
    """HGB with few trees, so the real training runs in under a second."""
    monkeypatch.setitem(modeling.HGB_PARAMS, "max_iter", 20)


def test_train_parse_args_defaults():
    args = train.parse_args([])

    assert args.tune is False
    assert args.log_file is None
    assert args.verbose is False


def test_ingest_then_train_saves_model_and_metrics(cli_env, training_span_observations, fast_hgb, capsys):
    training_span_observations.to_csv(cli_env / "energy_weather_es_hourly.csv", index=False)
    assert ingest.main([]) == EXIT_OK

    assert train.main([]) == EXIT_OK

    metrics = json.loads((cli_env / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert (cli_env / "models" / "model.joblib").exists()
    assert set(metrics["metrics"]) == {"naive", "ridge", "hgb"}
    output = capsys.readouterr().out
    assert "<- saved" in output and metrics["model_version"] in output


def test_train_without_observations_returns_exit_2(cli_env):
    assert train.main([]) == EXIT_INPUT_ERROR

    assert not (cli_env / "models" / "model.joblib").exists()


def test_train_invalid_configuration_returns_exit_2(cli_env, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "nvidia")  # without NVIDIA_API_KEY
    monkeypatch.setenv("NVIDIA_API_KEY", "")

    assert train.main([]) == EXIT_INPUT_ERROR


def test_train_unexpected_error_returns_exit_1(cli_env, monkeypatch):
    class BrokenService:
        def train(self, tune=False):
            raise RuntimeError("disk full")

    monkeypatch.setattr(train, "build_training_service", lambda settings: BrokenService())

    assert train.main([]) == EXIT_FAILURE


def test_train_passes_tune_flag_to_the_service(cli_env, monkeypatch):
    received = {}

    class RecordingService:
        def train(self, tune=False):
            received["tune"] = tune
            raise RuntimeError("stop here")

    monkeypatch.setattr(train, "build_training_service", lambda settings: RecordingService())

    train.main(["--tune"])

    assert received == {"tune": True}


# ---------------------------------------------------------------- forecast


@pytest.fixture
def fake_weather(monkeypatch, weather_forecast_24h):
    """Replaces Open-Meteo in the composition root: the CLI tests never use the network."""
    fake = FakeWeatherForecastClient(weather_forecast_24h)
    monkeypatch.setattr(bootstrap, "OpenMeteoForecastClient", lambda: fake)
    return fake


@pytest.fixture
def trained(cli_env, training_span_observations, fast_hgb):
    """Database with observations up to mid-January 2026 and a trained model, through the real CLIs."""
    training_span_observations.to_csv(cli_env / "energy_weather_es_hourly.csv", index=False)
    assert ingest.main([]) == EXIT_OK
    assert train.main([]) == EXIT_OK
    return cli_env


def test_forecast_parse_args_defaults():
    args = forecast.parse_args([])

    assert args.origin is None
    assert args.log_file is None
    assert args.verbose is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-01-14T23:00:00+00:00", datetime(2026, 1, 14, 23, tzinfo=timezone.utc)),
        ("2026-01-14T23:00", datetime(2026, 1, 14, 23, tzinfo=timezone.utc)),  # no timezone → UTC
    ],
)
def test_forecast_parse_origin(value, expected):
    assert forecast.parse_args(["--origin", value]).origin == expected


@pytest.mark.parametrize("value", ["yesterday", "2026-01-14T23:30:00+00:00"], ids=["not-iso", "not-on-the-hour"])
def test_forecast_invalid_origin_exits_with_2(value):
    with pytest.raises(SystemExit) as exit_info:
        forecast.parse_args(["--origin", value])

    assert exit_info.value.code == EXIT_INPUT_ERROR


def test_ingest_train_forecast_happy_path(trained, fake_weather, capsys):
    capsys.readouterr()  # discard the output of `train`

    assert forecast.main([]) == EXIT_OK

    report_file = trained / "reports" / "forecast_20260114T23.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["origin"] == "2026-01-14T23:00:00Z"
    assert len(report["predictions"]) == HORIZON_HOURS
    assert set(report["predictions"][0]) == {"time_utc", "predicted_mw"}
    assert report["model_version"].startswith(("naive-", "ridge-", "hgb-"))
    assert "Report written to" in capsys.readouterr().out
    assert fake_weather.origins == [datetime(2026, 1, 14, 23, tzinfo=timezone.utc)]


def test_forecast_with_explicit_origin(trained, fake_weather):
    assert forecast.main(["--origin", "2026-01-10T12:00:00+00:00"]) == EXIT_OK

    assert (trained / "reports" / "forecast_20260110T12.json").exists()


def test_forecast_saves_the_predictions_in_the_database(trained, fake_weather):
    assert forecast.main([]) == EXIT_OK

    engine = create_db_engine(load_settings().database_url)
    try:
        stored = SqlPredictionRepository(engine).between(
            datetime(2026, 1, 15, tzinfo=timezone.utc), datetime(2026, 1, 16, tzinfo=timezone.utc)
        )
    finally:
        engine.dispose()
    assert len(stored) == HORIZON_HOURS


def test_forecast_without_model_returns_exit_2(cli_env, hourly_observations, fake_weather):
    hourly_observations.to_csv(cli_env / "energy_weather_es_hourly.csv", index=False)
    assert ingest.main([]) == EXIT_OK

    assert forecast.main([]) == EXIT_INPUT_ERROR

    assert not list(cli_env.glob("reports/forecast_*.json"))
    assert fake_weather.origins == []  # it fails before calling the weather service


def test_forecast_without_enough_history_returns_exit_2(trained, fake_weather):
    assert forecast.main(["--origin", "2024-11-02T00:00:00+00:00"]) == EXIT_INPUT_ERROR


def test_forecast_weather_service_down_returns_exit_1(trained, monkeypatch):
    down = FakeWeatherForecastClient(error=ExternalServiceError("Open-Meteo answered 503"))
    monkeypatch.setattr(bootstrap, "OpenMeteoForecastClient", lambda: down)

    assert forecast.main([]) == EXIT_FAILURE


def test_forecast_unexpected_error_returns_exit_1(cli_env, monkeypatch):
    class BrokenService:
        def forecast(self, origin=None):
            raise RuntimeError("database is locked")

    monkeypatch.setattr(forecast, "build_forecast_service", lambda settings: BrokenService())

    assert forecast.main([]) == EXIT_FAILURE
