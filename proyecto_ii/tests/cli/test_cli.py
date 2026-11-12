"""Smoke tests of the CLIs (exit codes), with `tmp_path` and `AI_PROVIDER=mock`."""

import json

import pytest

from energy_forecast.cli import EXIT_FAILURE, EXIT_INPUT_ERROR, EXIT_OK
from energy_forecast.cli import ingest, train
from energy_forecast.domain import modeling
from energy_forecast.config import load_settings
from energy_forecast.infrastructure.db import create_db_engine
from energy_forecast.infrastructure.repositories import SqlObservationRepository

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
    monkeypatch.setenv("AI_PROVIDER", "hf")  # without HF_TOKEN / HF_MODEL
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)

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
    monkeypatch.setenv("AI_PROVIDER", "hf")  # without HF_TOKEN / HF_MODEL
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)

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


@pytest.mark.skip(reason="TODO (paso 3.3): implementar")
def test_ingest_train_forecast_happy_path(tmp_path, monkeypatch, hourly_observations):
    # TODO: ingest → 0, train → 0, forecast → 0 and `reports/forecast_*.json` exists.
    #       Replace `OpenMeteoForecastClient` with a fake via `monkeypatch` (no network).
    ...


@pytest.mark.skip(reason="TODO (paso 3.3): implementar")
def test_forecast_without_model_returns_exit_2(tmp_path, monkeypatch):
    # TODO: without `model.joblib` → EXIT_INPUT_ERROR.
    ...
