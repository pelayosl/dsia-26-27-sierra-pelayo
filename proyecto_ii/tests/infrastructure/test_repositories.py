"""Tests for `infrastructure.db` and the SQL repositories, on SQLite in `tmp_path`."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect

from energy_forecast.domain.models import Forecast, HourlyPrediction
from energy_forecast.domain.schema import HORIZON_HOURS, REQUIRED_COLUMNS
from energy_forecast.domain.validator import validate_observations
from energy_forecast.infrastructure.db import (
    OBSERVATIONS_TABLE,
    PIPELINE_RUNS_TABLE,
    PREDICTIONS_TABLE,
    create_db_engine,
    init_schema,
)
from energy_forecast.infrastructure.repositories import (
    SqlObservationRepository,
    SqlPipelineRunRepository,
    SqlPredictionRepository,
)

ORIGIN = datetime(2026, 11, 10, 23, tzinfo=timezone.utc)


@pytest.fixture
def engine(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'db' / 'energy.db').as_posix()}")
    init_schema(engine)
    yield engine
    engine.dispose()  # releases the SQLite file (needed on Windows)


@pytest.fixture
def observations(hourly_observations) -> pd.DataFrame:
    """Validated observations (the input `upsert` expects)."""
    valid, _ = validate_observations(hourly_observations)
    return valid


@pytest.fixture
def observation_repo(engine) -> SqlObservationRepository:
    return SqlObservationRepository(engine)


def make_forecast(origin: datetime = ORIGIN, base_mw: float = 30_000.0) -> Forecast:
    predictions = tuple(
        HourlyPrediction(time_utc=origin + timedelta(hours=hour), predicted_mw=base_mw + hour)
        for hour in range(1, HORIZON_HOURS + 1)
    )
    return Forecast(origin=origin, predictions=predictions, model_version="hgb-test")


# ---------------------------------------------------------------- db


def test_create_engine_creates_sqlite_folder(tmp_path):
    db_file = tmp_path / "nested" / "folder" / "energy.db"

    engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
    init_schema(engine)
    engine.dispose()

    assert db_file.exists()


def test_init_schema_creates_tables_and_is_idempotent(engine):
    init_schema(engine)  # second call must not fail

    tables = set(inspect(engine).get_table_names())

    assert {OBSERVATIONS_TABLE, PREDICTIONS_TABLE, PIPELINE_RUNS_TABLE} <= tables


# ---------------------------------------------------------------- observations


def test_upsert_returns_rows_written(observation_repo, observations):
    # Act
    written = observation_repo.upsert(observations)

    # Assert
    assert written == len(observations)
    assert len(observation_repo.range()) == len(observations)


def test_upsert_empty_frame_writes_nothing(observation_repo, observations):
    assert observation_repo.upsert(observations.iloc[0:0]) == 0
    assert observation_repo.range().empty


def test_upsert_is_idempotent(observation_repo, observations):
    observation_repo.upsert(observations)
    observation_repo.upsert(observations)

    assert len(observation_repo.range()) == len(observations)


def test_upsert_updates_existing_hours(observation_repo, observations):
    observation_repo.upsert(observations)
    corrected = observations.iloc[[5]].copy()
    corrected["load_mw"] = 12_345.0

    observation_repo.upsert(corrected)

    stored = observation_repo.range(start=corrected["time_utc"].iloc[0], end=corrected["time_utc"].iloc[0])
    assert stored["load_mw"].iloc[0] == 12_345.0
    assert len(observation_repo.range()) == len(observations)


def test_range_roundtrip_keeps_columns_types_and_nan(observation_repo, observations):
    observations.loc[10, "load_mw"] = np.nan
    observations.loc[11, "temperature_2m"] = np.nan
    observation_repo.upsert(observations)

    stored = observation_repo.range()

    assert list(stored.columns) == REQUIRED_COLUMNS
    assert str(stored["time_utc"].dt.tz) == "UTC"
    assert str(stored["time_local"].dt.tz) == "Europe/Madrid"
    pd.testing.assert_frame_equal(stored, observations, check_dtype=False)


def test_range_filters_inclusive_and_sorted(observation_repo, observations):
    observation_repo.upsert(observations.sample(frac=1, random_state=0))
    start, end = observations["time_utc"].iloc[10], observations["time_utc"].iloc[19]

    stored = observation_repo.range(start=start, end=end)

    assert len(stored) == 10
    assert stored["time_utc"].iloc[0] == start
    assert stored["time_utc"].iloc[-1] == end
    assert stored["time_utc"].is_monotonic_increasing


def test_range_accepts_other_timezones(observation_repo, observations):
    observation_repo.upsert(observations)
    start = observations["time_utc"].iloc[10]

    stored = observation_repo.range(start=start.tz_convert("Europe/Madrid"), end=start)

    assert len(stored) == 1
    assert stored["time_utc"].iloc[0] == start


def test_range_on_empty_database_returns_empty_frame_with_columns(observation_repo):
    stored = observation_repo.range()

    assert stored.empty
    assert list(stored.columns) == REQUIRED_COLUMNS


def test_last_hours_returns_expected_window(observation_repo, observations):
    observation_repo.upsert(observations)

    window = observation_repo.last_hours(168)

    assert len(window) == 168
    assert window["time_utc"].iloc[-1] == observation_repo.latest_time()
    assert window["time_utc"].is_monotonic_increasing


def test_last_hours_until(observation_repo, observations):
    observation_repo.upsert(observations)
    until = observations["time_utc"].iloc[200]

    window = observation_repo.last_hours(24, until=until)

    assert len(window) == 24
    assert window["time_utc"].iloc[-1] == until


@pytest.mark.parametrize("hours", [0, -1])
def test_last_hours_requires_positive_hours(observation_repo, hours):
    with pytest.raises(ValueError):
        observation_repo.last_hours(hours)


def test_last_hours_on_empty_database_is_empty(observation_repo):
    window = observation_repo.last_hours(24)

    assert window.empty
    assert list(window.columns) == REQUIRED_COLUMNS


def test_latest_time_ignores_hours_without_load(observation_repo, observations):
    observations.loc[observations.index[-3:], "load_mw"] = np.nan
    observation_repo.upsert(observations)

    latest = observation_repo.latest_time()

    assert latest == observations["time_utc"].iloc[-4]
    assert latest.tzinfo is not None


def test_latest_time_on_empty_database_is_none(observation_repo):
    assert observation_repo.latest_time() is None


# ---------------------------------------------------------------- predictions


def test_save_and_read_predictions(engine):
    repo = SqlPredictionRepository(engine)
    forecast = make_forecast()

    run_id = repo.save(forecast)
    stored = repo.between(ORIGIN, ORIGIN + timedelta(hours=HORIZON_HOURS))

    assert len(stored) == HORIZON_HOURS
    assert (stored["run_id"] == run_id).all()
    assert (stored["origin"] == ORIGIN).all()
    assert (stored["model_version"] == "hgb-test").all()
    assert list(stored["target_time_utc"]) == [p.time_utc for p in forecast.predictions]
    assert list(stored["predicted_mw"]) == [p.predicted_mw for p in forecast.predictions]


def test_between_excludes_hours_outside_window(engine):
    repo = SqlPredictionRepository(engine)
    repo.save(make_forecast())

    stored = repo.between(ORIGIN + timedelta(hours=1), ORIGIN + timedelta(hours=6))

    assert len(stored) == 6


def test_each_save_gets_a_new_run_id(engine):
    repo = SqlPredictionRepository(engine)

    first = repo.save(make_forecast())
    second = repo.save(make_forecast(base_mw=31_000.0))

    stored = repo.between(ORIGIN, ORIGIN + timedelta(hours=HORIZON_HOURS))
    assert first != second
    assert len(stored) == 2 * HORIZON_HOURS


# ---------------------------------------------------------------- pipeline runs


def test_pipeline_run_lifecycle(engine):
    repo = SqlPipelineRunRepository(engine)

    run_id = repo.start("ingest")
    running = repo.get(run_id)
    repo.finish(run_id, "ok", rows=384)
    finished = repo.get(run_id)

    assert running["status"] == "running"
    assert running["kind"] == "ingest"
    assert running["finished_at"] is None
    assert finished["status"] == "ok"
    assert finished["rows"] == 384
    assert finished["finished_at"] >= finished["started_at"]


def test_failed_run_records_error(engine):
    repo = SqlPipelineRunRepository(engine)
    run_id = repo.start("train")

    repo.finish(run_id, "error", error="SchemaError: missing columns")

    run = repo.get(run_id)
    assert run["status"] == "error"
    assert run["error"] == "SchemaError: missing columns"


def test_unknown_run_kind_raises(engine):
    with pytest.raises(ValueError):
        SqlPipelineRunRepository(engine).start("deploy")


def test_unknown_final_status_raises(engine):
    repo = SqlPipelineRunRepository(engine)
    run_id = repo.start("forecast")

    with pytest.raises(ValueError):
        repo.finish(run_id, "running")


def test_finish_unknown_run_raises(engine):
    with pytest.raises(ValueError):
        SqlPipelineRunRepository(engine).finish("does-not-exist", "ok")


def test_get_unknown_run_is_none(engine):
    assert SqlPipelineRunRepository(engine).get("does-not-exist") is None
