"""Contract tests: `SqlPredictionRepository` and `InMemoryPredictionRepository` must behave the same way."""

from datetime import datetime, timedelta, timezone

import pytest

from energy_forecast.domain.models import Forecast, HourlyPrediction
from energy_forecast.domain.schema import HORIZON_HOURS
from energy_forecast.infrastructure.db import create_db_engine, init_schema
from energy_forecast.infrastructure.repositories import PREDICTION_COLUMNS, SqlPredictionRepository
from tests.fakes import InMemoryPredictionRepository

ORIGIN = datetime(2026, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_forecast(origin: datetime = ORIGIN, model_version: str = "hgb-test") -> Forecast:
    predictions = tuple(
        HourlyPrediction(time_utc=origin + timedelta(hours=hour), predicted_mw=30_000.0 + hour)
        for hour in range(1, HORIZON_HOURS + 1)
    )
    return Forecast(origin=origin, predictions=predictions, model_version=model_version)


@pytest.fixture(params=["sql", "memory"])
def repo(request, tmp_path):
    if request.param == "memory":
        yield InMemoryPredictionRepository()
        return
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'energy.db').as_posix()}")
    init_schema(engine)
    yield SqlPredictionRepository(engine)
    engine.dispose()


def test_saved_predictions_are_returned_in_range(repo):
    repo.save(make_forecast())

    stored = repo.between(ORIGIN, ORIGIN + timedelta(hours=HORIZON_HOURS))

    assert list(stored.columns) == PREDICTION_COLUMNS
    assert len(stored) == HORIZON_HOURS
    assert stored["predicted_mw"].iloc[0] == pytest.approx(30_001.0)
    assert (stored["model_version"] == "hgb-test").all()
    assert str(stored["target_time_utc"].dt.tz) == "UTC"


def test_between_bounds_are_inclusive(repo):
    repo.save(make_forecast())
    first = ORIGIN + timedelta(hours=1)

    stored = repo.between(first, first + timedelta(hours=2))

    assert len(stored) == 3


def test_each_save_gets_its_own_run_id(repo):
    first = repo.save(make_forecast())
    second = repo.save(make_forecast(model_version="hgb-test-2"))

    stored = repo.between(ORIGIN, ORIGIN + timedelta(days=1))

    assert first != second
    assert set(stored["run_id"]) == {first, second}
    assert len(stored) == 2 * HORIZON_HOURS


def test_between_without_predictions_is_empty(repo):
    stored = repo.between(ORIGIN, ORIGIN + timedelta(days=1))

    assert stored.empty
    assert list(stored.columns) == PREDICTION_COLUMNS
