"""Contract tests: the SQL repository and its in-memory fake must behave the same way (Liskov).

If a test here fails only for `memory`, the fake is out of sync and the service tests that use it are not reliable.
"""

import numpy as np
import pandas as pd
import pytest

from energy_forecast.domain.schema import REQUIRED_COLUMNS
from energy_forecast.domain.validator import validate_observations
from energy_forecast.infrastructure.db import create_db_engine, init_schema
from energy_forecast.infrastructure.repositories import SqlObservationRepository
from tests.fakes import InMemoryObservationRepository


@pytest.fixture(params=["sql", "memory"])
def repo(request, tmp_path):
    if request.param == "memory":
        yield InMemoryObservationRepository()
        return
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'energy.db').as_posix()}")
    init_schema(engine)
    yield SqlObservationRepository(engine)
    engine.dispose()


@pytest.fixture
def observations(hourly_observations) -> pd.DataFrame:
    valid, _ = validate_observations(hourly_observations)
    return valid


def test_upsert_is_idempotent_and_updates(repo, observations):
    repo.upsert(observations)
    corrected = observations.iloc[[5]].copy()
    corrected["load_mw"] = 12_345.0

    written = repo.upsert(corrected)

    stored = repo.range()
    assert written == 1
    assert len(stored) == len(observations)
    assert stored.loc[5, "load_mw"] == 12_345.0


def test_range_returns_same_frame_as_validator(repo, observations):
    observations.loc[10, "load_mw"] = np.nan
    repo.upsert(observations.sample(frac=1, random_state=0))

    stored = repo.range()

    assert list(stored.columns) == REQUIRED_COLUMNS
    pd.testing.assert_frame_equal(stored, observations, check_dtype=False)


def test_range_is_inclusive_and_accepts_naive_utc(repo, observations):
    repo.upsert(observations)
    start, end = observations["time_utc"].iloc[10], observations["time_utc"].iloc[19]

    stored = repo.range(start=start.tz_localize(None), end=end.tz_convert("Europe/Madrid"))

    assert len(stored) == 10
    assert stored["time_utc"].iloc[0] == start


def test_last_hours_and_latest_time(repo, observations):
    observations.loc[observations.index[-2:], "load_mw"] = np.nan
    repo.upsert(observations)

    latest = repo.latest_time()
    window = repo.last_hours(168)

    assert latest == observations["time_utc"].iloc[-3]
    assert len(window) == 168
    assert window["time_utc"].iloc[-1] == latest


def test_empty_repository(repo):
    assert repo.latest_time() is None
    assert repo.range().empty
    assert repo.last_hours(24).empty
    assert list(repo.last_hours(24).columns) == REQUIRED_COLUMNS
