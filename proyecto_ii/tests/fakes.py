"""Fake implementations of the infrastructure `Protocol`s, to test services without database or network.

They must behave like the real implementations: `tests/infrastructure/test_observation_repository_contract.py`
runs the same tests against the SQL and in-memory repositories.
"""

from datetime import datetime, timedelta

import pandas as pd

from energy_forecast.domain.models import Forecast, ForecastSummary
from energy_forecast.domain.schema import REQUIRED_COLUMNS, TARGET_COLUMN, TIME_COLUMN
from energy_forecast.infrastructure.repositories import RUN_FINAL_STATUSES, RUN_KINDS, RUN_STATUS_RUNNING


def _as_utc(value: datetime) -> pd.Timestamp:
    """Same rule as the SQL repositories: naive datetimes are UTC."""
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


class FakeDatasetSource:
    """`DatasetSource` that returns a fixed DataFrame, or raises `error` if given."""

    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self._frame = frame
        self._error = error

    def read(self) -> pd.DataFrame:
        if self._error is not None:
            raise self._error
        return self._frame.copy()


class InMemoryObservationRepository:
    """`ObservationRepository` over an in-memory DataFrame."""

    def __init__(self) -> None:
        self._frame = pd.DataFrame(columns=REQUIRED_COLUMNS)

    def upsert(self, observations: pd.DataFrame) -> int:
        if observations.empty:
            return 0
        new_rows = observations[REQUIRED_COLUMNS]
        combined = new_rows.copy() if self._frame.empty else pd.concat([self._frame, new_rows])
        self._frame = (
            combined.drop_duplicates(TIME_COLUMN, keep="last").sort_values(TIME_COLUMN).reset_index(drop=True)
        )
        return len(observations)

    def range(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        frame = self._frame
        if start is not None:
            frame = frame[frame[TIME_COLUMN] >= _as_utc(start)]
        if end is not None:
            frame = frame[frame[TIME_COLUMN] <= _as_utc(end)]
        return frame.reset_index(drop=True).copy()

    def last_hours(self, hours: int, until: datetime | None = None) -> pd.DataFrame:
        if hours <= 0:
            raise ValueError("hours must be positive")
        until = until if until is not None else self.latest_time()
        if until is None:
            return self._frame.iloc[0:0].copy()
        return self.range(start=_as_utc(until) - timedelta(hours=hours - 1), end=until)

    def latest_time(self) -> datetime | None:
        known = self._frame.loc[self._frame[TARGET_COLUMN].notna(), TIME_COLUMN]
        return None if known.empty else known.max().to_pydatetime()


class InMemoryPredictionRepository:
    """`PredictionRepository` that stores the `Forecast`s in a list."""

    def __init__(self) -> None:
        self.saved: list[Forecast] = []

    def save(self, forecast: Forecast) -> str:
        # TODO: append to `saved` and return a predictable `run_id`.
        raise NotImplementedError

    def between(self, start: datetime, end: datetime) -> pd.DataFrame:
        # TODO: flatten `saved` and filter by `target_time_utc`.
        raise NotImplementedError


class InMemoryPipelineRunRepository:
    """`PipelineRunRepository` that records the runs in a dict (ids `run-1`, `run-2`, …)."""

    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    def start(self, kind: str) -> str:
        if kind not in RUN_KINDS:
            raise ValueError(f"Unknown run kind '{kind}'")
        run_id = f"run-{len(self.runs) + 1}"
        self.runs[run_id] = {"run_id": run_id, "kind": kind, "status": RUN_STATUS_RUNNING, "rows": 0, "error": None}
        return run_id

    def finish(self, run_id: str, status: str, rows: int = 0, error: str | None = None) -> None:
        if status not in RUN_FINAL_STATUSES:
            raise ValueError(f"Unknown run status '{status}'")
        if run_id not in self.runs:
            raise ValueError(f"Unknown run_id '{run_id}'")
        self.runs[run_id].update(status=status, rows=rows, error=error)

    def get(self, run_id: str) -> dict | None:
        run = self.runs.get(run_id)
        return None if run is None else dict(run)


class FakeWeatherForecastClient:
    """`WeatherForecastClient` that returns a fixed forecast."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def next_hours(self, origin: datetime, hours: int = 24) -> pd.DataFrame:
        # TODO: return `self._frame` with `time_utc` = origin + 1 h ... origin + hours.
        raise NotImplementedError


class FakeModelStore:
    """In-memory `ModelStore`."""

    def __init__(self, stored_model=None) -> None:
        self._stored = stored_model

    def save(self, model, metrics: dict) -> str:
        # TODO: store in memory and return a fixed version ("test-v1").
        raise NotImplementedError

    def load(self):
        # TODO: return `self._stored` or raise `ModelNotFoundError` if it is None.
        raise NotImplementedError


class FailingAIClient:
    """`AIClient` that always fails, to test the *fallback* to mock."""
    name = "hf"
    model = "fake/model"

    def explain(self, summary: ForecastSummary) -> str:
        # TODO: lanzar `ExternalServiceError`.
        raise NotImplementedError
