"""Fake implementations of the infrastructure `Protocol`s, to test services without database or network.

They must behave like the real implementations: `tests/infrastructure/test_observation_repository_contract.py`
runs the same tests against the SQL and in-memory repositories.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pydantic_core import to_jsonable_python
from sklearn.base import BaseEstimator

from energy_forecast.domain.errors import ExternalServiceError, ModelNotFoundError
from energy_forecast.domain.models import Forecast, ForecastSummary
from energy_forecast.domain.schema import (
    HORIZON_HOURS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEATHER_COLUMNS,
)
from energy_forecast.infrastructure.model_store import StoredModel
from energy_forecast.infrastructure.repositories import (
    PREDICTION_COLUMNS,
    RUN_FINAL_STATUSES,
    RUN_KINDS,
    RUN_STATUS_RUNNING,
)


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
    """`PredictionRepository` that stores the `Forecast`s in a list (`run_id`s `prediction-1`, `prediction-2`, …)."""

    def __init__(self) -> None:
        self.saved: list[Forecast] = []

    def save(self, forecast: Forecast) -> str:
        self.saved.append(forecast)
        return f"prediction-{len(self.saved)}"

    def between(self, start: datetime, end: datetime) -> pd.DataFrame:
        rows = [
            {
                "run_id": f"prediction-{number}",
                "created_at": pd.Timestamp.now(tz="UTC"),
                "origin": _as_utc(forecast.origin),
                "target_time_utc": _as_utc(prediction.time_utc),
                "predicted_mw": prediction.predicted_mw,
                "model_version": forecast.model_version,
            }
            for number, forecast in enumerate(self.saved, start=1)
            for prediction in forecast.predictions
        ]
        frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        in_range = (frame["target_time_utc"] >= _as_utc(start)) & (frame["target_time_utc"] <= _as_utc(end))
        return frame[in_range].reset_index(drop=True)


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
    """`WeatherForecastClient` that returns fixed weather values, moved to the requested hours.

    `frame` has `WEATHER_COLUMNS` (its `time_utc`, if any, is ignored); its first `hours` rows are returned
    with `time_utc` = origin + 1 h ... origin + hours. Raises `error` if given. Records the requested origins.
    """

    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self._frame = frame
        self._error = error
        self.origins: list[datetime] = []

    def next_hours(self, origin: datetime, hours: int = HORIZON_HOURS) -> pd.DataFrame:
        self.origins.append(origin)
        if self._error is not None:
            raise self._error
        if len(self._frame) < hours:
            raise ValueError(f"The fake forecast has {len(self._frame)} rows, {hours} requested")
        start = _as_utc(origin) + timedelta(hours=1)
        result = self._frame[WEATHER_COLUMNS].head(hours).reset_index(drop=True)
        result.insert(0, TIME_COLUMN, pd.date_range(start, periods=hours, freq="h"))
        return result


class FakeModelStore:
    """In-memory `ModelStore`. Versions are `<name>-test-v1`, `<name>-test-v2`, …"""

    def __init__(self, stored_model: StoredModel | None = None) -> None:
        self._stored = stored_model
        self.saves = 0

    def save(self, model: BaseEstimator, name: str, metrics: dict[str, Any]) -> str:
        self.saves += 1
        version = f"{name}-test-v{self.saves}"
        report = to_jsonable_python({"model_version": version, **metrics})
        self._stored = StoredModel(model=model, version=version, metrics=report)
        return version

    def load(self) -> StoredModel:
        if self._stored is None:
            raise ModelNotFoundError("No trained model in the fake store")
        return self._stored


class FakeAIClient:
    """`AIClient` that returns a fixed text and records the summaries it receives."""
    name = "nvidia"
    model = "fake/model"

    def __init__(self, text: str = "Fake explanation.") -> None:
        self._text = text
        self.summaries: list[ForecastSummary] = []

    def explain(self, summary: ForecastSummary) -> str:
        self.summaries.append(summary)
        return self._text


class FailingAIClient:
    """`AIClient` that always fails, to test the *fallback* to mock."""
    name = "nvidia"
    model = "fake/model"

    def explain(self, summary: ForecastSummary) -> str:
        raise ExternalServiceError("NVIDIA API request failed (TimeoutError): read timed out")
