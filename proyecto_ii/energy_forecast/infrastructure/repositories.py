"""Own persistence repositories: observations, predictions and pipeline runs."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import pandas as pd
from sqlalchemy import Table, func, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine

from energy_forecast.domain.models import Forecast
from energy_forecast.domain.schema import (
    FLAG_COLUMNS,
    LOCAL_TIMEZONE,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEATHER_COLUMNS,
)
from energy_forecast.infrastructure.db import observations, pipeline_runs, predictions

STORED_OBSERVATION_COLUMNS = [TIME_COLUMN, TARGET_COLUMN, *WEATHER_COLUMNS, *FLAG_COLUMNS]
PREDICTION_COLUMNS = ["run_id", "created_at", "origin", "target_time_utc", "predicted_mw", "model_version"]

RUN_KINDS = ("ingest", "train", "forecast")
RUN_STATUS_RUNNING = "running"
RUN_FINAL_STATUSES = ("ok", "error")


class ObservationRepository(Protocol):
    """Hourly load and weather history (source of truth after ingestion)."""
    def upsert(self, observations: pd.DataFrame) -> int:
        """Inserts or updates by `time_utc`; returns the rows written."""

    def range(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Observations between `start` and `end` (both inclusive), sorted by `time_utc`."""

    def last_hours(self, hours: int, until: datetime | None = None) -> pd.DataFrame:
        """The last `hours` hours up to `until` (by default, the latest one with known load)."""

    def latest_time(self) -> datetime | None:
        """Latest `time_utc` with known load, or `None` if there is no data."""


class PredictionRepository(Protocol):
    """Issued forecasts, for history and monitoring."""
    def save(self, forecast: Forecast) -> str:
        """Saves the 24 predictions; returns the `run_id`."""

    def between(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Predictions whose `target_time_utc` falls between `start` and `end` (both inclusive)."""


class PipelineRunRepository(Protocol):
    """Log of ingestion, training and forecast runs."""
    def start(self, kind: str) -> str:
        """Records the start of a run; returns its `run_id`."""

    def finish(self, run_id: str, status: str, rows: int = 0, error: str | None = None) -> None:
        """Marks the run as finished (`ok` or `error`)."""

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Returns the run as a dict, or `None` if it does not exist."""


def _to_db_time(value: datetime) -> datetime:
    """tz-aware → naive UTC, as stored in the database. Naive values are assumed to be UTC already."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_pydatetime()


def _from_db_time(value: datetime | None) -> datetime | None:
    """Naive UTC read from the database → tz-aware UTC."""
    return None if value is None else value.replace(tzinfo=timezone.utc)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _upsert_statement(engine: Engine, table: Table, key_columns: list[str]):
    """`INSERT ... ON CONFLICT DO UPDATE` for SQLite and Postgres (both share the same SQLAlchemy API)."""
    dialects = {"sqlite": sqlite, "postgresql": postgresql}
    dialect = dialects.get(engine.dialect.name)
    if dialect is None:
        raise NotImplementedError(f"Upsert not supported for database '{engine.dialect.name}'")
    statement = dialect.insert(table)
    updated = {column.name: statement.excluded[column.name] for column in table.columns if column.name not in key_columns}
    return statement.on_conflict_do_update(index_elements=key_columns, set_=updated)


class SqlObservationRepository:
    """`ObservationRepository` over SQL."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, observations_frame: pd.DataFrame) -> int:
        """Idempotent write by `time_utc`: running the same ingestion twice does not duplicate rows.

        Expects the output of `validate_observations` (`time_local` is ignored: it is derived on read).
        """
        if observations_frame.empty:
            return 0
        frame = observations_frame[STORED_OBSERVATION_COLUMNS].copy()
        frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True).dt.tz_localize(None)
        # NaN → None (SQL NULL); `to_dict` turns numpy scalars into native Python types
        records = frame.astype(object).where(frame.notna(), None).to_dict("records")

        with self._engine.begin() as connection:
            connection.execute(_upsert_statement(self._engine, observations, [TIME_COLUMN]), records)
        return len(records)

    def range(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Same columns and types as the output of `validate_observations`."""
        statement = select(observations).order_by(observations.c[TIME_COLUMN])
        if start is not None:
            statement = statement.where(observations.c[TIME_COLUMN] >= _to_db_time(start))
        if end is not None:
            statement = statement.where(observations.c[TIME_COLUMN] <= _to_db_time(end))

        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return self._to_observation_frame(rows)

    def last_hours(self, hours: int, until: datetime | None = None) -> pd.DataFrame:
        """`hours` hourly slots ending at `until` (inclusive). Empty if there is no data."""
        if hours <= 0:
            raise ValueError("hours must be positive")
        until = until if until is not None else self.latest_time()
        if until is None:
            return self._to_observation_frame([])
        return self.range(start=until - timedelta(hours=hours - 1), end=until)

    def latest_time(self) -> datetime | None:
        statement = select(func.max(observations.c[TIME_COLUMN])).where(observations.c[TARGET_COLUMN].is_not(None))
        with self._engine.connect() as connection:
            return _from_db_time(connection.execute(statement).scalar_one())

    @staticmethod
    def _to_observation_frame(rows) -> pd.DataFrame:
        """Database rows → DataFrame with the same columns and types as `validate_observations`."""
        frame = pd.DataFrame(rows, columns=STORED_OBSERVATION_COLUMNS)
        frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True)
        frame["time_local"] = frame[TIME_COLUMN].dt.tz_convert(LOCAL_TIMEZONE)
        frame[[TARGET_COLUMN, *WEATHER_COLUMNS]] = frame[[TARGET_COLUMN, *WEATHER_COLUMNS]].astype(float)
        frame[FLAG_COLUMNS] = frame[FLAG_COLUMNS].astype(int)
        return frame[REQUIRED_COLUMNS]


class SqlPredictionRepository:
    """`PredictionRepository` over SQL."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(self, forecast: Forecast) -> str:
        run_id = uuid.uuid4().hex
        created_at = _to_db_time(_now_utc())
        records = [
            {
                "run_id": run_id,
                "target_time_utc": _to_db_time(prediction.time_utc),
                "created_at": created_at,
                "origin": _to_db_time(forecast.origin),
                "predicted_mw": prediction.predicted_mw,
                "model_version": forecast.model_version,
            }
            for prediction in forecast.predictions
        ]
        with self._engine.begin() as connection:
            connection.execute(predictions.insert(), records)
        return run_id

    def between(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Used by monitoring (Project III) and `/history`. Datetimes are returned as tz-aware UTC."""
        statement = (
            select(predictions)
            .where(predictions.c.target_time_utc.between(_to_db_time(start), _to_db_time(end)))
            .order_by(predictions.c.created_at, predictions.c.target_time_utc)
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()

        frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        for column in ("created_at", "origin", "target_time_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        frame["predicted_mw"] = frame["predicted_mw"].astype(float)
        return frame


class SqlPipelineRunRepository:
    """`PipelineRunRepository` over SQL."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def start(self, kind: str) -> str:
        if kind not in RUN_KINDS:
            raise ValueError(f"Unknown run kind '{kind}', expected one of {RUN_KINDS}")
        run_id = uuid.uuid4().hex
        with self._engine.begin() as connection:
            connection.execute(
                pipeline_runs.insert().values(
                    run_id=run_id,
                    kind=kind,
                    status=RUN_STATUS_RUNNING,
                    started_at=_to_db_time(_now_utc()),
                    rows=0,
                )
            )
        return run_id

    def finish(self, run_id: str, status: str, rows: int = 0, error: str | None = None) -> None:
        if status not in RUN_FINAL_STATUSES:
            raise ValueError(f"Unknown run status '{status}', expected one of {RUN_FINAL_STATUSES}")
        statement = (
            update(pipeline_runs)
            .where(pipeline_runs.c.run_id == run_id)
            .values(status=status, finished_at=_to_db_time(_now_utc()), rows=rows, error=error)
        )
        with self._engine.begin() as connection:
            if connection.execute(statement).rowcount == 0:
                raise ValueError(f"Unknown run_id '{run_id}'")

    def get(self, run_id: str) -> dict[str, Any] | None:
        statement = select(pipeline_runs).where(pipeline_runs.c.run_id == run_id)
        with self._engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        if row is None:
            return None
        run = dict(row)
        run["started_at"] = _from_db_time(run["started_at"])
        run["finished_at"] = _from_db_time(run["finished_at"])
        return run
