"""Database connection and schema (SQLAlchemy Core).

Datetimes are stored as naive UTC (`DateTime` without timezone) so the same schema works in SQLite,
which has no timezone type, and in Postgres. The repositories convert them back to tz-aware UTC.
"""

from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, make_url
from sqlalchemy.engine import Engine

from energy_forecast.domain.schema import FLAG_COLUMNS, TARGET_COLUMN, TIME_COLUMN, WEATHER_COLUMNS

OBSERVATIONS_TABLE = "observations"
PREDICTIONS_TABLE = "predictions"
PIPELINE_RUNS_TABLE = "pipeline_runs"

metadata = MetaData()

# One row per hour. `time_local` is not stored: it is derived from `time_utc` when reading.
observations = Table(
    OBSERVATIONS_TABLE,
    metadata,
    Column(TIME_COLUMN, DateTime, primary_key=True),
    Column(TARGET_COLUMN, Float, nullable=True),  # NaN (blackout, out of range) is stored as NULL
    *[Column(column, Float, nullable=True) for column in WEATHER_COLUMNS],
    *[Column(column, Integer, nullable=False) for column in FLAG_COLUMNS],
)

# One row per predicted hour; the 24 rows of a forecast share `run_id`.
predictions = Table(
    PREDICTIONS_TABLE,
    metadata,
    Column("run_id", String(32), primary_key=True),
    Column("target_time_utc", DateTime, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("origin", DateTime, nullable=False),
    Column("predicted_mw", Float, nullable=False),
    Column("model_version", String(64), nullable=False),
)

pipeline_runs = Table(
    PIPELINE_RUNS_TABLE,
    metadata,
    Column("run_id", String(32), primary_key=True),
    Column("kind", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime, nullable=True),
    Column("rows", Integer, nullable=False, default=0),
    Column("error", Text, nullable=True),
)


def create_db_engine(database_url: str) -> Engine:
    """Creates the engine from `DATABASE_URL` (SQLite locally; Postgres possible in Project III).

    For a SQLite file, its folder is created if it does not exist.
    """
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def init_schema(engine: Engine) -> None:
    """Creates the tables if they do not exist (safe to call more than once)."""
    metadata.create_all(engine)
