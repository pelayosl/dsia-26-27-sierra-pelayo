"""Composition root: the only place where concrete implementations are chosen from `Settings`.

CLIs and the API request their services from here, so no other layer knows concrete infrastructure classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.engine import Engine

from energy_forecast.config import Settings
from energy_forecast.infrastructure.dataset_source import CsvDatasetSource
from energy_forecast.infrastructure.db import create_db_engine, init_schema
from energy_forecast.infrastructure.repositories import SqlObservationRepository, SqlPipelineRunRepository
from energy_forecast.services.ingestion import IngestionService

if TYPE_CHECKING:
    # TODO (steps 2.5 and 3.3): switch to regular imports when implementing `build_training_service` and `build_forecast_service`
    from energy_forecast.services.forecast import ForecastService
    from energy_forecast.services.training import TrainingService


def _database(settings: Settings) -> Engine:
    """Engine for `DATABASE_URL`, with the schema created if needed."""
    engine = create_db_engine(settings.database_url)
    init_schema(engine)
    return engine


def build_ingestion_service(settings: Settings) -> IngestionService:
    """`CsvDatasetSource(settings.dataset_path)` + SQL repositories."""
    engine = _database(settings)
    return IngestionService(
        source=CsvDatasetSource(settings.dataset_path),
        observations=SqlObservationRepository(engine),
        runs=SqlPipelineRunRepository(engine),
    )


def build_training_service(settings: Settings) -> TrainingService:
    """SQL repositories + `JoblibModelStore(settings.model_path, settings.reports_dir / "metrics.json")`."""
    # TODO: same as above, with `JoblibModelStore`.
    raise NotImplementedError


def build_forecast_service(settings: Settings) -> ForecastService:
    """SQL repositories + `OpenMeteoForecastClient` + `JoblibModelStore` + `get_client(settings)`."""
    # TODO: the `AIClient` comes from `get_client(settings)` (mock by default).
    raise NotImplementedError
