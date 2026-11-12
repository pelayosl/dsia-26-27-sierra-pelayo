"""Composition root: the only place where concrete implementations are chosen from `Settings`.

CLIs and the API request their services from here, so no other layer knows concrete infrastructure classes.
"""

from sqlalchemy.engine import Engine

from energy_forecast.config import Settings
from energy_forecast.infrastructure.dataset_source import CsvDatasetSource
from energy_forecast.infrastructure.db import create_db_engine, init_schema
from energy_forecast.infrastructure.model_store import JoblibModelStore
from energy_forecast.infrastructure.repositories import (
    SqlObservationRepository,
    SqlPipelineRunRepository,
    SqlPredictionRepository,
)
from energy_forecast.infrastructure.weather_client import OpenMeteoForecastClient
from energy_forecast.services.forecast import ForecastService
from energy_forecast.services.ingestion import IngestionService
from energy_forecast.services.training import TrainingService

METRICS_FILE_NAME = "metrics.json"


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


def _model_store(settings: Settings) -> JoblibModelStore:
    return JoblibModelStore(settings.model_path, settings.reports_dir / METRICS_FILE_NAME)


def build_training_service(settings: Settings) -> TrainingService:
    """SQL repositories + `JoblibModelStore(settings.model_path, settings.reports_dir / "metrics.json")`."""
    engine = _database(settings)
    return TrainingService(
        observations=SqlObservationRepository(engine),
        model_store=_model_store(settings),
        runs=SqlPipelineRunRepository(engine),
    )


def build_forecast_service(settings: Settings) -> ForecastService:
    """SQL repositories + `OpenMeteoForecastClient` + `JoblibModelStore` + `get_client(settings)`."""
    engine = _database(settings)
    return ForecastService(
        observations=SqlObservationRepository(engine),
        weather=OpenMeteoForecastClient(),
        model_store=_model_store(settings),
        predictions=SqlPredictionRepository(engine),
        runs=SqlPipelineRunRepository(engine),
        # TODO (step 4.4): `ai_client=get_client(settings)` (mock by default).
    )
