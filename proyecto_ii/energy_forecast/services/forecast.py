"""Use case: 24 h forecast with explanation. Used by the `forecast` CLI (Project II) and the API (Project III)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from energy_forecast.domain.errors import InsufficientHistoryError
from energy_forecast.domain.features import build_inference_frame
from energy_forecast.domain.models import Forecast
from energy_forecast.domain.predictor import predict_forecast
from energy_forecast.domain.schema import MIN_HISTORY_HOURS
from energy_forecast.infrastructure.model_store import ModelStore
from energy_forecast.infrastructure.repositories import (
    ObservationRepository,
    PipelineRunRepository,
    PredictionRepository,
)
from energy_forecast.infrastructure.weather_client import WeatherForecastClient
from energy_forecast.logging_config import get_logger

if TYPE_CHECKING:
    # TODO (step 4.4): switch to a regular import when the explanation is implemented
    from energy_forecast.infrastructure.ai_client import AIClient

logger = get_logger("forecast")

FALLBACK_PROVIDER = "mock-fallback"


class ForecastService:
    """Orchestrates history + weather forecast → features → model → database → AI explanation."""

    def __init__(
        self,
        observations: ObservationRepository,
        weather: WeatherForecastClient,
        model_store: ModelStore,
        predictions: PredictionRepository,
        runs: PipelineRunRepository,
        ai_client: AIClient | None = None,
    ) -> None:
        self._observations = observations
        self._weather = weather
        self._model_store = model_store
        self._predictions = predictions
        self._runs = runs
        self._ai_client = ai_client

    def forecast(self, origin: datetime | None = None) -> Forecast:
        """Predicts the 24 hours after `origin` (by default, the last hour with known demand) and saves them.

        The run is recorded in `pipeline_runs`. On any error it is marked as `error` and the exception is
        re-raised: `ModelNotFoundError` (no trained model), `InsufficientHistoryError` (not enough history
        before `origin`), `ExternalServiceError` (weather forecast unavailable), `ValueError` (invalid origin).
        """
        run_id = self._runs.start("forecast")
        try:
            forecast = self._forecast(origin)
        except Exception as exc:
            self._runs.finish(run_id, "error", error=f"{type(exc).__name__}: {exc}")
            logger.error("Forecast failed (run %s): %s", run_id, exc)
            raise

        self._runs.finish(run_id, "ok", rows=len(forecast.predictions))
        logger.info("Forecast finished (run %s): origin %s, model %s",
                    run_id, forecast.origin.isoformat(), forecast.model_version)
        # TODO (step 4.4): add the explanation with `_explain(...)` (`forecast.model_copy(update={...})`).
        return forecast

    def _forecast(self, origin: datetime | None) -> Forecast:
        # The model first: it is local and cheap, so a missing model fails before calling Open-Meteo
        stored = self._model_store.load()
        if origin is None:
            origin = self._observations.latest_time()
            if origin is None:
                raise InsufficientHistoryError("There are no observations in the database: run `ingest` first")
        logger.info("Forecast started: origin %s, model %s", origin.isoformat(), stored.version)

        history = self._observations.last_hours(MIN_HISTORY_HOURS, until=origin)
        weather_forecast = self._weather.next_hours(origin)
        features = build_inference_frame(history, weather_forecast, origin)
        forecast = predict_forecast(stored.model, features, origin, stored.version)
        self._predictions.save(forecast)
        return forecast

    def _explain(self, forecast: Forecast, summary) -> Forecast:
        """Adds `explanation`, `provider` and `ai_model`; if the AI provider fails, uses the mock (the forecast does not fail)."""
        # TODO (step 4.4): `text = ai_client.explain(summary)` → provider = ai_client.name, ai_model = ai_client.model.
        # TODO (step 4.4): on `ExternalServiceError`: warning log, `MockClient().explain(summary)`, provider = FALLBACK_PROVIDER.
        raise NotImplementedError
