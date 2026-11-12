"""Use case: 24 h forecast with explanation. Used by the `forecast` CLI (Project II) and the API (Project III)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from energy_forecast.domain.errors import ExternalServiceError, InsufficientHistoryError
from energy_forecast.domain.features import build_inference_frame
from energy_forecast.domain.models import Forecast
from energy_forecast.domain.predictor import predict_forecast
from energy_forecast.domain.schema import MIN_HISTORY_HOURS
from energy_forecast.domain.summary import summarize_forecast
from energy_forecast.infrastructure.ai_client import AIClient, MockClient
from energy_forecast.infrastructure.model_store import ModelStore
from energy_forecast.infrastructure.repositories import (
    ObservationRepository,
    PipelineRunRepository,
    PredictionRepository,
)
from energy_forecast.infrastructure.weather_client import WeatherForecastClient
from energy_forecast.logging_config import get_logger

logger = get_logger("forecast")

FALLBACK_PROVIDER = "mock-fallback"


@dataclass(frozen=True)
class _Prediction:
    """What the explanation needs from the prediction step, besides the `Forecast` itself."""
    forecast: Forecast
    history: pd.DataFrame
    weather_forecast: pd.DataFrame
    model_metrics: dict[str, Any]


def _model_mae(metrics: dict[str, Any]) -> float | None:
    """Test MAE of the saved model, from the metrics saved by `train` (`None` if they are missing)."""
    try:
        return float(metrics["metrics"][metrics["best_model"]]["mae"])
    except (KeyError, TypeError, ValueError):
        return None


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
        self._ai_client = ai_client if ai_client is not None else MockClient()

    def forecast(self, origin: datetime | None = None, explain: bool = True) -> Forecast:
        """Predicts the 24 hours after `origin` (by default, the last hour with known demand), saves them
        and, if `explain`, adds the AI explanation.

        The run is recorded in `pipeline_runs`. On any error it is marked as `error` and the exception is
        re-raised: `ModelNotFoundError` (no trained model), `InsufficientHistoryError` (not enough history
        before `origin`), `ExternalServiceError` (weather forecast unavailable), `ValueError` (invalid origin).
        A failure of the AI provider never fails the forecast: the mock explanation is used instead.
        """
        run_id = self._runs.start("forecast")
        try:
            prediction = self._predict(origin)
            forecast = self._explain(prediction) if explain else prediction.forecast
        except Exception as exc:
            self._runs.finish(run_id, "error", error=f"{type(exc).__name__}: {exc}")
            logger.error("Forecast failed (run %s): %s", run_id, exc)
            raise

        self._runs.finish(run_id, "ok", rows=len(forecast.predictions))
        logger.info("Forecast finished (run %s): origin %s, model %s, explanation by %s",
                    run_id, forecast.origin.isoformat(), forecast.model_version, forecast.provider)
        return forecast

    def _predict(self, origin: datetime | None) -> _Prediction:
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
        # The predictions are saved before the explanation: they do not depend on the AI provider
        self._predictions.save(forecast)
        return _Prediction(forecast, history, weather_forecast, stored.metrics)

    def _explain(self, prediction: _Prediction) -> Forecast:
        """Adds `explanation`, `provider` and `ai_model`; if the AI provider fails, uses the mock (§13)."""
        summary = summarize_forecast(
            prediction.forecast, prediction.history, prediction.weather_forecast, _model_mae(prediction.model_metrics)
        )
        try:
            text = self._ai_client.explain(summary)
            provider, ai_model = self._ai_client.name, self._ai_client.model
        except ExternalServiceError as exc:
            logger.warning("AI provider '%s' failed, using the mock explanation: %s", self._ai_client.name, exc)
            text = MockClient().explain(summary)
            provider, ai_model = FALLBACK_PROVIDER, None
        return prediction.forecast.model_copy(update={"explanation": text, "provider": provider, "ai_model": ai_model})
