"""Tests of `ForecastService` with fakes (no network or database)."""

from datetime import timedelta

import pytest

from energy_forecast.domain.errors import ExternalServiceError, InsufficientHistoryError, ModelNotFoundError
from energy_forecast.domain.schema import HORIZON_HOURS, MIN_HISTORY_HOURS
from energy_forecast.domain.validator import validate_observations
from energy_forecast.infrastructure.model_store import StoredModel
from energy_forecast.services.forecast import ForecastService
from tests.conftest import FORECAST_ORIGIN
from tests.fakes import (
    FakeModelStore,
    FakeWeatherForecastClient,
    InMemoryObservationRepository,
    InMemoryPipelineRunRepository,
    InMemoryPredictionRepository,
)

MODEL_VERSION = "hgb-test-v1"


@pytest.fixture
def observations(hourly_observations) -> InMemoryObservationRepository:
    repository = InMemoryObservationRepository()
    repository.upsert(validate_observations(hourly_observations)[0])
    return repository


@pytest.fixture
def store(small_trained_model) -> FakeModelStore:
    return FakeModelStore(StoredModel(model=small_trained_model, version=MODEL_VERSION, metrics={}))


@pytest.fixture
def weather(weather_forecast_24h) -> FakeWeatherForecastClient:
    return FakeWeatherForecastClient(weather_forecast_24h)


@pytest.fixture
def predictions() -> InMemoryPredictionRepository:
    return InMemoryPredictionRepository()


@pytest.fixture
def runs() -> InMemoryPipelineRunRepository:
    return InMemoryPipelineRunRepository()


@pytest.fixture
def service(observations, weather, store, predictions, runs) -> ForecastService:
    return ForecastService(observations, weather, store, predictions, runs)


def test_forecast_returns_24_predictions_and_saves_them(service, predictions):
    forecast = service.forecast()

    assert len(forecast.predictions) == HORIZON_HOURS
    assert forecast.model_version == MODEL_VERSION
    assert predictions.saved == [forecast]


def test_default_origin_is_the_last_hour_with_known_demand(service, weather):
    forecast = service.forecast()

    assert forecast.origin == FORECAST_ORIGIN
    assert weather.origins == [FORECAST_ORIGIN]
    assert forecast.predictions[0].time_utc == FORECAST_ORIGIN + timedelta(hours=1)


def test_explicit_origin_is_used(service, weather):
    origin = FORECAST_ORIGIN - timedelta(hours=10)

    forecast = service.forecast(origin=origin)

    assert forecast.origin == origin
    assert weather.origins == [origin]


def test_forecast_values_are_the_model_output(service, small_trained_model):
    forecast = service.forecast()

    assert all(prediction.predicted_mw > 0 for prediction in forecast.predictions)
    assert len({prediction.predicted_mw for prediction in forecast.predictions}) > 1  # depends on the hour


def test_forecast_run_is_recorded(service, runs):
    service.forecast()

    run = runs.get("run-1")
    assert (run["kind"], run["status"], run["rows"]) == ("forecast", "ok", HORIZON_HOURS)


def test_forecast_without_model_raises_before_calling_the_weather_service(observations, weather, predictions, runs):
    service = ForecastService(observations, weather, FakeModelStore(None), predictions, runs)

    with pytest.raises(ModelNotFoundError):
        service.forecast()

    assert weather.origins == []  # no network call when there is no model
    assert runs.get("run-1")["status"] == "error"


def test_forecast_without_observations_raises(store, weather, predictions, runs):
    service = ForecastService(InMemoryObservationRepository(), weather, store, predictions, runs)

    with pytest.raises(InsufficientHistoryError, match="run `ingest` first"):
        service.forecast()


def test_origin_without_enough_history_raises(service, predictions, runs):
    too_early = FORECAST_ORIGIN - timedelta(hours=HORIZON_HOURS + MIN_HISTORY_HOURS)

    with pytest.raises(InsufficientHistoryError):
        service.forecast(origin=too_early)

    assert predictions.saved == []
    assert "InsufficientHistoryError" in runs.get("run-1")["error"]


def test_origin_after_the_last_observation_raises(service):
    with pytest.raises(InsufficientHistoryError):
        service.forecast(origin=FORECAST_ORIGIN + timedelta(hours=5))


def test_weather_service_failure_is_propagated_and_nothing_is_saved(observations, store, predictions, runs):
    weather = FakeWeatherForecastClient(error=ExternalServiceError("Open-Meteo answered 503"))
    service = ForecastService(observations, weather, store, predictions, runs)

    with pytest.raises(ExternalServiceError):
        service.forecast()

    assert predictions.saved == []
    assert runs.get("run-1")["status"] == "error"


# ---------------------------------------------------------------- AI explanation (step 4.4)


@pytest.mark.skip(reason="TODO (paso 4.4): explicación con MockClient")
def test_forecast_with_mock_explanation(small_trained_model, hourly_observations, weather_forecast_24h):
    # TODO: non-empty `explanation` and `provider == "mock"`.
    ...


@pytest.mark.skip(reason="TODO (paso 4.4): fallback a mock")
def test_forecast_falls_back_to_mock_when_ai_fails(small_trained_model, hourly_observations, weather_forecast_24h):
    # TODO: `FailingAIClient` → the forecast is still returned with `provider == "mock-fallback"`.
    ...
