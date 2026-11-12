"""Tests of `ForecastService` with fakes (no network or database)."""

from datetime import timedelta

import pytest

from energy_forecast.domain.errors import ExternalServiceError, InsufficientHistoryError, ModelNotFoundError
from energy_forecast.domain.schema import HORIZON_HOURS, MIN_HISTORY_HOURS
from energy_forecast.domain.validator import validate_observations
from energy_forecast.infrastructure.model_store import StoredModel
from energy_forecast.services.forecast import FALLBACK_PROVIDER, ForecastService
from tests.conftest import FORECAST_ORIGIN, FORECAST_TEMPERATURE_C
from tests.fakes import (
    FailingAIClient,
    FakeAIClient,
    FakeModelStore,
    FakeWeatherForecastClient,
    InMemoryObservationRepository,
    InMemoryPipelineRunRepository,
    InMemoryPredictionRepository,
)

MODEL_VERSION = "hgb-test-v1"
# The shape of the metrics that `train` saves with the model (only the fields the explanation reads)
SAVED_METRICS = {"best_model": "hgb", "metrics": {"hgb": {"mae": 640.4, "rmse": 930.0, "mape": 2.3}}}


@pytest.fixture
def observations(hourly_observations) -> InMemoryObservationRepository:
    repository = InMemoryObservationRepository()
    repository.upsert(validate_observations(hourly_observations)[0])
    return repository


@pytest.fixture
def store(small_trained_model) -> FakeModelStore:
    return FakeModelStore(StoredModel(model=small_trained_model, version=MODEL_VERSION, metrics=SAVED_METRICS))


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
    assert len(predictions.saved) == 1
    assert predictions.saved[0].predictions == forecast.predictions


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


# ---------------------------------------------------------------- AI explanation


def test_forecast_with_mock_explanation(service):
    # Without an AI client the service uses the mock (AI_PROVIDER=mock by default)
    forecast = service.forecast()

    assert forecast.explanation
    assert (forecast.provider, forecast.ai_model) == ("mock", None)
    assert "±640 MW" in forecast.explanation  # MAE read from the saved metrics


def test_forecast_with_the_real_provider(observations, weather, store, predictions, runs):
    ai_client = FakeAIClient("The peak is at 20:00.")
    service = ForecastService(observations, weather, store, predictions, runs, ai_client=ai_client)

    forecast = service.forecast()

    assert forecast.explanation == "The peak is at 20:00."
    assert (forecast.provider, forecast.ai_model) == ("nvidia", "fake/model")


def test_the_ai_client_receives_the_summary_of_this_forecast(observations, weather, store, predictions, runs):
    ai_client = FakeAIClient()
    service = ForecastService(observations, weather, store, predictions, runs, ai_client=ai_client)

    forecast = service.forecast()

    summary = ai_client.summaries[0]
    predicted = [p.predicted_mw for p in forecast.predictions]
    assert summary.peak_mw == round(max(predicted))
    assert summary.mean_temperature_c == pytest.approx(FORECAST_TEMPERATURE_C)
    assert summary.model_mae_mw == 640
    assert summary.change_vs_last_week_pct is not None  # the history covers the previous week


def test_forecast_falls_back_to_mock_when_ai_fails(observations, weather, store, predictions, runs, caplog):
    service = ForecastService(observations, weather, store, predictions, runs, ai_client=FailingAIClient())

    forecast = service.forecast()

    assert forecast.explanation  # the mock text
    assert (forecast.provider, forecast.ai_model) == (FALLBACK_PROVIDER, None)
    assert runs.get("run-1")["status"] == "ok"  # the AI provider is not in the critical path
    assert "using the mock explanation" in caplog.text


def test_forecast_without_explanation(observations, weather, store, predictions, runs):
    ai_client = FakeAIClient()
    service = ForecastService(observations, weather, store, predictions, runs, ai_client=ai_client)

    forecast = service.forecast(explain=False)

    assert (forecast.explanation, forecast.provider, forecast.ai_model) == (None, None, None)
    assert ai_client.summaries == []  # no call to the AI provider


def test_explanation_without_saved_metrics_omits_the_model_error(observations, weather, predictions, runs,
                                                                 small_trained_model):
    store = FakeModelStore(StoredModel(model=small_trained_model, version=MODEL_VERSION, metrics={}))
    ai_client = FakeAIClient()

    ForecastService(observations, weather, store, predictions, runs, ai_client=ai_client).forecast()

    assert ai_client.summaries[0].model_mae_mw is None


def test_predictions_are_saved_even_if_the_explanation_fails(observations, weather, store, predictions, runs):
    ForecastService(observations, weather, store, predictions, runs, ai_client=FailingAIClient()).forecast()

    assert len(predictions.saved) == 1
