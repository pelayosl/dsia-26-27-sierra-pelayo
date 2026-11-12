"""Tests of `domain.predictor`: from the 24 feature rows to a `Forecast`."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from energy_forecast.domain.features import FEATURE_COLUMNS, build_inference_frame
from energy_forecast.domain.modeling import SeasonalNaiveModel
from energy_forecast.domain.models import Forecast
from energy_forecast.domain.predictor import predict_forecast
from energy_forecast.domain.schema import HORIZON_HOURS
from energy_forecast.domain.validator import validate_observations
from tests.conftest import FORECAST_ORIGIN

MODEL_VERSION = "hgb-test"


@pytest.fixture
def features(hourly_observations, weather_forecast_24h) -> pd.DataFrame:
    """The 24 inference rows after `FORECAST_ORIGIN`, as the forecast service builds them."""
    history, _ = validate_observations(hourly_observations)
    return build_inference_frame(history, weather_forecast_24h, FORECAST_ORIGIN)


class ConstantModel:
    """Returns fixed values, to test what `predict_forecast` does with them."""

    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=float)
        self.received_columns: list[str] = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.received_columns = list(X.columns)
        return self.values


def test_predict_forecast_returns_24_hourly_predictions(small_trained_model, features):
    # Act
    forecast = predict_forecast(small_trained_model, features, FORECAST_ORIGIN, MODEL_VERSION)

    # Assert
    assert isinstance(forecast, Forecast)
    assert len(forecast.predictions) == HORIZON_HOURS
    assert forecast.origin == FORECAST_ORIGIN
    assert forecast.predictions[0].time_utc == FORECAST_ORIGIN + timedelta(hours=1)
    assert forecast.predictions[-1].time_utc == FORECAST_ORIGIN + timedelta(hours=HORIZON_HOURS)
    assert forecast.model_version == MODEL_VERSION
    assert forecast.explanation is None
    assert forecast.provider is None


def test_predicted_values_are_the_model_output(small_trained_model, features):
    forecast = predict_forecast(small_trained_model, features, FORECAST_ORIGIN, MODEL_VERSION)

    expected = small_trained_model.predict(features[FEATURE_COLUMNS])
    assert [p.predicted_mw for p in forecast.predictions] == pytest.approx(expected.tolist())


def test_features_are_passed_in_training_order(features):
    model = ConstantModel([30_000.0] * HORIZON_HOURS)
    shuffled = features[list(reversed(FEATURE_COLUMNS))].assign(extra_column=1)

    predict_forecast(model, shuffled, FORECAST_ORIGIN, MODEL_VERSION)

    assert model.received_columns == FEATURE_COLUMNS  # same order, without extra columns


def test_origin_in_another_timezone_is_stored_in_utc(features):
    madrid_origin = pd.Timestamp(FORECAST_ORIGIN).tz_convert("Europe/Madrid").to_pydatetime()

    forecast = predict_forecast(ConstantModel([1.0] * HORIZON_HOURS), features, madrid_origin, MODEL_VERSION)

    assert forecast.origin.utcoffset() == timedelta(0)
    assert forecast.origin == FORECAST_ORIGIN


def test_forecast_is_serialised_with_the_report_shape(features):
    forecast = predict_forecast(ConstantModel(np.arange(HORIZON_HOURS) + 25_000.0), features, FORECAST_ORIGIN, "v1")

    report = forecast.model_dump(mode="json")

    assert report["predictions"][0] == {"time_utc": "2025-01-22T00:00:00Z", "predicted_mw": 25_000.0}
    assert report["origin"] == "2025-01-21T23:00:00Z"


@pytest.mark.parametrize("rows", [23, 25])
def test_wrong_number_of_rows_raises(features, rows):
    frame = pd.concat([features, features]).iloc[:rows]

    with pytest.raises(ValueError, match=f"got {rows}"):
        predict_forecast(ConstantModel([1.0] * rows), frame, FORECAST_ORIGIN, MODEL_VERSION)


def test_missing_feature_column_raises(features):
    with pytest.raises(ValueError, match="lag_168"):
        predict_forecast(ConstantModel([1.0] * HORIZON_HOURS), features.drop(columns="lag_168"), FORECAST_ORIGIN,
                         MODEL_VERSION)


def test_non_finite_predictions_raise(features):
    # The naive baseline predicts NaN when `lag_168` is unknown (e.g. a blackout a week ago)
    frame = features.copy()
    frame.loc[frame.index[5], "lag_168"] = np.nan
    model = SeasonalNaiveModel().fit(frame)

    with pytest.raises(ValueError, match="invalid predictions"):
        predict_forecast(model, frame, FORECAST_ORIGIN, MODEL_VERSION)


def test_negative_predictions_are_rejected(features):
    values = [30_000.0] * HORIZON_HOURS
    values[3] = -10.0

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        predict_forecast(ConstantModel(values), features, FORECAST_ORIGIN, MODEL_VERSION)


def test_features_for_another_origin_are_rejected(features):
    with pytest.raises(ValidationError, match="must be"):
        predict_forecast(ConstantModel([1.0] * HORIZON_HOURS), features, FORECAST_ORIGIN + timedelta(hours=1),
                         MODEL_VERSION)


def test_naive_origin_raises(features):
    with pytest.raises(ValueError, match="timezone-aware"):
        predict_forecast(ConstantModel([1.0] * HORIZON_HOURS), features, datetime(2025, 1, 21, 23), MODEL_VERSION)


def test_predict_forecast_with_the_real_baseline(features):
    forecast = predict_forecast(SeasonalNaiveModel().fit(features), features, FORECAST_ORIGIN, "naive-test")

    assert [p.predicted_mw for p in forecast.predictions] == pytest.approx(features["lag_168"].tolist())
    assert forecast.predictions[0].time_utc.utcoffset() == timedelta(0)
