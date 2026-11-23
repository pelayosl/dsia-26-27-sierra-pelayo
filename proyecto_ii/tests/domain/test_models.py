"""Tests of the `Forecast` contract (`domain.models`): 24 consecutive UTC hours after the origin."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from energy_forecast.domain.models import Forecast, HourlyPrediction
from energy_forecast.domain.schema import HORIZON_HOURS

ORIGIN = datetime(2026, 11, 12, 0, 0, tzinfo=timezone.utc)


def predictions(origin: datetime = ORIGIN, hours: int = HORIZON_HOURS, first: int = 1) -> tuple[HourlyPrediction, ...]:
    return tuple(
        HourlyPrediction(time_utc=origin + timedelta(hours=first + i), predicted_mw=30_000.0) for i in range(hours)
    )


def test_valid_forecast():
    forecast = Forecast(origin=ORIGIN, predictions=predictions(), model_version="v1")

    assert len(forecast.predictions) == HORIZON_HOURS


@pytest.mark.parametrize("hours", [23, 25])
def test_forecast_needs_exactly_24_predictions(hours):
    with pytest.raises(ValidationError):
        Forecast(origin=ORIGIN, predictions=predictions(hours=hours), model_version="v1")


def test_predictions_must_start_one_hour_after_origin():
    with pytest.raises(ValidationError, match=r"predictions\[0\].time_utc must be 2026-11-12T01:00:00\+00:00"):
        Forecast(origin=ORIGIN, predictions=predictions(first=2), model_version="v1")


def test_predictions_must_be_consecutive():
    hours = list(predictions())
    hours[10] = HourlyPrediction(time_utc=hours[11].time_utc, predicted_mw=30_000.0)  # repeated hour

    with pytest.raises(ValidationError, match=r"predictions\[10\]"):
        Forecast(origin=ORIGIN, predictions=tuple(hours), model_version="v1")


@pytest.mark.parametrize(
    "origin",
    [datetime(2026, 11, 12, 0, 0), datetime(2026, 11, 12, 1, 0, tzinfo=timezone(timedelta(hours=1)))],
    ids=["naive", "madrid"],
)
def test_origin_must_be_utc(origin):
    with pytest.raises(ValidationError, match="origin must be in UTC"):
        Forecast(origin=origin, predictions=predictions(), model_version="v1")
