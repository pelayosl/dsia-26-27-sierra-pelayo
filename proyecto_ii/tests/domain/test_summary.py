"""Tests of `domain.summary`: the figures the LLM will write up, computed with known forecasts."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from energy_forecast.domain.models import Forecast, ForecastSummary, HourlyPrediction
from energy_forecast.domain.schema import HORIZON_HOURS
from energy_forecast.domain.summary import summarize_forecast

# 23:00 UTC of Sunday 5 January 2025 → forecast for Monday 6 January (Epiphany, national holiday), local time
HOLIDAY_ORIGIN = datetime(2025, 1, 5, 23, tzinfo=timezone.utc)
# 23:00 UTC of Tuesday 7 January → forecast for Wednesday 8 January, an ordinary working day
WORKING_DAY_ORIGIN = datetime(2025, 1, 7, 23, tzinfo=timezone.utc)
MODEL_MAE = 640.4


def make_forecast(values, origin: datetime = WORKING_DAY_ORIGIN) -> Forecast:
    predictions = tuple(
        HourlyPrediction(time_utc=origin + timedelta(hours=i + 1), predicted_mw=float(value))
        for i, value in enumerate(values)
    )
    return Forecast(origin=origin, predictions=predictions, model_version="hgb-test")


def last_week_history(forecast: Forecast, values) -> pd.DataFrame:
    """Observations of the same 24 hours one week before the forecast (`time_utc` + `load_mw`)."""
    times = [p.time_utc - timedelta(hours=168) for p in forecast.predictions]
    return pd.DataFrame({"time_utc": pd.to_datetime(times, utc=True), "load_mw": values})


def weather_for(forecast: Forecast, temperature=12.0) -> pd.DataFrame:
    times = [p.time_utc for p in forecast.predictions]
    return pd.DataFrame({"time_utc": pd.to_datetime(times, utc=True), "temperature_2m": temperature})


def summarize(forecast: Forecast, history: pd.DataFrame | None = None, weather: pd.DataFrame | None = None):
    history = history if history is not None else last_week_history(forecast, [30_000.0] * HORIZON_HOURS)
    weather = weather if weather is not None else weather_for(forecast)
    return summarize_forecast(forecast, history, weather, MODEL_MAE)


FLAT = [30_000.0] * HORIZON_HOURS


# ---------------------------------------------------------------- peak, valley and mean


def test_summary_peak_valley_and_mean():
    # Arrange: flat 30 000 MW with a peak in position 19 and a valley in position 3
    values = list(FLAT)
    values[19], values[3] = 40_000.0, 20_000.0
    forecast = make_forecast(values)

    # Act
    summary = summarize(forecast)

    # Assert: position i is the hour origin + i + 1 UTC; in winter Madrid is UTC+1
    assert isinstance(summary, ForecastSummary)
    assert (summary.peak_mw, summary.valley_mw, summary.mean_mw) == (40_000, 20_000, 30_000)
    assert summary.peak_time_local.replace(tzinfo=None) == datetime(2025, 1, 8, 20, 0)
    assert summary.valley_time_local.replace(tzinfo=None) == datetime(2025, 1, 8, 4, 0)
    assert summary.peak_time_local.utcoffset() == timedelta(hours=1)


def test_period_is_reported_in_local_time():
    summary = summarize(make_forecast(FLAT))

    assert summary.period_start_local.replace(tzinfo=None) == datetime(2025, 1, 8, 1, 0)
    assert summary.period_end_local.replace(tzinfo=None) == datetime(2025, 1, 9, 0, 0)


def test_figures_are_rounded_for_the_explanation():
    values = [30_000.2 + i * 0.01 for i in range(HORIZON_HOURS)]  # mean 30 000.315

    summary = summarize(make_forecast(values), weather=weather_for(make_forecast(values), temperature=12.345))

    assert summary.mean_mw == 30_000
    assert summary.mean_temperature_c == 12.3
    assert summary.model_mae_mw == 640


def test_ties_take_the_first_hour():
    summary = summarize(make_forecast(FLAT))

    assert summary.peak_time_local == summary.valley_time_local == summary.period_start_local


# ---------------------------------------------------------------- change vs last week


def test_summary_change_vs_last_week():
    # Last week 10 % less → this week is 1 / 0.9 - 1 = +11.1 %
    forecast = make_forecast(FLAT)

    summary = summarize(forecast, history=last_week_history(forecast, [27_000.0] * HORIZON_HOURS))

    assert summary.change_vs_last_week_pct == pytest.approx(11.1)


def test_change_vs_last_week_can_be_negative():
    forecast = make_forecast(FLAT)

    summary = summarize(forecast, history=last_week_history(forecast, [33_333.3] * HORIZON_HOURS))

    assert summary.change_vs_last_week_pct == pytest.approx(-10.0)


def test_change_vs_last_week_compares_only_hours_known_last_week():
    # Forecast: 10 000 the first 12 hours, 40 000 the last 12. Last week: unknown, then 40 000.
    forecast = make_forecast([10_000.0] * 12 + [40_000.0] * 12)
    last_week = [np.nan] * 12 + [40_000.0] * 12

    summary = summarize(forecast, history=last_week_history(forecast, last_week))

    # Only the last 12 hours are compared (40 000 vs 40 000 → 0 %). Comparing the whole means
    # (25 000 vs 40 000) would report a false −37.5 %.
    assert summary.change_vs_last_week_pct == pytest.approx(0.0)


def test_change_vs_last_week_ignores_history_outside_those_hours():
    forecast = make_forecast(FLAT)
    history = pd.concat([
        last_week_history(forecast, [30_000.0] * HORIZON_HOURS),
        pd.DataFrame({"time_utc": [pd.Timestamp(WORKING_DAY_ORIGIN)], "load_mw": [99_999.0]}),  # the origin itself
    ])

    assert summarize(forecast, history=history).change_vs_last_week_pct == pytest.approx(0.0)


@pytest.mark.parametrize("history_values", [[np.nan] * HORIZON_HOURS, None], ids=["all-unknown", "empty"])
def test_change_vs_last_week_is_none_without_last_week_data(history_values):
    forecast = make_forecast(FLAT)
    if history_values is None:
        history = pd.DataFrame({"time_utc": pd.to_datetime([], utc=True), "load_mw": []})
    else:
        history = last_week_history(forecast, history_values)

    assert summarize(forecast, history=history).change_vs_last_week_pct is None


# ---------------------------------------------------------------- weather and calendar


def test_mean_temperature_uses_only_the_forecast_hours():
    forecast = make_forecast(FLAT)
    weather = pd.concat([
        weather_for(forecast, temperature=10.0),
        pd.DataFrame({"time_utc": [pd.Timestamp(WORKING_DAY_ORIGIN)], "temperature_2m": [40.0]}),  # outside
    ])

    assert summarize(forecast, weather=weather).mean_temperature_c == 10.0


def test_weather_without_the_forecast_hours_raises():
    forecast = make_forecast(FLAT)
    other_day = weather_for(make_forecast(FLAT, origin=HOLIDAY_ORIGIN))

    with pytest.raises(ValueError, match="no temperature"):
        summarize(forecast, weather=other_day)


def test_holiday_is_detected():
    summary = summarize(make_forecast(FLAT, origin=HOLIDAY_ORIGIN))

    assert summary.is_holiday is True  # 6 January, Epiphany
    assert summary.is_weekend is False  # Monday


def test_weekend_is_detected():
    saturday = datetime(2025, 1, 10, 23, tzinfo=timezone.utc)  # → Saturday 11 January, local time

    summary = summarize(make_forecast(FLAT, origin=saturday))

    assert (summary.is_weekend, summary.is_holiday) == (True, False)


def test_ordinary_working_day():
    summary = summarize(make_forecast(FLAT))

    assert (summary.is_weekend, summary.is_holiday) == (False, False)


def test_period_touching_a_weekend_counts_as_weekend():
    # 19:00 UTC on Friday 10 January → 21:00 local Friday ... 20:00 local Saturday
    friday_evening = datetime(2025, 1, 10, 19, tzinfo=timezone.utc)

    assert summarize(make_forecast(FLAT, origin=friday_evening)).is_weekend is True


# ---------------------------------------------------------------- serialisation


def test_summary_serialises_to_json_for_the_llm():
    summary = summarize(make_forecast(FLAT))

    payload = summary.model_dump(mode="json")

    assert payload["peak_time_local"] == "2025-01-08T01:00:00+01:00"
    assert payload["change_vs_last_week_pct"] == 0.0
    assert set(payload) == set(ForecastSummary.model_fields)
