"""Tests of `domain.features`: above all, that no feature uses data after the origin."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from energy_forecast.domain.errors import InsufficientHistoryError, ValidationError
from energy_forecast.domain.features import (
    FEATURE_COLUMNS,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_inference_frame,
    build_training_frame,
)
from energy_forecast.domain.schema import HORIZON_HOURS, MIN_HISTORY_HOURS, WEATHER_COLUMNS
from energy_forecast.domain.validator import validate_observations
from tests.conftest import FORECAST_ORIGIN, FORECAST_TEMPERATURE_C, OBSERVATIONS_HOURS


@pytest.fixture
def observations(hourly_observations):
    """`hourly_observations` after validation (UTC-aware `time_utc`), as the repository returns them."""
    valid, _ = validate_observations(hourly_observations)
    return valid


def load_at(frame: pd.DataFrame, time: pd.Timestamp) -> float:
    return frame.loc[frame["time_utc"] == time, "load_mw"].item()


def row_at(frame: pd.DataFrame, time: pd.Timestamp) -> pd.Series:
    return frame.loc[frame["time_utc"] == time].iloc[0]


def local_hours_frame(start_utc: str, hours: int) -> pd.DataFrame:
    return pd.DataFrame({"time_utc": pd.date_range(start_utc, periods=hours, freq="h", tz="UTC")})


# ---------------------------------------------------------------- lags


@pytest.mark.parametrize("lag", [24, 48, 168])
def test_lag_equals_load_k_hours_before(observations, lag):
    # Arrange
    target = observations["time_utc"].iloc[200]

    # Act
    result = add_lag_features(observations)

    # Assert
    assert row_at(result, target)[f"lag_{lag}"] == load_at(observations, target - pd.Timedelta(hours=lag))


def test_lags_are_nan_at_the_start(observations):
    result = add_lag_features(observations)

    assert result["lag_24"].iloc[:24].isna().all()
    assert result["lag_24"].iloc[24:].notna().all()


def test_lags_use_time_not_row_position_when_there_are_gaps(observations):
    # A missing row (rejected by the validator) must not shift the lags of later rows
    target = observations["time_utc"].iloc[200]
    missing = target - pd.Timedelta(hours=10)
    with_gap = observations[observations["time_utc"] != missing]

    result = add_lag_features(with_gap)

    assert row_at(result, target)["lag_24"] == load_at(observations, target - pd.Timedelta(hours=24))
    assert np.isnan(row_at(result, missing + pd.Timedelta(hours=24))["lag_24"])


def test_lag_features_do_not_mutate_input(observations):
    before = observations.copy()

    add_lag_features(observations)

    pd.testing.assert_frame_equal(observations, before)


# ---------------------------------------------------------------- rolling means


def test_rolling_window_ends_at_t_minus_24(observations):
    target = observations["time_utc"].iloc[200]
    window = observations[
        (observations["time_utc"] >= target - pd.Timedelta(hours=47))
        & (observations["time_utc"] <= target - pd.Timedelta(hours=24))
    ]

    result = add_rolling_features(observations)

    assert len(window) == 24
    assert row_at(result, target)["rolling_mean_24"] == pytest.approx(window["load_mw"].mean())


def test_rolling_mean_168_is_known_once_the_window_is_half_full(observations):
    result = add_rolling_features(observations)

    # The window of t ends at t - 24, so it has data from t = 24; it needs 84 known hours
    assert result["rolling_mean_168"].iloc[: 24 + 84 - 1].isna().all()
    assert result["rolling_mean_168"].iloc[24 + 84 - 1 :].notna().all()


def test_rolling_mean_ignores_unknown_hours(observations):
    target = observations["time_utc"].iloc[200]
    blackout = target - pd.Timedelta(hours=30)
    frame = observations.copy()
    frame.loc[frame["time_utc"] == blackout, "load_mw"] = np.nan
    window = frame[
        (frame["time_utc"] >= target - pd.Timedelta(hours=47)) & (frame["time_utc"] <= target - pd.Timedelta(hours=24))
    ]

    result = add_rolling_features(frame)

    assert row_at(result, target)["rolling_mean_24"] == pytest.approx(window["load_mw"].mean())  # mean of 23


# ---------------------------------------------------------------- calendar


def test_calendar_features_use_local_time():
    # 2025-03-30: Spain moves from UTC+1 to UTC+2 at 01:00 UTC (02:00 local does not exist)
    frame = local_hours_frame("2025-03-29T22:00:00", 5)

    result = add_calendar_features(frame)

    assert result["hour"].tolist() == [23, 0, 1, 3, 4]
    assert result["weekday"].tolist() == [5, 6, 6, 6, 6]  # Saturday 29 → Sunday 30 at local midnight
    assert result["is_weekend"].tolist() == [1, 1, 1, 1, 1]


def test_calendar_features_on_autumn_dst_change_repeat_local_hour():
    # 2025-10-26: 03:00 local goes back to 02:00, so local hour 2 happens twice
    frame = local_hours_frame("2025-10-25T23:00:00", 4)

    result = add_calendar_features(frame)

    assert result["hour"].tolist() == [1, 2, 2, 3]


def test_is_holiday_is_computed_when_missing():
    # 6 January (Epiphany) is a national holiday; 7 January is not
    frame = local_hours_frame("2025-01-05T23:00:00", 48)  # local 6 Jan 00:00 → 7 Jan 23:00

    result = add_calendar_features(frame)

    assert result["is_holiday"].tolist() == [1] * 24 + [0] * 24


def test_is_holiday_from_dataset_is_kept_and_gaps_are_filled():
    frame = local_hours_frame("2025-01-05T23:00:00", 3)
    frame["is_holiday"] = [0, np.nan, np.nan]  # the dataset says 0 for the first hour

    result = add_calendar_features(frame)

    assert result["is_holiday"].tolist() == [0, 1, 1]
    assert result["is_holiday"].dtype == int


def test_calendar_month_and_weekday(observations):
    result = add_calendar_features(observations)
    first = result.iloc[0]  # 2025-01-06 01:00 local, a Monday

    assert (first["hour"], first["weekday"], first["month"], first["is_weekend"]) == (1, 0, 1, 0)


# ---------------------------------------------------------------- training frame


def test_training_frame_excludes_warm_up(observations):
    X, y = build_training_frame(observations)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == OBSERVATIONS_HOURS - MIN_HISTORY_HOURS
    assert X.index[0] == observations["time_utc"].iloc[0] + pd.Timedelta(hours=MIN_HISTORY_HOURS)
    assert X.index.equals(y.index)
    assert X.notna().all().all()


def test_training_frame_excludes_nan_target_and_rows_with_unknown_lags(observations):
    blackout = observations["time_utc"].iloc[MIN_HISTORY_HOURS + 5]
    frame = observations.copy()
    frame.loc[frame["time_utc"] == blackout, "load_mw"] = np.nan

    X, y = build_training_frame(frame)

    assert y.notna().all()
    assert blackout not in y.index
    assert blackout + pd.Timedelta(hours=24) not in X.index  # its `lag_24` is the blackout hour
    assert len(X) == OBSERVATIONS_HOURS - MIN_HISTORY_HOURS - 2


def test_training_frame_with_invalid_observations(invalid_observations):
    valid, _ = validate_observations(invalid_observations)

    X, y = build_training_frame(valid)

    assert y.notna().all()
    assert X.notna().all().all()
    assert len(X) > 0


def test_training_frame_does_not_depend_on_row_order(observations):
    X_sorted, y_sorted = build_training_frame(observations)
    X_shuffled, y_shuffled = build_training_frame(observations.sample(frac=1, random_state=0))

    pd.testing.assert_frame_equal(X_sorted, X_shuffled)
    pd.testing.assert_series_equal(y_sorted, y_shuffled)


def test_training_frame_of_empty_observations(observations):
    X, y = build_training_frame(observations.iloc[0:0])

    assert X.empty and y.empty
    assert list(X.columns) == FEATURE_COLUMNS


# ---------------------------------------------------------------- inference frame


def test_inference_frame_has_24_rows_and_feature_columns(observations, weather_forecast_24h):
    X = build_inference_frame(observations, weather_forecast_24h, FORECAST_ORIGIN)

    assert len(X) == HORIZON_HOURS
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index[0] == FORECAST_ORIGIN + pd.Timedelta(hours=1)
    assert X.index[-1] == FORECAST_ORIGIN + pd.Timedelta(hours=HORIZON_HOURS)
    assert X.notna().all().all()
    assert (X["temperature_2m"] == FORECAST_TEMPERATURE_C).all()
    assert (X["covid_flag"] == 0).all()


def test_inference_lags_come_from_history(observations, weather_forecast_24h):
    X = build_inference_frame(observations, weather_forecast_24h, FORECAST_ORIGIN)

    # Last target hour (origin + 24): its lag_24 is the demand at the origin
    assert X["lag_24"].iloc[-1] == load_at(observations, FORECAST_ORIGIN)
    assert X["lag_24"].iloc[0] == load_at(observations, FORECAST_ORIGIN - pd.Timedelta(hours=23))


def test_inference_frame_ignores_data_after_origin(observations, weather_forecast_24h):
    origin = FORECAST_ORIGIN - pd.Timedelta(hours=HORIZON_HOURS)
    weather = weather_forecast_24h.assign(time_utc=weather_forecast_24h["time_utc"] - pd.Timedelta(hours=24))
    future_changed = observations.copy()
    future_changed.loc[future_changed["time_utc"] > origin, "load_mw"] = 99_999.0

    X = build_inference_frame(observations, weather, origin)
    X_future_changed = build_inference_frame(future_changed, weather, origin)

    pd.testing.assert_frame_equal(X, X_future_changed)


def test_inference_features_match_training_features(observations):
    """The same code for training and prediction (D7): same features for the same hour."""
    origin = FORECAST_ORIGIN - pd.Timedelta(hours=HORIZON_HOURS)
    target_hours = observations[observations["time_utc"] > origin]
    weather = target_hours[["time_utc", *WEATHER_COLUMNS]]

    X_inference = build_inference_frame(observations, weather, origin)
    X_training, _ = build_training_frame(observations)

    pd.testing.assert_frame_equal(X_inference, X_training.loc[X_inference.index], check_dtype=False, check_freq=False)


def test_inference_frame_without_enough_history_raises(observations, weather_forecast_24h):
    short_history = observations.tail(MIN_HISTORY_HOURS - 1)

    with pytest.raises(InsufficientHistoryError):
        build_inference_frame(short_history, weather_forecast_24h, FORECAST_ORIGIN)


def test_inference_frame_with_unknown_demand_at_origin_raises(observations, weather_forecast_24h):
    frame = observations.copy()
    frame.loc[frame["time_utc"] == FORECAST_ORIGIN, "load_mw"] = np.nan

    with pytest.raises(InsufficientHistoryError):
        build_inference_frame(frame, weather_forecast_24h, FORECAST_ORIGIN)


def test_inference_frame_with_incomplete_weather_raises(observations, weather_forecast_24h):
    with pytest.raises(ValidationError, match="misses 1 target hours"):
        build_inference_frame(observations, weather_forecast_24h.iloc[:-1], FORECAST_ORIGIN)


@pytest.mark.parametrize(
    "origin",
    [datetime(2025, 1, 21, 23), FORECAST_ORIGIN + pd.Timedelta(minutes=30)],
    ids=["naive", "not-on-the-hour"],
)
def test_inference_frame_with_invalid_origin_raises(observations, weather_forecast_24h, origin):
    with pytest.raises(ValueError, match="origin"):
        build_inference_frame(observations, weather_forecast_24h, origin)


def test_inference_frame_accepts_origin_in_other_timezone(observations, weather_forecast_24h):
    madrid_origin = FORECAST_ORIGIN.tz_convert("Europe/Madrid")

    X = build_inference_frame(observations, weather_forecast_24h, madrid_origin)

    assert X.index[0] == FORECAST_ORIGIN + pd.Timedelta(hours=1)
