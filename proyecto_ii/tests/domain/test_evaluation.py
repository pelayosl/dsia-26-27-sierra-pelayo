"""Tests of `domain.evaluation`."""

import numpy as np
import pandas as pd
import pytest

from energy_forecast.domain.evaluation import evaluate, mae, mae_by_hour, mape, rmse
from energy_forecast.domain.models import ModelMetrics


def hourly_series(values, start="2025-01-06T00:00:00+00:00") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h", name="time_utc"), dtype=float)


# ---------------------------------------------------------------- metrics


def test_metrics_with_known_result():
    # Arrange
    y_true, y_pred = [100.0, 200.0], [110.0, 190.0]

    # Act
    metrics = evaluate(y_true, y_pred)

    # Assert: errors of 10 and 10 → 10 % and 5 % → MAPE 7.5 %
    assert isinstance(metrics, ModelMetrics)
    assert (metrics.mae, metrics.rmse, metrics.mape) == pytest.approx((10.0, 10.0, 7.5))


def test_rmse_penalises_large_errors_more_than_mae():
    y_true, y_pred = [100.0, 100.0], [100.0, 120.0]

    assert mae(y_true, y_pred) == pytest.approx(10.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(200.0))  # sqrt((0 + 400) / 2)


def test_perfect_prediction_has_zero_error():
    y = [25_000.0, 30_000.0, 28_000.0]

    assert evaluate(y, y) == ModelMetrics(mae=0.0, rmse=0.0, mape=0.0)


def test_metrics_return_python_floats():
    assert type(mape([100.0], [110.0])) is float
    assert mape([100.0], [110.0]) == pytest.approx(10.0)  # percentage, not 0.1


@pytest.mark.parametrize("y_type", [list, np.array, pd.Series])
def test_metrics_accept_lists_arrays_and_series(y_type):
    assert mae(y_type([100.0, 200.0]), y_type([110.0, 190.0])) == pytest.approx(10.0)


def test_unknown_real_or_predicted_values_are_ignored():
    y_true = [100.0, np.nan, 200.0, 300.0]
    y_pred = [110.0, 999.0, 190.0, np.nan]

    assert evaluate(y_true, y_pred) == evaluate([100.0, 200.0], [110.0, 190.0])


def test_metrics_with_different_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        mae([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([], []), ([np.nan], [1.0])],
    ids=["empty", "all-unknown"],
)
def test_metrics_without_known_pairs_raise(y_true, y_pred):
    with pytest.raises(ValueError, match="no pairs"):
        evaluate(y_true, y_pred)


# ---------------------------------------------------------------- MAE by hour


def test_mae_by_hour_groups_by_local_hour():
    # Arrange: two days; the error equals the UTC hour + 1, so each local hour has a known error
    y_true = hourly_series([30_000.0] * 48)
    utc_hour = y_true.index.hour.to_numpy()
    y_pred = y_true.to_numpy() + utc_hour + 1

    # Act
    result = mae_by_hour(y_true, y_pred)

    # Assert: winter in Madrid is UTC+1, so local hour h is UTC hour h - 1
    assert sorted(result) == list(range(24))
    assert result[1] == pytest.approx(1.0)  # 00:00 UTC
    assert result[0] == pytest.approx(24.0)  # 23:00 UTC


def test_mae_by_hour_averages_days_and_ignores_unknown_values():
    y_true = hourly_series([100.0] * 48)
    y_pred = np.full(48, 100.0)
    y_pred[0], y_pred[24] = 110.0, 130.0  # same hour (01:00 local) on two days
    y_true.iloc[1] = np.nan  # 02:00 local of the first day: unknown real demand

    result = mae_by_hour(y_true, y_pred)

    assert result[1] == pytest.approx(20.0)
    assert result[2] == pytest.approx(0.0)  # only the second day counts


def test_mae_by_hour_keys_are_python_ints():
    result = mae_by_hour(hourly_series([1.0] * 24), np.ones(24))

    assert all(type(hour) is int and type(value) is float for hour, value in result.items())


@pytest.mark.parametrize(
    "index",
    [pd.RangeIndex(3), pd.date_range("2025-01-06", periods=3, freq="h")],
    ids=["not-datetime", "naive"],
)
def test_mae_by_hour_requires_timezone_aware_index(index):
    with pytest.raises(ValueError, match="timezone-aware"):
        mae_by_hour(pd.Series([1.0, 2.0, 3.0], index=index), [1.0, 2.0, 3.0])


def test_mae_by_hour_with_different_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        mae_by_hour(hourly_series([1.0, 2.0]), [1.0])
