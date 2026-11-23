"""Forecast evaluation metrics.

Used when training (test set of each candidate model) and when monitoring (stored predictions vs. the
real demand ingested afterwards). Pairs where the real or the predicted value is unknown (NaN) are
ignored: e.g. the hours of the April 2025 blackout have no real demand to compare with.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from energy_forecast.domain.models import ModelMetrics
from energy_forecast.domain.schema import LOCAL_TIMEZONE


def _known_pairs(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Both as float arrays, without the positions where either of them is NaN."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same length, got {len(true)} and {len(pred)}")
    known = ~(np.isnan(true) | np.isnan(pred))
    if not known.any():
        raise ValueError("There are no pairs with known real and predicted values to evaluate")
    return true[known], pred[known]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error (MW)."""
    return float(mean_absolute_error(*_known_pairs(y_true, y_pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error (MW). Penalises large errors more than the MAE."""
    return float(root_mean_squared_error(*_known_pairs(y_true, y_pred)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error (%). Safe here: the real demand is never close to 0."""
    return float(mean_absolute_percentage_error(*_known_pairs(y_true, y_pred)) * 100)


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> ModelMetrics:
    """MAE, RMSE and MAPE of a prediction."""
    return ModelMetrics(mae=mae(y_true, y_pred), rmse=rmse(y_true, y_pred), mape=mape(y_true, y_pred))


def mae_by_hour(y_true: pd.Series, y_pred: ArrayLike) -> dict[int, float]:
    """MAE grouped by local hour of the day (0-23), from the `time_utc` index of `y_true`.

    The local hour is the one that matters for demand (and for whoever reads the forecast). Only the
    hours with some known pair appear; with a full year of data there are 24 keys.
    """
    if not isinstance(y_true.index, pd.DatetimeIndex) or y_true.index.tz is None:
        raise ValueError("y_true must be indexed by a timezone-aware `time_utc`")
    pred = np.asarray(y_pred, dtype=float)
    if len(pred) != len(y_true):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(y_true)} and {len(pred)}")

    errors = pd.Series(np.abs(y_true.to_numpy(dtype=float) - pred), index=y_true.index).dropna()
    local_hour = errors.index.tz_convert(LOCAL_TIMEZONE).hour
    return {int(hour): float(value) for hour, value in errors.groupby(local_hour).mean().items()}
