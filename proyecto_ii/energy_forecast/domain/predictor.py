"""Inference: turns the features of the 24 target hours into a `Forecast`."""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from energy_forecast.domain.features import FEATURE_COLUMNS
from energy_forecast.domain.models import Forecast, HourlyPrediction
from energy_forecast.domain.schema import HORIZON_HOURS


def predict_forecast(
    model: BaseEstimator, features: pd.DataFrame, origin: datetime, model_version: str
) -> Forecast:
    """Runs `model.predict` on the 24 rows of `build_inference_frame` and builds the `Forecast` (no explanation).

    `features` must be indexed by `time_utc` (origin + 1 h ... origin + 24 h) and contain `FEATURE_COLUMNS`;
    they are passed to the model in that order, the same as in training. `Forecast` checks that the hours are
    consecutive after `origin`; `explanation` is added later by the service.
    Raises `ValueError` if the features do not have that shape or the model returns non-finite values.
    """
    if pd.Timestamp(origin).tzinfo is None:
        raise ValueError("origin must be timezone-aware")
    if len(features) != HORIZON_HOURS:
        raise ValueError(f"Expected {HORIZON_HOURS} feature rows, got {len(features)}")
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Features miss the columns {missing}")

    predicted = np.asarray(model.predict(features[FEATURE_COLUMNS]), dtype=float)
    if predicted.shape != (HORIZON_HOURS,) or not np.isfinite(predicted).all():
        raise ValueError(f"The model returned invalid predictions: {predicted.tolist()}")

    target_hours = pd.DatetimeIndex(features.index).tz_convert("UTC")
    predictions = tuple(
        HourlyPrediction(time_utc=hour.to_pydatetime(), predicted_mw=float(value))
        for hour, value in zip(target_hours, predicted)
    )
    return Forecast(
        origin=pd.Timestamp(origin).tz_convert("UTC").to_pydatetime(),
        predictions=predictions,
        model_version=model_version,
    )
