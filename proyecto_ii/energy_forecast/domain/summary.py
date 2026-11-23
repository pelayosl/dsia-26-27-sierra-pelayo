"""Numeric summary of the forecast, which the AI edge turns into text.

The domain computes every figure; the LLM only writes them up, so it cannot invent numbers.
Figures are rounded to what a reader needs (whole MW, one decimal for % and °C): they are the ones the
explanation will quote.
"""

import pandas as pd

from energy_forecast.domain.features import add_calendar_features
from energy_forecast.domain.models import Forecast, ForecastSummary
from energy_forecast.domain.schema import LOCAL_TIMEZONE, TARGET_COLUMN, TIME_COLUMN

WEEK = pd.Timedelta(hours=168)


def _predictions_frame(forecast: Forecast) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIME_COLUMN: pd.to_datetime([p.time_utc for p in forecast.predictions], utc=True),
            "predicted_mw": [p.predicted_mw for p in forecast.predictions],
        }
    )


def _change_vs_last_week_pct(predictions: pd.DataFrame, history: pd.DataFrame) -> float | None:
    """% change of the mean forecast vs. the real demand of the same hours one week earlier.

    Only hours whose demand a week earlier is known are compared (pairwise), so a blackout last week
    does not bias the mean. `None` if none is known.
    """
    last_week = history.assign(**{TIME_COLUMN: pd.to_datetime(history[TIME_COLUMN], utc=True) + WEEK})
    pairs = predictions.merge(last_week[[TIME_COLUMN, TARGET_COLUMN]], on=TIME_COLUMN, how="inner").dropna()
    if pairs.empty or pairs[TARGET_COLUMN].mean() == 0:
        return None
    return round((pairs["predicted_mw"].mean() / pairs[TARGET_COLUMN].mean() - 1) * 100, 1)


def _mean_temperature(predictions: pd.DataFrame, weather_forecast: pd.DataFrame) -> float:
    weather = weather_forecast.assign(**{TIME_COLUMN: pd.to_datetime(weather_forecast[TIME_COLUMN], utc=True)})
    in_period = weather[weather[TIME_COLUMN].isin(predictions[TIME_COLUMN])]["temperature_2m"].dropna()
    if in_period.empty:
        raise ValueError("The weather forecast has no temperature for the forecast hours")
    return round(float(in_period.mean()), 1)


def summarize_forecast(
    forecast: Forecast, history: pd.DataFrame, weather_forecast: pd.DataFrame, model_mae_mw: float | None
) -> ForecastSummary:
    """Peak, valley, mean, change vs. the previous week and context of the forecast period.

    - `history`: observations before the origin (as `ObservationRepository.last_hours` returns them);
      only the demand of the same hours a week earlier is used.
    - `weather_forecast`: the forecast used for the prediction (`time_utc` + `temperature_2m`).
    - `model_mae_mw`: test MAE of the model, from the saved metrics, as an indication of uncertainty
      (`None` if unknown).
    """
    predictions = _predictions_frame(forecast)
    local_time = predictions[TIME_COLUMN].dt.tz_convert(LOCAL_TIMEZONE)
    peak = predictions["predicted_mw"].idxmax()
    valley = predictions["predicted_mw"].idxmin()
    # Same calendar as the features (D7): national holidays and weekends in local time
    calendar = add_calendar_features(predictions[[TIME_COLUMN]])

    return ForecastSummary(
        period_start_local=local_time.iloc[0].to_pydatetime(),
        period_end_local=local_time.iloc[-1].to_pydatetime(),
        peak_time_local=local_time[peak].to_pydatetime(),
        peak_mw=round(float(predictions["predicted_mw"][peak])),
        valley_time_local=local_time[valley].to_pydatetime(),
        valley_mw=round(float(predictions["predicted_mw"][valley])),
        mean_mw=round(float(predictions["predicted_mw"].mean())),
        change_vs_last_week_pct=_change_vs_last_week_pct(predictions, history),
        mean_temperature_c=_mean_temperature(predictions, weather_forecast),
        is_holiday=bool(calendar["is_holiday"].any()),
        is_weekend=bool(calendar["is_weekend"].any()),
        model_mae_mw=None if model_mae_mw is None else round(float(model_mae_mw)),
    )
