"""Feature construction. The same code is used for training and for prediction (D7).

Every feature of a target hour t must be known at the origin T = t - h (h = 1..24):
- lags of at least `HORIZON_HOURS` hours,
- rolling means over windows that end at t - `HORIZON_HOURS`,
- calendar and weather of t itself (the weather comes from a forecast at prediction time).

Input frames have the format of `validate_observations` (and of `ObservationRepository.range`):
`time_utc` UTC-aware and unique. Gaps in the hourly series are allowed (rejected rows): lags and
windows are computed on a continuous hourly grid, so a gap never shifts the values of other rows.
"""

from datetime import datetime

import holidays
import pandas as pd

from energy_forecast.domain.errors import InsufficientHistoryError, ValidationError
from energy_forecast.domain.schema import (
    HORIZON_HOURS,
    LAG_HOURS,
    LOCAL_TIMEZONE,
    MIN_HISTORY_HOURS,
    ROLLING_WINDOWS_HOURS,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEATHER_COLUMNS,
)

LAG_FEATURES = [f"lag_{hours}" for hours in LAG_HOURS]
ROLLING_FEATURES = [f"rolling_mean_{hours}" for hours in ROLLING_WINDOWS_HOURS]
CALENDAR_FEATURES = ["hour", "weekday", "month", "is_holiday", "is_weekend"]
CATEGORICAL_FEATURES = ["hour", "weekday", "month"]
FEATURE_COLUMNS = [*LAG_FEATURES, *ROLLING_FEATURES, *CALENDAR_FEATURES, *WEATHER_COLUMNS, "covid_flag"]

HOLIDAYS_COUNTRY = "ES"  # national holidays, same as dsia-dataset
# A rolling mean needs at least half of its window with known demand (NaN = blackout, out of range)
ROLLING_MIN_FRACTION = 0.5

if min(LAG_HOURS) < HORIZON_HOURS:
    # With a lag shorter than the horizon, the last target hours would need demand after the origin
    raise ValueError(f"All LAG_HOURS must be >= HORIZON_HOURS ({HORIZON_HOURS}), got {LAG_HOURS}")


def _hourly_load(frame: pd.DataFrame) -> pd.Series:
    """`load_mw` on a continuous hourly grid indexed by `time_utc` (missing hours → NaN)."""
    load = frame.set_index(TIME_COLUMN)[TARGET_COLUMN].astype(float).sort_index()
    return load.asfreq("h")


def _national_holidays(local_dates: pd.Series) -> pd.Series:
    """1 if the local date is a national holiday in Spain, 0 otherwise."""
    years = range(local_dates.min().year, local_dates.max().year + 1) if not local_dates.empty else []
    calendar = holidays.country_holidays(HOLIDAYS_COUNTRY, years=years)
    return local_dates.map(lambda day: int(day in calendar))


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds `hour`, `weekday`, `month`, `is_weekend` and the missing `is_holiday` values, in local time.

    Demand follows the Madrid clock, so everything is derived from `time_utc` converted to `LOCAL_TIMEZONE`.
    The `is_holiday` of the dataset is kept; it is only computed where it is missing (future hours).
    """
    result = frame.copy()
    local = pd.to_datetime(result[TIME_COLUMN], utc=True).dt.tz_convert(LOCAL_TIMEZONE)
    result["hour"] = local.dt.hour
    result["weekday"] = local.dt.day_of_week  # Monday = 0
    result["month"] = local.dt.month
    result["is_weekend"] = (local.dt.day_of_week >= 5).astype(int)

    computed = _national_holidays(local.dt.date)
    if "is_holiday" in result.columns:
        result["is_holiday"] = result["is_holiday"].fillna(computed)
    else:
        result["is_holiday"] = computed
    result["is_holiday"] = result["is_holiday"].astype(int)
    return result


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds `lag_k` = demand at t - k hours, for each k of `LAG_HOURS` (NaN if that hour is unknown)."""
    result = frame.copy()
    load = _hourly_load(frame)
    for hours in LAG_HOURS:
        lagged = load.shift(hours)
        result[f"lag_{hours}"] = lagged.reindex(result[TIME_COLUMN]).to_numpy()
    return result


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds `rolling_mean_w` = mean demand in the `w`-hour window ending at t - `HORIZON_HOURS`.

    For any horizon from 1 to 24 the window ends at the origin at the latest, the same as the lags.
    Unknown hours inside the window are ignored if at least half of the window is known.
    """
    result = frame.copy()
    shifted = _hourly_load(frame).shift(HORIZON_HOURS)
    for hours in ROLLING_WINDOWS_HOURS:
        min_periods = max(1, int(hours * ROLLING_MIN_FRACTION))
        rolling = shifted.rolling(hours, min_periods=min_periods).mean()
        result[f"rolling_mean_{hours}"] = rolling.reindex(result[TIME_COLUMN]).to_numpy()
    return result


def _add_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    return add_rolling_features(add_lag_features(add_calendar_features(frame)))


def build_training_frame(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Builds `(X, y)` for training from the historical observations.

    Excluded rows (they are still used to compute the lags of later rows):
    - the first `MIN_HISTORY_HOURS` hours (warm-up: incomplete lags and windows),
    - rows whose target `load_mw` is NaN (blackout, out-of-range values),
    - rows with some unknown feature (e.g. a lag that falls on a NaN demand).
    """
    if observations.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(name=TARGET_COLUMN, dtype=float)

    frame = observations.sort_values(TIME_COLUMN).reset_index(drop=True)
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True)
    frame = _add_all_features(frame).set_index(TIME_COLUMN)

    warm_up_end = frame.index.min() + pd.Timedelta(hours=MIN_HISTORY_HOURS)
    usable = (
        (frame.index >= warm_up_end)
        & frame[TARGET_COLUMN].notna()
        & frame[FEATURE_COLUMNS].notna().all(axis=1)
    )
    frame = frame[usable]
    return frame[FEATURE_COLUMNS], frame[TARGET_COLUMN].astype(float)


def _utc_hour(origin: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(origin)
    if timestamp.tzinfo is None:
        raise ValueError("origin must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("h"):
        raise ValueError(f"origin must be on the hour, got {timestamp.isoformat()}")
    return timestamp


def build_inference_frame(
    history: pd.DataFrame, weather_forecast: pd.DataFrame, origin: datetime
) -> pd.DataFrame:
    """Builds the 24 feature rows for the hours `origin + 1 h ... origin + 24 h`.

    `history`: observations up to `origin` (later rows are ignored, so the future can never leak in).
    `weather_forecast`: `time_utc` + `WEATHER_COLUMNS` covering the 24 target hours.
    Raises `InsufficientHistoryError` if `history` does not cover the `MIN_HISTORY_HOURS` up to `origin`
    (with demand known at `origin`), and `ValidationError` if the weather forecast misses target hours.
    """
    origin = _utc_hour(origin)
    history = history.copy()
    history[TIME_COLUMN] = pd.to_datetime(history[TIME_COLUMN], utc=True)
    history = history[history[TIME_COLUMN] <= origin]

    earliest_needed = origin - pd.Timedelta(hours=MIN_HISTORY_HOURS - 1)
    known_at_origin = history.loc[history[TIME_COLUMN] == origin, TARGET_COLUMN].notna().any()
    if history.empty or history[TIME_COLUMN].min() > earliest_needed or not known_at_origin:
        raise InsufficientHistoryError(
            f"Need {MIN_HISTORY_HOURS} h of history from {earliest_needed.isoformat()} "
            f"up to the origin {origin.isoformat()} (with known demand at the origin)"
        )

    target_times = pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON_HOURS, freq="h")
    weather = weather_forecast.copy()
    weather[TIME_COLUMN] = pd.to_datetime(weather[TIME_COLUMN], utc=True)
    weather = weather.drop_duplicates(TIME_COLUMN).set_index(TIME_COLUMN)[WEATHER_COLUMNS]
    missing = target_times.difference(weather.index)
    if len(missing) > 0:
        raise ValidationError(f"Weather forecast misses {len(missing)} target hours, first {missing[0].isoformat()}")

    target = weather.reindex(target_times).rename_axis(TIME_COLUMN).reset_index()
    target[TARGET_COLUMN] = float("nan")  # unknown: that is what we predict
    target["covid_flag"] = 0  # no COVID restrictions in future hours
    # `is_holiday` is left out: `add_calendar_features` computes it for the target hours

    combined = pd.concat([history, target], ignore_index=True)
    features = _add_all_features(combined).set_index(TIME_COLUMN)
    return features.loc[target_times, FEATURE_COLUMNS].rename_axis(TIME_COLUMN)
