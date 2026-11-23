"""Input contract validation for hourly observations.

In project i, this code was implemented using OOP. However, given the fact that
project ii involves sending vectorised data to repositories and sklearn,
creating objects and then turning them back into DataFrames produces unnecessary
overhead.

OOP is used later with Pydantic models to serialise data to JSON and send it
through services, CLI and API."""

import pandas as pd

from energy_forecast.domain.errors import SchemaError
from energy_forecast.domain.schema import (
    FLAG_COLUMNS,
    HUMIDITY_RANGE_PCT,
    LOAD_MW_RANGE,
    LOCAL_TIMEZONE,
    NON_NEGATIVE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEMPERATURE_RANGE_C,
    TIME_COLUMN,
    WEATHER_COLUMNS,
)

REJECTION_REASON_COLUMN = "rejection_reason"


def validate_observations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split usable rows from rejected ones.

    Rules:

    - every column of `REQUIRED_COLUMNS` must exist (extra columns are ignored), otherwise `SchemaError`
    - `time_utc` must be a valid ISO 8601 date and must not be duplicated (all copies of a repeated hour
      are rejected); `time_local` is recomputed from `time_utc`, which is the source of truth
    - flags (`FLAG_COLUMNS`) must be 0 or 1, otherwise the row is rejected
    - `load_mw` null or outside `LOAD_MW_RANGE` --> the row is kept with `load_mw` = NaN, so the lags of the
      following hours stay aligned (training excludes it as a target)
    - weather outside `TEMPERATURE_RANGE_C`, `HUMIDITY_RANGE_PCT` or negative in `NON_NEGATIVE_COLUMNS` --> NaN

    Returns `(observations, rejected)`:

    - `observations`: `REQUIRED_COLUMNS` with parsed types, sorted by `time_utc`, new index
    - `rejected`: the original rows (unparsed values) plus a `rejection_reason` column
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    temp = frame[REQUIRED_COLUMNS].copy()

    # Parse types; values that cannot be parsed become NaT / NaN
    temp[TIME_COLUMN] = pd.to_datetime(temp[TIME_COLUMN], utc=True, format="ISO8601", errors="coerce")
    temp["time_local"] = temp[TIME_COLUMN].dt.tz_convert(LOCAL_TIMEZONE)
    for column in [TARGET_COLUMN, *WEATHER_COLUMNS, *FLAG_COLUMNS]:
        temp[column] = pd.to_numeric(temp[column], errors="coerce")

    # Rows that break the hourly grid or have invalid flags are rejected
    invalid_time = temp[TIME_COLUMN].isna()
    duplicated_time = temp[TIME_COLUMN].duplicated(keep=False) & ~invalid_time
    invalid_flags = ~temp[FLAG_COLUMNS].isin([0, 1]).all(axis=1)

    reasons = pd.Series(pd.NA, index=temp.index, dtype="string")
    reasons = reasons.mask(invalid_flags, "invalid_flag")
    reasons = reasons.mask(duplicated_time, "duplicated_time_utc")
    reasons = reasons.mask(invalid_time, "invalid_time_utc")
    rejected_mask = reasons.notna()

    # Out-of-range values in kept rows become NaN instead of rejecting the hour
    temp[TARGET_COLUMN] = temp[TARGET_COLUMN].where(temp[TARGET_COLUMN].between(*LOAD_MW_RANGE))
    temp["temperature_2m"] = temp["temperature_2m"].where(temp["temperature_2m"].between(*TEMPERATURE_RANGE_C))
    temp["relative_humidity_2m"] = temp["relative_humidity_2m"].where(
        temp["relative_humidity_2m"].between(*HUMIDITY_RANGE_PCT)
    )
    for column in NON_NEGATIVE_COLUMNS:
        temp[column] = temp[column].where(temp[column] >= 0)

    observations = temp.loc[~rejected_mask].copy()
    observations[FLAG_COLUMNS] = observations[FLAG_COLUMNS].astype(int)
    observations = observations.sort_values(TIME_COLUMN).reset_index(drop=True)

    rejected = frame.loc[rejected_mask].copy()
    rejected[REJECTION_REASON_COLUMN] = reasons[rejected_mask]

    return observations, rejected
