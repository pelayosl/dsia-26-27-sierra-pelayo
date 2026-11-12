"""Tests for `domain.validator`."""

from pathlib import Path

import pandas as pd
import pytest

from energy_forecast.domain.errors import SchemaError
from energy_forecast.domain.schema import FLAG_COLUMNS, REQUIRED_COLUMNS
from energy_forecast.domain.validator import REJECTION_REASON_COLUMN, validate_observations
from tests.conftest import (
    DUPLICATED_ROWS,
    INVALID_FLAG_ROW,
    NULL_LOAD_ROW,
    OUT_OF_RANGE_LOAD_ROW,
)

SAMPLE_PATH = Path(__file__).parents[3] / "proyecto_i" / "data" / "energy_weather_es_hourly_sample.csv"
TARGET_ROW = 5


def _value_at(observations: pd.DataFrame, raw: pd.DataFrame, row: int, column: str):
    """Value of `column` in the validated row that comes from `raw` row `row` (matched by `time_utc`)."""
    time = pd.Timestamp(raw.loc[row, "time_utc"])
    return observations.loc[observations["time_utc"] == time, column].iloc[0]


def test_valid_data_is_kept_with_parsed_types(hourly_observations):
    # Arrange: the fixture is fully valid

    # Act
    observations, rejected = validate_observations(hourly_observations)

    # Assert
    assert rejected.empty
    assert len(observations) == len(hourly_observations)
    assert list(observations.columns) == REQUIRED_COLUMNS
    assert str(observations["time_utc"].dt.tz) == "UTC"
    assert str(observations["time_local"].dt.tz) == "Europe/Madrid"
    assert observations["load_mw"].notna().all()
    assert all(pd.api.types.is_integer_dtype(observations[column]) for column in FLAG_COLUMNS)


@pytest.mark.parametrize("column", ["time_utc", "load_mw", "temperature_2m", "covid_flag"])
def test_missing_column_raises_schema_error(hourly_observations, column):
    with pytest.raises(SchemaError, match=column):
        validate_observations(hourly_observations.drop(columns=[column]))


def test_extra_columns_are_ignored(hourly_observations):
    hourly_observations["energy_mwh"] = 1.0

    observations, rejected = validate_observations(hourly_observations)

    assert rejected.empty
    assert list(observations.columns) == REQUIRED_COLUMNS


def test_duplicated_time_rows_are_rejected(invalid_observations):
    observations, rejected = validate_observations(invalid_observations)

    duplicated = rejected.loc[list(DUPLICATED_ROWS)]
    assert (duplicated[REJECTION_REASON_COLUMN] == "duplicated_time_utc").all()
    assert not observations["time_utc"].duplicated().any()


@pytest.mark.parametrize("time_utc", ["not a date", None, ""])
def test_invalid_time_is_rejected(hourly_observations, time_utc):
    hourly_observations.loc[TARGET_ROW, "time_utc"] = time_utc

    observations, rejected = validate_observations(hourly_observations)

    assert rejected.loc[TARGET_ROW, REJECTION_REASON_COLUMN] == "invalid_time_utc"
    assert len(observations) == len(hourly_observations) - 1


@pytest.mark.parametrize("load_mw", [None, 0.0, -1.0, 14_999.0, 50_001.0, "n/a"])
def test_invalid_load_is_kept_as_nan(hourly_observations, load_mw):
    hourly_observations["load_mw"] = hourly_observations["load_mw"].astype(object)
    hourly_observations.loc[TARGET_ROW, "load_mw"] = load_mw

    observations, rejected = validate_observations(hourly_observations)

    assert rejected.empty
    assert len(observations) == len(hourly_observations)
    assert pd.isna(_value_at(observations, hourly_observations, TARGET_ROW, "load_mw"))


@pytest.mark.parametrize("load_mw", [15_000.0, 50_000.0])
def test_load_at_range_limits_is_kept(hourly_observations, load_mw):
    hourly_observations.loc[TARGET_ROW, "load_mw"] = load_mw

    observations, _ = validate_observations(hourly_observations)

    assert _value_at(observations, hourly_observations, TARGET_ROW, "load_mw") == load_mw


@pytest.mark.parametrize(
    "column, value",
    [
        ("temperature_2m", -20.1),
        ("temperature_2m", 50.1),
        ("relative_humidity_2m", -1.0),
        ("relative_humidity_2m", 100.1),
        ("wind_speed_10m", -0.1),
        ("shortwave_radiation", -1.0),
        ("precipitation", -0.5),
    ],
)
def test_out_of_range_weather_becomes_nan(hourly_observations, column, value):
    hourly_observations.loc[TARGET_ROW, column] = value

    observations, rejected = validate_observations(hourly_observations)

    assert rejected.empty
    assert pd.isna(_value_at(observations, hourly_observations, TARGET_ROW, column))
    assert _value_at(observations, hourly_observations, TARGET_ROW, "load_mw") > 0


@pytest.mark.parametrize("flag_value", [2, -1, None])
def test_invalid_flag_rows_are_rejected(hourly_observations, flag_value):
    hourly_observations["covid_flag"] = hourly_observations["covid_flag"].astype(object)
    hourly_observations.loc[TARGET_ROW, "covid_flag"] = flag_value

    observations, rejected = validate_observations(hourly_observations)

    assert rejected.loc[TARGET_ROW, REJECTION_REASON_COLUMN] == "invalid_flag"
    assert len(observations) == len(hourly_observations) - 1


def test_invalid_observations_fixture(invalid_observations):
    observations, rejected = validate_observations(invalid_observations)

    assert sorted(rejected.index) == sorted([*DUPLICATED_ROWS, INVALID_FLAG_ROW])
    assert rejected.loc[INVALID_FLAG_ROW, REJECTION_REASON_COLUMN] == "invalid_flag"
    assert len(observations) == len(invalid_observations) - 3
    assert pd.isna(_value_at(observations, invalid_observations, NULL_LOAD_ROW, "load_mw"))
    assert pd.isna(_value_at(observations, invalid_observations, OUT_OF_RANGE_LOAD_ROW, "load_mw"))


def test_rejected_rows_keep_original_values(invalid_observations):
    _, rejected = validate_observations(invalid_observations)

    assert rejected.loc[INVALID_FLAG_ROW, "is_holiday"] == 2
    assert rejected.loc[INVALID_FLAG_ROW, "time_utc"] == invalid_observations.loc[INVALID_FLAG_ROW, "time_utc"]


def test_output_is_sorted_by_time(hourly_observations):
    shuffled = hourly_observations.sample(frac=1, random_state=0)

    observations, _ = validate_observations(shuffled)

    assert observations["time_utc"].is_monotonic_increasing
    assert list(observations.index) == list(range(len(observations)))


def test_time_local_is_recomputed_from_time_utc(hourly_observations):
    hourly_observations.loc[TARGET_ROW, "time_local"] = "wrong value"

    observations, rejected = validate_observations(hourly_observations)

    assert rejected.empty
    local = _value_at(observations, hourly_observations, TARGET_ROW, "time_local")
    utc = _value_at(observations, hourly_observations, TARGET_ROW, "time_utc")
    assert local == utc.tz_convert("Europe/Madrid")


def test_input_frame_is_not_mutated(invalid_observations):
    original = invalid_observations.copy()

    validate_observations(invalid_observations)

    pd.testing.assert_frame_equal(invalid_observations, original)


@pytest.mark.integration
def test_real_sample_matches_documented_quality():
    # The sample contains the blackout of 2025-04-28/29 (35 h without load) and 2 out-of-range values
    raw = pd.read_csv(SAMPLE_PATH)

    observations, rejected = validate_observations(raw)

    assert rejected.empty
    assert len(observations) == len(raw)
    assert observations["load_mw"].isna().sum() == 37
