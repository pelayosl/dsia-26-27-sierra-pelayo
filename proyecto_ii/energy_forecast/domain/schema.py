"""Input contract (columns of the dsia-dataset CSV) and problem constants."""

LOCAL_TIMEZONE = "Europe/Madrid"

TARGET_COLUMN = "load_mw"
TIME_COLUMN = "time_utc"

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "precipitation",
]
FLAG_COLUMNS = ["is_holiday", "covid_flag", "load_interpolated", "weather_interpolated"]
REQUIRED_COLUMNS = [TIME_COLUMN, "time_local", TARGET_COLUMN, *WEATHER_COLUMNS, *FLAG_COLUMNS]

# Expected ranges (same thresholds as the dsia-dataset quality report)
LOAD_MW_RANGE = (15_000.0, 50_000.0)
TEMPERATURE_RANGE_C = (-20.0, 50.0)
HUMIDITY_RANGE_PCT = (0.0, 100.0)
NON_NEGATIVE_COLUMNS = ["wind_speed_10m", "shortwave_radiation", "precipitation"]

# Horizon and lags (direct strategy: all lags must be >= HORIZON_HOURS)
HORIZON_HOURS = 24
LAG_HOURS = (24, 48, 168)
ROLLING_WINDOWS_HOURS = (24, 168)
MIN_HISTORY_HOURS = max(LAG_HOURS) + max(ROLLING_WINDOWS_HOURS)
