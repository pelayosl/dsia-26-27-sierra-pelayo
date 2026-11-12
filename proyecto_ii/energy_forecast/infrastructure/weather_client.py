"""Weather forecast for the next hours (Open-Meteo forecast API).

The model is trained with the national weather of dsia-dataset: a population-weighted mean of 5 cities.
The forecast is aggregated in exactly the same way, so the model receives comparable values.

Open-Meteo serves forecasts from about 3 months in the past to about 15 days ahead. An origin outside that
range is answered with a 400, which becomes an `ExternalServiceError`.
"""

from datetime import datetime
from typing import Any, Protocol

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from energy_forecast.domain.errors import ExternalServiceError
from energy_forecast.domain.schema import HORIZON_HOURS, TIME_COLUMN, WEATHER_COLUMNS
from energy_forecast.logging_config import get_logger

logger = get_logger("weather_client")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_S = 10
RETRIES = 3  # connection errors, 429 and 5xx, with exponential backoff
HOUR_FORMAT = "%Y-%m-%dT%H:%M"  # format of `start_hour` / `end_hour` and of the `time` of the response

# Same cities and weights as dsia-dataset
CITIES = {
    "Madrid": {"latitude": 40.42, "longitude": -3.70, "weight": 0.30},
    "Barcelona": {"latitude": 41.39, "longitude": 2.17, "weight": 0.25},
    "Valencia": {"latitude": 39.47, "longitude": -0.38, "weight": 0.15},
    "Sevilla": {"latitude": 37.39, "longitude": -5.98, "weight": 0.15},
    "Bilbao": {"latitude": 43.26, "longitude": -2.93, "weight": 0.15},
}


class WeatherForecastClient(Protocol):
    """Weather forecast provider abstraction."""
    def next_hours(self, origin: datetime, hours: int = HORIZON_HOURS) -> pd.DataFrame:
        """National weather forecast for `origin + 1 h ... origin + hours` (`time_utc` + `WEATHER_COLUMNS`)."""


def _target_hours(origin: datetime, hours: int) -> pd.DatetimeIndex:
    timestamp = pd.Timestamp(origin)
    if timestamp.tzinfo is None:
        raise ValueError("origin must be timezone-aware")
    if hours <= 0:
        raise ValueError("hours must be positive")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("h"):
        raise ValueError(f"origin must be on the hour, got {timestamp.isoformat()}")
    return pd.date_range(timestamp + pd.Timedelta(hours=1), periods=hours, freq="h", name=TIME_COLUMN)


def _default_session() -> requests.Session:
    retry = Retry(total=RETRIES, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET"])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class OpenMeteoForecastClient:
    """Open-Meteo client: forecast per city (one request for all of them) and population-weighted mean."""

    def __init__(self, timeout_s: float = REQUEST_TIMEOUT_S, session: requests.Session | None = None) -> None:
        self._timeout_s = timeout_s
        self._session = session if session is not None else _default_session()

    def next_hours(self, origin: datetime, hours: int = HORIZON_HOURS) -> pd.DataFrame:
        """Raises `ExternalServiceError` on HTTP errors, timeouts or incomplete responses."""
        target = _target_hours(origin, hours)
        payload = self._request(target)
        city_frames = self._parse(payload, target)
        weights = pd.Series({city: values["weight"] for city, values in CITIES.items()})
        national = sum(frame * (weights[city] / weights.sum()) for city, frame in city_frames.items())
        logger.info("Weather forecast received for %s - %s", target[0].isoformat(), target[-1].isoformat())
        return national.reset_index()[[TIME_COLUMN, *WEATHER_COLUMNS]]

    def _request(self, target: pd.DatetimeIndex) -> Any:
        params = {
            "latitude": ",".join(str(city["latitude"]) for city in CITIES.values()),
            "longitude": ",".join(str(city["longitude"]) for city in CITIES.values()),
            "hourly": ",".join(WEATHER_COLUMNS),
            "timezone": "UTC",
            "start_hour": target[0].strftime(HOUR_FORMAT),
            "end_hour": target[-1].strftime(HOUR_FORMAT),
        }
        try:
            response = self._session.get(FORECAST_URL, params=params, timeout=self._timeout_s)
        except requests.RequestException as exc:
            raise ExternalServiceError(f"Open-Meteo could not be reached: {exc}") from exc
        if not response.ok:
            raise ExternalServiceError(f"Open-Meteo answered {response.status_code}: {self._reason(response)}")
        try:
            return response.json()
        except ValueError as exc:
            raise ExternalServiceError("Open-Meteo answered with invalid JSON") from exc

    @staticmethod
    def _reason(response: requests.Response) -> str:
        try:
            return str(response.json().get("reason", response.text))
        except ValueError:
            return response.text[:200]

    @staticmethod
    def _parse(payload: Any, target: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        """One DataFrame per city indexed by `time_utc`, with every target hour and no NaN."""
        locations = payload if isinstance(payload, list) else [payload]
        if len(locations) != len(CITIES):
            raise ExternalServiceError(f"Open-Meteo returned {len(locations)} locations, expected {len(CITIES)}")

        frames = {}
        # The API returns locations in the same order as CITIES.values(), so we can zip them together
        for city, location in zip(CITIES, locations):
            hourly = location.get("hourly", {}) if isinstance(location, dict) else {}
            missing_columns = [column for column in ["time", *WEATHER_COLUMNS] if column not in hourly]
            if missing_columns:
                raise ExternalServiceError(f"Open-Meteo response for {city} misses {missing_columns}")
            index = pd.to_datetime(hourly["time"], format=HOUR_FORMAT).tz_localize("UTC").rename(TIME_COLUMN)
            frame = pd.DataFrame(
                {column: pd.to_numeric(pd.Series(hourly[column]), errors="coerce").to_numpy() for column in WEATHER_COLUMNS},
                index=index,
            )
            frame = frame[~frame.index.duplicated()].reindex(target)
            if frame.isna().any().any():
                hours_with_gaps = int(frame.isna().any(axis=1).sum())
                raise ExternalServiceError(f"Open-Meteo forecast for {city} is incomplete ({hours_with_gaps} hours)")
            frames[city] = frame
        return frames
