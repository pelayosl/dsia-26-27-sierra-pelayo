"""Tests of `infrastructure.weather_client` with a fake HTTP session (no network).

The real API was checked by hand (step 3.1); `test_open_meteo_real_request` repeats that check and is
marked `integration`, so CI skips it.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import requests

from energy_forecast.domain.errors import ExternalServiceError
from energy_forecast.domain.schema import HORIZON_HOURS, WEATHER_COLUMNS
from energy_forecast.infrastructure.weather_client import CITIES, FORECAST_URL, OpenMeteoForecastClient
from tests.fakes import FakeWeatherForecastClient

ORIGIN = datetime(2026, 11, 12, 0, 0, tzinfo=timezone.utc)
CITY_TEMPERATURES = {"Madrid": 10.0, "Barcelona": 20.0, "Valencia": 30.0, "Sevilla": 40.0, "Bilbao": 0.0}
# 0.30·10 + 0.25·20 + 0.15·30 + 0.15·40 + 0.15·0 = 18.5
EXPECTED_TEMPERATURE = 18.5


def city_payload(temperature: float, origin: datetime = ORIGIN, hours: int = HORIZON_HOURS) -> dict:
    """One location of the Open-Meteo answer, as the API returns it with `timezone=UTC`."""
    times = pd.date_range(origin + timedelta(hours=1), periods=hours, freq="h")
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for column in WEATHER_COLUMNS:
        hourly[column] = [1.0] * hours
    hourly["temperature_2m"] = [temperature] * hours
    return {"latitude": 0.0, "longitude": 0.0, "hourly": hourly}


def full_payload() -> list[dict]:
    return [city_payload(CITY_TEMPERATURES[city]) for city in CITIES]


class FakeResponse:
    def __init__(self, payload=None, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    """Records the requests and returns `response`, or raises `error`."""

    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.requests: list[tuple[str, dict, float]] = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, params, timeout))
        if self._error is not None:
            raise self._error
        return self._response


def client_with(response: FakeResponse | None = None, error: Exception | None = None):
    session = FakeSession(response, error)
    return OpenMeteoForecastClient(timeout_s=5, session=session), session


# ---------------------------------------------------------------- happy path


def test_next_hours_returns_24_hours_after_origin():
    client, _ = client_with(FakeResponse(full_payload()))

    forecast = client.next_hours(ORIGIN)

    assert list(forecast.columns) == ["time_utc", *WEATHER_COLUMNS]
    assert len(forecast) == HORIZON_HOURS
    assert forecast["time_utc"].iloc[0] == pd.Timestamp(ORIGIN) + pd.Timedelta(hours=1)
    assert forecast["time_utc"].iloc[-1] == pd.Timestamp(ORIGIN) + pd.Timedelta(hours=HORIZON_HOURS)
    assert str(forecast["time_utc"].dt.tz) == "UTC"


def test_next_hours_is_the_population_weighted_mean_of_the_cities():
    client, _ = client_with(FakeResponse(full_payload()))

    forecast = client.next_hours(ORIGIN)

    assert forecast["temperature_2m"].tolist() == pytest.approx([EXPECTED_TEMPERATURE] * HORIZON_HOURS)
    assert forecast["wind_speed_10m"].tolist() == pytest.approx([1.0] * HORIZON_HOURS)


def test_request_asks_for_every_city_and_the_exact_hours_in_utc():
    client, session = client_with(FakeResponse(full_payload()))

    client.next_hours(ORIGIN)

    url, params, timeout = session.requests[0]
    assert url == FORECAST_URL
    assert timeout == 5
    assert len(params["latitude"].split(",")) == len(CITIES)
    assert params["hourly"].split(",") == WEATHER_COLUMNS
    assert params["timezone"] == "UTC"
    assert (params["start_hour"], params["end_hour"]) == ("2026-11-12T01:00", "2026-11-13T00:00")


def test_origin_in_another_timezone_is_converted_to_utc():
    madrid = timezone(timedelta(hours=1))
    client, session = client_with(FakeResponse(full_payload()))

    client.next_hours(datetime(2026, 11, 12, 1, 0, tzinfo=madrid))  # = 00:00 UTC

    assert session.requests[0][1]["start_hour"] == "2026-11-12T01:00"


def test_extra_hours_in_the_answer_are_ignored():
    payload = [city_payload(CITY_TEMPERATURES[city], origin=ORIGIN - timedelta(hours=5), hours=40) for city in CITIES]
    client, _ = client_with(FakeResponse(payload))

    forecast = client.next_hours(ORIGIN)

    assert len(forecast) == HORIZON_HOURS
    assert forecast["time_utc"].iloc[0] == pd.Timestamp(ORIGIN) + pd.Timedelta(hours=1)


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "error",
    [requests.Timeout("read timed out"), requests.ConnectionError("no route to host")],
    ids=["timeout", "connection"],
)
def test_network_errors_raise_external_service_error(error):
    client, _ = client_with(error=error)

    with pytest.raises(ExternalServiceError, match="could not be reached"):
        client.next_hours(ORIGIN)


def test_http_error_includes_open_meteo_reason():
    reason = {"error": True, "reason": "Parameter 'start_hour' is out of allowed range"}
    client, _ = client_with(FakeResponse(reason, status_code=400))

    with pytest.raises(ExternalServiceError, match="400: Parameter 'start_hour' is out of allowed range"):
        client.next_hours(ORIGIN)


def test_server_error_without_json_raises():
    client, _ = client_with(FakeResponse(None, status_code=503, text="Service Unavailable"))

    with pytest.raises(ExternalServiceError, match="503: Service Unavailable"):
        client.next_hours(ORIGIN)


def test_invalid_json_raises():
    client, _ = client_with(FakeResponse(None, status_code=200))

    with pytest.raises(ExternalServiceError, match="invalid JSON"):
        client.next_hours(ORIGIN)


def test_missing_city_raises():
    client, _ = client_with(FakeResponse(full_payload()[:-1]))

    with pytest.raises(ExternalServiceError, match="4 locations, expected 5"):
        client.next_hours(ORIGIN)


def test_missing_variable_raises():
    payload = full_payload()
    del payload[2]["hourly"]["precipitation"]
    client, _ = client_with(FakeResponse(payload))

    with pytest.raises(ExternalServiceError, match=r"Valencia misses \['precipitation'\]"):
        client.next_hours(ORIGIN)


def test_missing_hours_raise():
    payload = full_payload()
    payload[0] = city_payload(10.0, hours=20)
    client, _ = client_with(FakeResponse(payload))

    with pytest.raises(ExternalServiceError, match="Madrid is incomplete"):
        client.next_hours(ORIGIN)


def test_null_values_raise():
    payload = full_payload()
    payload[1]["hourly"]["shortwave_radiation"][3] = None
    client, _ = client_with(FakeResponse(payload))

    with pytest.raises(ExternalServiceError, match="Barcelona is incomplete"):
        client.next_hours(ORIGIN)


@pytest.mark.parametrize(
    "origin",
    [datetime(2026, 11, 12, 0, 0), ORIGIN + timedelta(minutes=30)],
    ids=["naive", "not-on-the-hour"],
)
def test_invalid_origin_raises_before_any_request(origin):
    client, session = client_with(FakeResponse(full_payload()))

    with pytest.raises(ValueError, match="origin"):
        client.next_hours(origin)

    assert session.requests == []


# ---------------------------------------------------------------- fake


def test_fake_moves_the_fixed_values_to_the_requested_hours(weather_forecast_24h):
    fake = FakeWeatherForecastClient(weather_forecast_24h)

    forecast = fake.next_hours(ORIGIN)

    assert list(forecast.columns) == ["time_utc", *WEATHER_COLUMNS]
    assert forecast["time_utc"].iloc[0] == pd.Timestamp(ORIGIN) + pd.Timedelta(hours=1)
    assert forecast["temperature_2m"].tolist() == weather_forecast_24h["temperature_2m"].tolist()
    assert fake.origins == [ORIGIN]


def test_fake_can_simulate_a_failure():
    fake = FakeWeatherForecastClient(error=ExternalServiceError("down"))

    with pytest.raises(ExternalServiceError):
        fake.next_hours(ORIGIN)


# ---------------------------------------------------------------- real API


@pytest.mark.integration
def test_open_meteo_real_request():
    origin = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    forecast = OpenMeteoForecastClient().next_hours(origin)

    assert len(forecast) == HORIZON_HOURS
    assert forecast[WEATHER_COLUMNS].notna().all().all()
    assert forecast["temperature_2m"].between(-20, 50).all()
