import pytest


@pytest.fixture
def fila_valida() -> dict:
    """Una hora que cumple todas las reglas del validador (15-ene-2025, 08:00 en Madrid)."""
    return {
        "time_utc": "2025-01-15T07:00:00+00:00",
        "time_local": "2025-01-15T08:00:00+01:00",
        "load_mw": 30000.0,
        "temperature_2m": 3.0,
        "relative_humidity_2m": 70.0,
        "wind_speed_10m": 10.0,
        "shortwave_radiation": 5.0,
        "precipitation": 0.0,
        "is_holiday": 0,
        "covid_flag": 0,
        "load_interpolated": 0,
        "weather_interpolated": 0,
    }
