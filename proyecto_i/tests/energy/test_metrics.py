"""Tests de las métricas de `energy_forecast_app.metrics`."""

import pandas as pd
import pytest

from energy_forecast_app.metrics import EnergyMetrics
from energy_forecast_app.validator import EnergyRecord


def registro(time_local: str, load_mw: float, temperature_2m: float, is_holiday: bool = False) -> EnergyRecord:
    """Crea un `EnergyRecord` con valores fijos salvo hora, demanda, temperatura y festivo."""
    hora_local = pd.Timestamp(time_local)
    return EnergyRecord(
        time_utc=hora_local.tz_convert("UTC"),
        time_local=hora_local,
        load_mw=load_mw,
        temperature_2m=temperature_2m,
        relative_humidity_2m=70.0,
        wind_speed_10m=10.0,
        shortwave_radiation=0.0,
        precipitation=0.0,
        is_holiday=is_holiday,
        covid_flag=False,
        load_interpolated=False,
        weather_interpolated=False,
    )


@pytest.fixture
def registros():
    """Tres horas repartidas en dos meses, dos horas locales y dos tramos de temperatura; una es festivo."""
    return [
        registro("2025-01-15T08:00:00+01:00", 30000.0, 3.0),
        registro("2025-01-15T09:00:00+01:00", 20000.0, 7.0, is_holiday=True),
        registro("2025-02-10T08:00:00+01:00", 40000.0, 4.0),
    ]


def test_energia_total_por_mes_en_gwh(registros):
    """La energía se suma por mes y se pasa a GWh."""
    totales = EnergyMetrics().total_energy_by_month(registros)

    assert totales == {"2025-01": 50.0, "2025-02": 40.0}


def test_demanda_media_por_hora_local(registros):
    """La demanda se promedia por hora local."""
    medias = EnergyMetrics().mean_load_by_hour(registros)

    assert medias == {8: 35000.0, 9: 20000.0}


def test_demanda_media_por_tramo_de_temperatura(registros):
    """La demanda se promedia por tramos de 5 °C, en orden de temperatura."""
    medias = EnergyMetrics().mean_load_by_temperature_band(registros)

    assert list(medias.items()) == [("[0, 5) °C", 35000.0), ("[5, 10) °C", 20000.0)]


@pytest.mark.parametrize(
    "temperatura, tramo",
    [(-0.5, "[-5, 0) °C"), (0.0, "[0, 5) °C"), (4.99, "[0, 5) °C"), (5.0, "[5, 10) °C")],
)
def test_bordes_de_tramo_de_temperatura(temperatura, tramo):
    """Cada temperatura cae en el tramo [inicio, inicio + 5), también con valores negativos."""
    medias = EnergyMetrics().mean_load_by_temperature_band(
        [registro("2025-01-15T08:00:00+01:00", 30000.0, temperatura)]
    )

    assert list(medias) == [tramo]


def test_demanda_media_por_tipo_de_dia(registros):
    """La demanda se promedia por separado en festivos y no festivos."""
    medias = EnergyMetrics().mean_load_by_day_type(registros)

    assert medias == {"no festivo": 35000.0, "festivo": 20000.0}


def test_metricas_sin_registros_devuelven_diccionarios_vacios():
    """Sin registros, todas las métricas devuelven un diccionario vacío."""
    metricas = EnergyMetrics()

    assert metricas.total_energy_by_month([]) == {}
    assert metricas.mean_load_by_hour([]) == {}
    assert metricas.mean_load_by_temperature_band([]) == {}
    assert metricas.mean_load_by_day_type([]) == {}
