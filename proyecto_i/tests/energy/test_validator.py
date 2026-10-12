from pathlib import Path

import pandas as pd
import pytest

from energy_forecast_app.errors import SchemaError
from energy_forecast_app.loader import CsvEnergyRepository
from energy_forecast_app.validator import NON_NEGATIVE_COLUMNS, EnergyValidator


@pytest.fixture
def energia_mini(fila_valida):
    return pd.DataFrame(
        [
            fila_valida,
            {**fila_valida, "time_utc": "2025-01-15T08:00:00+00:00",
             "time_local": "2025-01-15T09:00:00+01:00", "load_mw": None},
            {**fila_valida, "time_utc": "2025-01-15T09:00:00+00:00",
             "time_local": "2025-01-15T10:00:00+01:00", "relative_humidity_2m": 120.0},
        ]
    )


def test_validador_separa_filas_validas_e_invalidas(energia_mini):
    validos, errores = EnergyValidator().validate_energy(energia_mini)

    assert len(validos) == 1
    assert len(errores) == 2


def test_validador_conserva_datos_de_la_fila_valida(energia_mini):
    validos, _ = EnergyValidator().validate_energy(energia_mini)

    registro = validos[0]
    assert registro.load_mw == 30000.0
    assert registro.energy_mwh == 30000.0
    assert registro.hour_local == 8
    assert registro.year_month == "2025-01"
    assert registro.is_holiday is False


@pytest.mark.parametrize(
    "carga, esperado",
    [(14999.0, 0), (15000.0, 1), (50000.0, 1), (50001.0, 0), (0, 0), (-1, 0), (None, 0), ("abc", 0)],
)
def test_carga_debe_estar_en_rango(fila_valida, carga, esperado):
    datos = pd.DataFrame([{**fila_valida, "load_mw": carga}])

    validos, errores = EnergyValidator().validate_energy(datos)

    assert len(validos) == esperado
    assert len(errores) == 1 - esperado


@pytest.mark.parametrize(
    "temperatura, esperado", [(-20.1, 0), (-20.0, 1), (50.0, 1), (50.1, 0), (None, 0)]
)
def test_temperatura_debe_estar_en_rango(fila_valida, temperatura, esperado):
    datos = pd.DataFrame([{**fila_valida, "temperature_2m": temperatura}])

    validos, _ = EnergyValidator().validate_energy(datos)

    assert len(validos) == esperado


@pytest.mark.parametrize("columna", NON_NEGATIVE_COLUMNS)
def test_variables_meteorologicas_negativas_son_invalidas(fila_valida, columna):
    datos = pd.DataFrame([{**fila_valida, columna: -0.1}])

    validos, errores = EnergyValidator().validate_energy(datos)

    assert validos == []
    assert len(errores) == 1


@pytest.mark.parametrize("indicador", [2, -1, None])
def test_indicadores_deben_ser_0_o_1(fila_valida, indicador):
    datos = pd.DataFrame([{**fila_valida, "is_holiday": indicador}])

    validos, _ = EnergyValidator().validate_energy(datos)

    assert validos == []


def test_fecha_no_parseable_es_invalida(fila_valida):
    datos = pd.DataFrame([{**fila_valida, "time_utc": "no-es-una-fecha"}])

    validos, _ = EnergyValidator().validate_energy(datos)

    assert validos == []


def test_hora_duplicada_invalida_todas_sus_filas(fila_valida):
    datos = pd.DataFrame([fila_valida, {**fila_valida, "load_mw": 31000.0}])

    validos, errores = EnergyValidator().validate_energy(datos)

    assert validos == []
    assert len(errores) == 2


@pytest.mark.parametrize("columna", ["time_utc", "load_mw", "is_holiday"])
def test_columna_obligatoria_ausente_lanza_error(energia_mini, columna):
    with pytest.raises(SchemaError):
        EnergyValidator().validate_energy(energia_mini.drop(columns=columna))


@pytest.mark.integration
def test_muestra_tiene_8723_validas_y_37_invalidas():
    ruta = Path(__file__).parents[2] / "data" / "energy_weather_es_hourly_sample.csv"
    datos = CsvEnergyRepository(ruta).load()

    validos, errores = EnergyValidator().validate_energy(datos)

    assert len(validos) == 8723
    assert len(errores) == 37
