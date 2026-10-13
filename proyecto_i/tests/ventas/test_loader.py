"""Tests de los repositorios de `ventas_app.loader`."""

import pandas as pd
import pytest

from ventas_app.errors import DataLoadError
from ventas_app.loader import CsvSalesRepository, JsonSalesRepository


def test_carga_de_csv_valido(tmp_path):
    """Un CSV existente se carga como DataFrame con sus columnas."""
    ruta = tmp_path / "ventas.csv"
    pd.DataFrame([{"region": "Norte", "importe": 20.0}]).to_csv(
        ruta, index=False
    )

    cargado = CsvSalesRepository(ruta).load()

    assert list(cargado["region"]) == ["Norte"]
    assert list(cargado["importe"]) == [20.0]


def test_carga_de_csv_inexistente_lanza_error(tmp_path):
    """Un CSV que no existe lanza `DataLoadError`."""
    repositorio = CsvSalesRepository(tmp_path / "no-existe.csv")

    with pytest.raises(DataLoadError):
        repositorio.load()


def test_carga_de_json_inexistente_lanza_error(tmp_path):
    """Un JSON que no existe lanza `DataLoadError`."""
    repositorio = JsonSalesRepository(tmp_path / "no-existe.json")

    with pytest.raises(DataLoadError):
        repositorio.load()


def test_carga_de_json_valido(tmp_path):
    """Un JSON de registros se carga como DataFrame con sus columnas."""
    ruta = tmp_path / "ventas.json"
    pd.DataFrame([{"region": "Norte", "importe": 20.0}]).to_json(
        ruta, orient="records"
    )

    cargado = JsonSalesRepository(ruta).load()

    assert list(cargado["region"]) == ["Norte"]
    assert list(cargado["importe"]) == [20.0]
