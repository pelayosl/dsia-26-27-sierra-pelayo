from pathlib import Path

import pandas as pd
import pytest

from ventas_app.errors import DataLoadError
from ventas_app.loader import CsvSalesRepository
from ventas_app.validator import SalesValidator


@pytest.fixture
def ventas_mini():
    return pd.DataFrame(
        {
            "fecha": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "region": ["Norte", "Sur", "Norte"],
            "producto": ["A", "B", "A"],
            "unidades": [2, None, 5],
            "precio_unitario": [10.0, 20.0, -1.0],
            "cliente_id": ["C1", "C2", "C1"],
        }
    )


def test_validador_separa_filas_validas_e_invalidas(ventas_mini):
    validos, errores = SalesValidator().validate_sales(ventas_mini)

    assert len(validos) == 1
    assert len(errores) == 2


def test_validador_conserva_datos_de_la_fila_valida(ventas_mini):
    validos, _ = SalesValidator().validate_sales(ventas_mini)

    assert validos[0].region == "Norte"
    assert validos[0].product == "A"
    assert validos[0].amount == 20.0


def test_unidades_nulas_o_no_positivas_son_invalidas(ventas_mini):
    datos = ventas_mini.iloc[[0, 1]].copy()
    datos.loc[datos.index[0], "unidades"] = 0

    validos, errores = SalesValidator().validate_sales(datos)

    assert validos == []
    assert len(errores) == 2


@pytest.mark.parametrize("precio, esperado", [(0, 0), (-1, 0), (1, 1)])
def test_precio_debe_ser_positivo(ventas_mini, precio, esperado):
    datos = ventas_mini.iloc[[0]].copy()
    datos["precio_unitario"] = precio

    validos, errores = SalesValidator().validate_sales(datos)

    assert len(validos) == esperado
    assert len(errores) == 1 - esperado


def test_carga_de_csv_inexistente_lanza_error(tmp_path):
    repositorio = CsvSalesRepository(tmp_path / "no-existe.csv")

    with pytest.raises(DataLoadError):
        repositorio.load()


@pytest.mark.integration
def test_csv_curso_tiene_140_validas_y_10_invalidas():
    ruta = Path(__file__).parents[1] / "Datos" / "ventas.csv"
    datos = CsvSalesRepository(ruta).load()

    validos, errores = SalesValidator().validate_sales(datos)

    assert len(validos) == 140
    assert len(errores) == 10
