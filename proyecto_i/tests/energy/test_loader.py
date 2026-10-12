import pandas as pd
import pytest

from energy_forecast_app.errors import DataLoadError
from energy_forecast_app.loader import CsvEnergyRepository


def test_carga_de_csv_valido(tmp_path, fila_valida):
    ruta = tmp_path / "energia.csv"
    pd.DataFrame([fila_valida]).to_csv(ruta, index=False)

    cargado = CsvEnergyRepository(ruta).load()

    assert list(cargado.columns) == list(fila_valida)
    assert list(cargado["load_mw"]) == [30000.0]


def test_carga_de_csv_inexistente_lanza_error(tmp_path):
    repositorio = CsvEnergyRepository(tmp_path / "no-existe.csv")

    with pytest.raises(DataLoadError):
        repositorio.load()


def test_carga_de_csv_vacio_lanza_error(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text("")

    with pytest.raises(DataLoadError):
        CsvEnergyRepository(ruta).load()
