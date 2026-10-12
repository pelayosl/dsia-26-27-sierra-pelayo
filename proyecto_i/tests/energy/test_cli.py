import logging
from pathlib import Path

import pandas as pd
import pytest

from energy_forecast_app.cli import main, parse_args


def test_cli_genera_csv_con_registros_validos(tmp_path: Path, fila_valida, caplog):
    input_path = tmp_path / "energia.csv"
    output_path = tmp_path / "energia_limpia.csv"
    pd.DataFrame(
        [
            fila_valida,
            {**fila_valida, "time_utc": "2025-01-15T08:00:00+00:00",
             "time_local": "2025-01-15T09:00:00+01:00", "load_mw": 0.0},
        ]
    ).to_csv(input_path, index=False)

    with caplog.at_level(logging.INFO):
        main(input_path, output_path)

    result = pd.read_csv(output_path)
    assert len(result) == 1
    assert result.loc[0, "time_local"] == "2025-01-15T08:00:00+01:00"
    assert result.loc[0, "energy_mwh"] == 30000.0
    assert "Registros válidos: 1 | inválidos: 1" in caplog.text


def test_parse_args_lee_entrada_y_salida():
    args = parse_args(["--input", "entrada.csv", "--output", "salida.csv"])

    assert args.input == "entrada.csv"
    assert args.output == "salida.csv"


def test_parse_args_sin_argumentos_obligatorios_termina():
    with pytest.raises(SystemExit):
        parse_args([])
