from pathlib import Path

import pandas as pd

from ventas_app.cli import main


def test_cli_genera_csv_con_registros_validos(tmp_path: Path):
    input_path = tmp_path / "ventas.csv"
    output_path = tmp_path / "ventas_limpias.csv"
    pd.DataFrame(
        [
            {
                "fecha": "2026-01-01",
                "region": "Norte",
                "producto": "A",
                "unidades": 2,
                "precio_unitario": 10,
            }
        ]
    ).to_csv(input_path, index=False)

    main(input_path, output_path)

    result = pd.read_csv(output_path)
    assert len(result) == 1
    assert result.loc[0, "importe"] == 20
