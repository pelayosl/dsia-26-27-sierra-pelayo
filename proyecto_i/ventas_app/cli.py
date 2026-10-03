import logging
from pathlib import Path

import pandas as pd

from ventas_app.loader import CsvSalesRepository, SalesRepository
from ventas_app.metrics import SalesMetrics
from ventas_app.validator import SalesValidator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# En cli.py, orquesta con argparse:
# python -m ventas_app.cli --input ventas_app/Datos/ventas.csv --output Datos/ventas_limpias.csv

def main(input_path: Path, output_path: Path):
    repo: SalesRepository = CsvSalesRepository(input_path)
    validator = SalesValidator()
    metrics = SalesMetrics()

    # Cargar datos
    df = repo.load()
    # Validar datos
    records, errors = validator.validate_sales(df)

    # Guardar datos válidos en un nuevo archivo CSV
    output_path = Path(output_path)

    if not output_path.is_absolute():
        if output_path.parts and output_path.parts[0] == "Datos":
            output_path = Path(__file__).parent / output_path
        else:
            output_path = Path(__file__).parent / "Datos" / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_df = pd.DataFrame(
    [
        {
            "region": r.region,
            "product": r.product,
            "units": r.units,
            "unit_price": r.unit_price,
            "importe": r.amount,
        }
        for r in records
    ]
)
    valid_df.to_csv(output_path, index=False)

    logger.info("Registros válidos: %s | inválidos: %s", len(records), len(errors))
    logger.info("Importe por región:")
    for region, total in metrics.total_by_region(records).items():
        logger.info("  %s: %.2f", region, total)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procesar ventas.")
    parser.add_argument("--input", required=True, help="Ruta del archivo CSV de entrada.")
    parser.add_argument("--output", required=True, help="Ruta del archivo CSV de salida para ventas válidas.")

    args = parser.parse_args()

    main(args.input, args.output)
