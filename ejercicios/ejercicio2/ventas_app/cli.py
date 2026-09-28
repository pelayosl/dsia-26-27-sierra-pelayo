import pandas as pd
from ventas_app.loader import CsvSalesRepository, SalesRepository
from ventas_app.validator import SalesValidator
from ventas_app.metrics import SalesMetrics
from pathlib import Path

# En cli.py, orquesta con argparse:
# python -m ventas_app.cli --input Datos/ventas.csv --output Datos/ventas_limpias.csv

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

    print(f"Registros válidos: {len(records)} | inválidos: {len(errors)}")
    print("Importe por región:")
    for region, total in metrics.total_by_region(records).items():
        print(f"  {region}: {total:.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procesar ventas.")
    parser.add_argument("--input", required=True, help="Ruta del archivo CSV de entrada.")
    parser.add_argument("--output", required=True, help="Ruta del archivo CSV de salida para ventas válidas.")

    args = parser.parse_args()

    main(args.input, args.output)
