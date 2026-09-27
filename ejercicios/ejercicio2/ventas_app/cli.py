import pandas as pd
from ventas_app.loader import load
from ventas_app.validator import validar_ventas
from ventas_app.metrics import get_importe_region, get_top_3_productos, get_cliente_id
from pathlib import Path

# En cli.py, orquesta con argparse:
# python -m ventas_app.cli --input Datos/ventas.csv --output Datos/ventas_limpias.csv

def main(input_path: Path, output_path: Path):

    # Cargar datos
    df = load(input_path)

    # Validar datos
    validos, errores = validar_ventas(df)

    # Guardar datos válidos en un nuevo archivo CSV
    output_path = Path(output_path)

    if not output_path.is_absolute():
        if output_path.parts and output_path.parts[0] == "Datos":
            output_path = Path(__file__).parent / output_path
        else:
            output_path = Path(__file__).parent / "Datos" / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    validos.to_csv(output_path, index=False)

    # Métricas
    print("Importe por región:")
    print(get_importe_region(validos))
    
    print("\nTop 3 productos por importe:")
    print(get_top_3_productos(validos))
    
    print("\nClientes con más de una compra:")
    print(get_cliente_id(validos))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procesar ventas.")
    parser.add_argument("--input", required=True, help="Ruta del archivo CSV de entrada.")
    parser.add_argument("--output", required=True, help="Ruta del archivo CSV de salida para ventas válidas.")

    args = parser.parse_args()

    main(args.input, args.output)
