"""Punto de entrada por línea de comandos del pipeline de energía."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_forecast_app.loader import CsvEnergyRepository, EnergyRepository
from energy_forecast_app.metrics import EnergyMetrics
from energy_forecast_app.validator import EnergyValidator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Orquesta con argparse:
# python -m energy_forecast_app.cli --input data/energy_weather_es_hourly_sample.csv --output data/energy_weather_es_hourly_limpio.csv

def main(input_path: Path, output_path: Path):
    """Carga, valida y guarda las horas válidas, y muestra las métricas de demanda.

    Si `output_path` es relativa, el fichero se guarda dentro de `proyecto_i/data/`.
    """
    repo: EnergyRepository = CsvEnergyRepository(input_path)
    validator = EnergyValidator()
    metrics = EnergyMetrics()

    # Cargar datos
    df = repo.load()
    # Validar datos
    records, errors = validator.validate_energy(df)

    # Guardar datos válidos en un nuevo archivo CSV
    output_path = Path(output_path)

    if not output_path.is_absolute():
        if output_path.parts and output_path.parts[0] == "data":
            output_path = Path(__file__).parents[1] / output_path
        else:
            output_path = Path(__file__).parents[1] / "data" / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_df = pd.DataFrame(
    [
        {
            "time_utc": r.time_utc.isoformat(),
            "time_local": r.time_local.isoformat(),
            "load_mw": r.load_mw,
            "temperature_2m": r.temperature_2m,
            "relative_humidity_2m": r.relative_humidity_2m,
            "wind_speed_10m": r.wind_speed_10m,
            "shortwave_radiation": r.shortwave_radiation,
            "precipitation": r.precipitation,
            "is_holiday": int(r.is_holiday),
            "covid_flag": int(r.covid_flag),
            "load_interpolated": int(r.load_interpolated),
            "weather_interpolated": int(r.weather_interpolated),
            "energy_mwh": r.energy_mwh,
        }
        for r in records
    ]
)
    valid_df.to_csv(output_path, index=False)

    logger.info("Registros válidos: %s | inválidos: %s", len(records), len(errors))
    logger.info("Energía por mes (GWh):")
    for month, total in metrics.total_energy_by_month(records).items():
        logger.info("  %s: %.2f", month, total)
    logger.info("Demanda media por hora local (MW):")
    for hour, mean in metrics.mean_load_by_hour(records).items():
        logger.info("  %02d h: %.0f", hour, mean)
    logger.info("Demanda media por tramo de temperatura (MW):")
    for band, mean in metrics.mean_load_by_temperature_band(records).items():
        logger.info("  %s: %.0f", band, mean)
    logger.info("Demanda media por tipo de día (MW):")
    for day_type, mean in metrics.mean_load_by_day_type(records).items():
        logger.info("  %s: %.0f", day_type, mean)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lee `--input` y `--output`; con `argv=None` usa los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Procesar demanda eléctrica horaria y meteorología.")
    parser.add_argument("--input", required=True, help="Ruta del archivo CSV de entrada.")
    parser.add_argument("--output", required=True, help="Ruta del archivo CSV de salida para registros válidos.")

    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()

    main(args.input, args.output)
