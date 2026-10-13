"""Carga de los datos horarios de energía y meteorología."""

from pathlib import Path
from typing import Protocol
import pandas as pd
from energy_forecast_app.errors import DataLoadError

DATA_DIR = Path(__file__).parents[1] / "data"

class EnergyRepository(Protocol):
    """Abstracción de lectura (Dependency Inversion)."""
    def load(self) -> pd.DataFrame:
        """Devuelve los datos de origen como DataFrame."""


class CsvEnergyRepository:
    """Lee el CSV horario de demanda eléctrica + meteorología."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> pd.DataFrame:
        """Carga el CSV; lanza `DataLoadError` si el fichero no existe o está vacío."""
        if not self._path.exists():
            raise DataLoadError(f"File not found: {self._path}")

        try:
            return pd.read_csv(self._path)
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError(f"Empty file: {self._path}") from exc
