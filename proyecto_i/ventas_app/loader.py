"""Carga de los datos de ventas desde fichero (CSV o JSON)."""

from pathlib import Path
from typing import Protocol
import pandas as pd
from ventas_app.errors import DataLoadError

DATA_DIR = Path(__file__).parents[1] / "data"

class SalesRepository(Protocol):
    """Abstracción de lectura (Dependency Inversion)."""
    def load(self) -> pd.DataFrame:
        """Devuelve los datos de origen como DataFrame."""


class CsvSalesRepository:
    """Lee las ventas desde un fichero CSV."""
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        
    def load(self) -> pd.DataFrame:
        """Carga el CSV; lanza `DataLoadError` si el fichero no existe."""
        if not self._path.exists():
            raise DataLoadError(f"File not found: {self._path}")

        return pd.read_csv(self._path)

class JsonSalesRepository:
    """Lee las ventas desde un fichero JSON (lista de registros)."""
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        
    def load(self) -> pd.DataFrame:
        """Carga el JSON; lanza `DataLoadError` si el fichero no existe."""
        if not self._path.exists():
            raise DataLoadError(f"File not found: {self._path}")

        return pd.read_json(self._path)