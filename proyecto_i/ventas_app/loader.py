from pathlib import Path
from typing import Protocol
import pandas as pd
from ventas_app.errors import DataLoadError

DATA_DIR = Path(__file__).parent / "Datos"

class SalesRepository(Protocol):
    """Abstracción de lectura (Dependency Inversion)."""
    def load(self) -> pd.DataFrame: ...


class CsvSalesRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        
    def load(self) -> pd.DataFrame:
        if not self._path.exists():
            raise DataLoadError(f"File not found: {self._path}")

        return pd.read_csv(self._path)

class JsonSalesRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        
    def load(self) -> pd.DataFrame:
        if not self._path.exists():
            raise DataLoadError(f"File not found: {self._path}")

        return pd.read_json(self._path)