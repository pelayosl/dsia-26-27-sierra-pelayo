"""Reads the CSV produced by dsia-dataset (external source, read-only)."""

from pathlib import Path
from typing import Protocol

import pandas as pd
from energy_forecast.domain.errors import DataLoadError



class DatasetSource(Protocol):
    """Abstraction of the external data source (Dependency Inversion)."""
    def read(self) -> pd.DataFrame:
        """Return the source data, unvalidated."""


class CsvDatasetSource:
    """Read `energy_weather_es_hourly.csv` from a local path or a URL."""

    URL_PREFIXES = ("http://", "https://", "file://")

    def __init__(self, location: str | Path) -> None:
        self._location = str(location)

    def read(self) -> pd.DataFrame:
        if not self._location.startswith(self.URL_PREFIXES) and not Path(self._location).exists():
            raise DataLoadError(f"File not found: {self._location}")
        try:
            return pd.read_csv(self._location)
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError(f"Empty file: {self._location}") from exc
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise DataLoadError(f"Malformed CSV: {self._location}") from exc
        except OSError as exc:  # includes URLError / HTTPError (download failures)
            raise DataLoadError(f"Could not read {self._location}: {exc}") from exc
