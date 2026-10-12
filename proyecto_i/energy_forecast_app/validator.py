
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from energy_forecast_app.errors import SchemaError

LOCAL_TIMEZONE = "Europe/Madrid"
HOURS_PER_RECORD = 1.0

# Rangos esperados (mismos umbrales que el informe de calidad de dsia-dataset)
LOAD_MW_RANGE = (15_000.0, 50_000.0)
TEMPERATURE_RANGE_C = (-20.0, 50.0)
HUMIDITY_RANGE_PCT = (0.0, 100.0)

NON_NEGATIVE_COLUMNS = ["wind_speed_10m", "shortwave_radiation", "precipitation"]
NUMERIC_COLUMNS = ["load_mw", "temperature_2m", "relative_humidity_2m", *NON_NEGATIVE_COLUMNS]
FLAG_COLUMNS = ["is_holiday", "covid_flag", "load_interpolated", "weather_interpolated"]
REQUIRED_COLUMNS = ["time_utc", "time_local", *NUMERIC_COLUMNS, *FLAG_COLUMNS]


@dataclass(frozen=True)
class EnergyRecord:
    time_utc: datetime
    time_local: datetime
    load_mw: float
    temperature_2m: float
    relative_humidity_2m: float
    wind_speed_10m: float
    shortwave_radiation: float
    precipitation: float
    is_holiday: bool
    covid_flag: bool
    load_interpolated: bool
    weather_interpolated: bool

    @property
    def energy_mwh(self) -> float:
        '''Energía consumida en la hora (demanda media x duración del registro).'''
        return self.load_mw * HOURS_PER_RECORD

    @property
    def hour_local(self) -> int:
        return self.time_local.hour

    @property
    def year_month(self) -> str:
        return self.time_local.strftime("%Y-%m")


class EnergyValidator:

    def validate_energy(self, frame: pd.DataFrame) -> tuple[list[EnergyRecord], pd.DataFrame]:
        '''
        Reglas:

        - deben existir todas las columnas de `REQUIRED_COLUMNS` (si no, `SchemaError`)
        - `time_utc` y `time_local` fechas ISO 8601 válidas
        - `time_utc` no duplicado (se invalidan todas las filas de una hora repetida)
        - `load_mw` numérica y dentro de `LOAD_MW_RANGE`
        - `temperature_2m` dentro de `TEMPERATURE_RANGE_C`
        - `relative_humidity_2m` dentro de `HUMIDITY_RANGE_PCT`
        - viento, radiación y precipitación numéricas y `>= 0`
        - indicadores (`is_holiday`, `covid_flag`, `*_interpolated`) con valor 0 o 1

        Los nulos invalidan la fila en todas las reglas.

        Devuelve `(validos, errores)`.
        '''
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise SchemaError(f"Missing required columns: {missing}")

        temp = frame.copy()

        temp["time_utc"] = pd.to_datetime(temp["time_utc"], utc=True, format="ISO8601", errors="coerce")
        temp["time_local"] = pd.to_datetime(
            temp["time_local"], utc=True, format="ISO8601", errors="coerce"
        ).dt.tz_convert(LOCAL_TIMEZONE)
        for column in NUMERIC_COLUMNS + FLAG_COLUMNS:
            temp[column] = pd.to_numeric(temp[column], errors="coerce")

        valid_vals = (
            temp["time_utc"].notna() &
            ~temp["time_utc"].duplicated(keep=False) &
            temp["time_local"].notna() &
            temp["load_mw"].between(*LOAD_MW_RANGE) &
            temp["temperature_2m"].between(*TEMPERATURE_RANGE_C) &
            temp["relative_humidity_2m"].between(*HUMIDITY_RANGE_PCT) &
            temp[NON_NEGATIVE_COLUMNS].ge(0).all(axis=1) &
            temp[FLAG_COLUMNS].isin([0, 1]).all(axis=1)
        )

        validos = [
                EnergyRecord(
                    time_utc=row.time_utc,
                    time_local=row.time_local,
                    load_mw=float(row.load_mw),
                    temperature_2m=float(row.temperature_2m),
                    relative_humidity_2m=float(row.relative_humidity_2m),
                    wind_speed_10m=float(row.wind_speed_10m),
                    shortwave_radiation=float(row.shortwave_radiation),
                    precipitation=float(row.precipitation),
                    is_holiday=bool(row.is_holiday),
                    covid_flag=bool(row.covid_flag),
                    load_interpolated=bool(row.load_interpolated),
                    weather_interpolated=bool(row.weather_interpolated),
                )
                for row in temp.loc[valid_vals].itertuples(index=False)
        ]
        errores = temp.loc[~valid_vals].copy()
        return validos, errores
