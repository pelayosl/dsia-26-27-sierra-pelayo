"""Métricas sobre las horas válidas de demanda y meteorología."""

import math
from collections.abc import Callable, Hashable

from energy_forecast_app.validator import EnergyRecord

MWH_PER_GWH = 1_000.0
TEMPERATURE_BAND_WIDTH_C = 5


class EnergyMetrics:
    """Agregaciones sobre registros válidos. Las claves de tiempo van en orden cronológico."""

    def total_energy_by_month(self, validos: list[EnergyRecord]) -> dict[str, float]:
        '''Energía total consumida por mes local (GWh).'''
        totals: dict[str, float] = {}
        for record in validos:
            totals[record.year_month] = totals.get(record.year_month, 0.0) + record.energy_mwh

        return {month: total / MWH_PER_GWH for month, total in sorted(totals.items())}

    def mean_load_by_hour(self, validos: list[EnergyRecord]) -> dict[int, float]:
        '''Perfil diario: demanda media (MW) por hora local.'''
        return self._mean_load_by(validos, key=lambda record: record.hour_local)

    def mean_load_by_temperature_band(self, validos: list[EnergyRecord]) -> dict[str, float]:
        '''Demanda media (MW) por tramo de temperatura de `TEMPERATURE_BAND_WIDTH_C` grados.'''
        by_band_start = self._mean_load_by(
            validos,
            key=lambda record: math.floor(record.temperature_2m / TEMPERATURE_BAND_WIDTH_C)
            * TEMPERATURE_BAND_WIDTH_C,
        )

        return {
            f"[{start}, {start + TEMPERATURE_BAND_WIDTH_C}) °C": mean
            for start, mean in by_band_start.items()
        }

    def mean_load_by_day_type(self, validos: list[EnergyRecord]) -> dict[str, float]:
        '''Demanda media (MW) en festivos nacionales frente al resto de días.'''
        by_holiday = self._mean_load_by(validos, key=lambda record: record.is_holiday)

        labels = {False: "no festivo", True: "festivo"}
        return {labels[is_holiday]: mean for is_holiday, mean in by_holiday.items()}

    @staticmethod
    def _mean_load_by(
        validos: list[EnergyRecord], key: Callable[[EnergyRecord], Hashable]
    ) -> dict:
        """Demanda media por grupo; `key` indica a qué grupo pertenece cada registro."""
        sums: dict = {}
        counts: dict = {}
        for record in validos:
            group = key(record)
            sums[group] = sums.get(group, 0.0) + record.load_mw
            counts[group] = counts.get(group, 0) + 1

        return {group: sums[group] / counts[group] for group in sorted(sums)}
