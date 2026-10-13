"""Métricas de negocio sobre las ventas válidas."""

import pandas as pd
from ventas_app.validator import SalesRecord

class SalesMetrics:
    """Agregaciones sobre registros de venta válidos."""
    def total_by_region(self, validos: list[SalesRecord]) -> dict[str, float]:
        """Importe total por región, ordenado de mayor a menor."""
        totals: dict[str, float] = {}
        for record in validos:
            totals[record.region] = totals.get(record.region, 0.0) + record.amount

        return dict(
            sorted(
                totals.items(), 
                key=lambda item: item[1], reverse=True
                )
            )