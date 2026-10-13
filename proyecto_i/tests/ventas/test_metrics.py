"""Tests de las métricas de `ventas_app.metrics`."""

from ventas_app.metrics import SalesMetrics
from ventas_app.validator import SalesRecord


def test_metricas_ordenan_totales_por_region_descendente():
    """El importe se suma por región y se ordena de mayor a menor."""
    validos = [
        SalesRecord(region="Sur", product="A", units=1, unit_price=10),
        SalesRecord(region="Norte", product="B", units=3, unit_price=10),
        SalesRecord(region="Sur", product="C", units=2, unit_price=10),
    ]

    totales = SalesMetrics().total_by_region(validos)

    assert list(totales.items()) == [("Sur", 30.0), ("Norte", 30.0)]
