"""Validación de las ventas y conversión de las filas válidas a `SalesRecord`."""

import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class SalesRecord:
    """Venta válida (inmutable)."""
    region: str
    product: str
    units: float
    unit_price: float

    @property
    def amount(self) -> float:
        """Importe de la venta: unidades x precio unitario."""
        return self.units * self.unit_price


class SalesValidator:
    """Separa las ventas válidas de las inválidas (solo valida, SRP)."""

    def validate_sales(self, frame: pd.DataFrame) -> tuple[list[SalesRecord], pd.DataFrame]:
        '''
        Reglas mínimas:

        - `unidades` numérica y `> 0`
        - `precio_unitario` numérico y `> 0`
        - columnas derivadas: `importe = unidades * precio_unitario` solo en válidos

        Devuelve `(validos, errores)`.
        '''
        temp = frame.copy()

        temp["unidades"] = pd.to_numeric(temp["unidades"], errors = "coerce")
        temp["precio_unitario"] = pd.to_numeric(temp["precio_unitario"], errors = "coerce")

        valid_vals = (
            temp["unidades"].notna() &
            (temp["unidades"] > 0) &
            temp["precio_unitario"].notna() &
            (temp["precio_unitario"] > 0)
        )

        validos = [
                SalesRecord(
                    region=str(row.region),
                    product=str(row.producto),
                    units=float(row.unidades),
                    unit_price=float(row.precio_unitario),
                )
                for row in temp.loc[valid_vals].itertuples(index=False)
        ]
        errores = temp.loc[~valid_vals].copy()
        return validos, errores