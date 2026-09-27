
import pandas as pd

def validar_ventas(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    validos = temp.loc[valid_vals].copy()
    errores = temp.loc[~valid_vals].copy()
    validos["importe"] = validos["unidades"] * validos["precio_unitario"]

    return validos, errores