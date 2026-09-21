from __future__ import annotations

from datetime import date
from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_DIR = Path("Datos")
# Parte 1 - Diagnóstico

def load_data(__file__: str | Path, dir: str | Path) -> pd.DataFrame:
    data_path = Path(dir)
    return pd.read_csv(data_path / __file__)

def get_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.isna().any(axis=1)]

df = load_data("ventas.csv", DATA_DIR)

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nNumber of null values:")
print(df.isna().sum())

print("\nNull rows:")
print(get_na_rows(df))

# Utilizando el método get_na_rows(), obtenemos las filas con valores nulos:
# fecha region  producto unidades  precio_unitario cliente_id
# 54  2026-02-01    Sur  Sensor B      NaN             32.0       C004
# 63  2026-02-11  Norte  Sensor A      NaN             19.5       C018
# 66  2026-02-17  Oeste   Hub IoT        5              NaN       C025
#
# Aparentemente, estas serían las filas consideradas inválidas debido a la falta de precio unitario
# y de id del cliente.
# ------------------------

# Parte 2 - Validación

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

validos, errores = validar_ventas(df)
print("="*40)
print(f"Ventas válidas: {len(validos)}")
print(f"Ventas inválidas: {len(errores)}")

# Parte 3 - Agregaciones

def get_importe_region(validos: pd.DataFrame):
    return (
        validos.groupby("region")["importe"]
        .sum()
        .sort_values(ascending=False)
    )

def get_top_3_productos(validos: pd.DataFrame):
    return (
        validos.groupby("producto")["importe"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )

def get_cliente_id(validos: pd.DataFrame):
    compras_cliente = validos["cliente_id"].value_counts()
    return compras_cliente[compras_cliente > 1]

print(f"Importe por región: {get_importe_region(validos)}")
print(f"Top 3 productos por importe: {get_top_3_productos(validos)}")
print(f"Clientes con más de una compra: {get_cliente_id(validos)}")

# Parte 4 - Exportación
def exportar(validos: pd.DataFrame, errores: pd.DataFrame):
    ventas_csv = DATA_DIR / "ventas_limpias.csv"
    calidad_datos_json = DATA_DIR / "calidad_datos.json"

    validos.to_csv(ventas_csv, index=False)

    calidad_data = {
        "filas_totales": len(df),
        "filas_validas": len(validos),
        "filas_invalidas": len(errores),
        "importe_total": validos["importe"].sum()
    }
    calidad_datos_json.write_text(json.dumps(calidad_data), encoding="utf-8")
    return ventas_csv, calidad_datos_json

ventas, calidad = exportar(validos, errores)
print(f"escrito: {ventas}")
print(f"escrito: {calidad}")

