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