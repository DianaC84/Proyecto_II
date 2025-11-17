
import pandas as pd
import numpy as np

def marcar_materializacion_30d(grupo: pd.DataFrame) -> pd.DataFrame:
    """
    Marca si en los 30 días siguientes a cada fila existe materialización
    interna o del cliente dentro del mismo Riesgo_CD.
    """

    fechas = grupo["FechaSeguimiento_DT"].values.astype("datetime64[ns]")
    interno = grupo["ValorInterno_VR"].fillna(0).values
    cliente = grupo["ValorCliente_VR"].fillna(0).values

    resultado = []

    for i, f in enumerate(fechas):
        fecha_limite = f + np.timedelta64(30, "D")
        hay_evento = np.any(
            (fechas > f)
            & (fechas <= fecha_limite)
            & ((interno > 0) | (cliente > 0))
        )
        resultado.append(hay_evento)

    grupo = grupo.copy()
    grupo["materializado_30d"] = resultado
    return grupo


def crear_materializado_30d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la función marcar_materializacion_30d por Riesgo_CD
    y devuelve df con la columna materializado_30d binaria (0/1).
    """
    df = df.groupby("Riesgo_CD", group_keys=False).apply(marcar_materializacion_30d)
    df["materializado_30d"] = np.where(df["materializado_30d"], 1, 0)
    return df