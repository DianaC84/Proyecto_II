import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os
import joblib


# 1. Crear variables de materialización

def crear_features(df: pd.DataFrame) -> pd.DataFrame:

    # MATERIALIZADO GLOBAL

    riesgos_materializados = (
        df.groupby("Riesgo_CD")
        .apply(lambda x: any((x["ValorInterno_VR"] > 0) | (x["ValorCliente_VR"] > 0)))
        .reset_index(name="alguna_vez_materializado")
    )

    df = df.merge(riesgos_materializados, on="Riesgo_CD", how="left")
    df["materializado_global"] = np.where(df["alguna_vez_materializado"], 1, 0)

    # MATERIALIZADO EN 30 DÍAS

    def marcar_materializacion_30d(grupo):
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

        grupo["materializado_30d"] = resultado
        return grupo

    df = df.groupby("Riesgo_CD", group_keys=False).apply(marcar_materializacion_30d)
    df["materializado_30d"] = np.where(df["materializado_30d"], 1, 0)

    return df

# 2. Valor acumulado

def crear_valor_acumulado(df: pd.DataFrame) -> pd.DataFrame:
    df["valor_acumulado"] = (
        df["ValorInterno_VR"].fillna(0) +
        df["ValorCliente_VR"].fillna(0)
    )
    return df


# 3. Limpieza final

def limpieza_final(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["FechaSeguimiento_DT", "Severidad_DS", "materializado_30d"])
    return df

# 4. Variables temporales

def crear_variables_temporales(df: pd.DataFrame) -> pd.DataFrame:
    df["FechaSeguimiento_DT"] = pd.to_datetime(df["FechaSeguimiento_DT"])
    df["mes"] = df["FechaSeguimiento_DT"].dt.month
    df["año"] = df["FechaSeguimiento_DT"].dt.year
    return df

# 5. Mapear severidad a número

def mapear_severidad(df: pd.DataFrame) -> pd.DataFrame:

    mapa_severidad = {
        "Riesgo Bajo": 1,
        "Riesgo Medio": 2,
        "Riesgo Significativo": 3,
        "Riesgo Alto": 4,
        "Riesgo Crítico": 5
    }

    df["Severidad_NUM"] = df["Severidad_DS"].map(mapa_severidad)
    return df


# 6. Codificar categorías

def codificar_categorias(df: pd.DataFrame):
    le_causa = LabelEncoder()
    le_agente = LabelEncoder()

    df["Causa_DS"] = le_causa.fit_transform(df["Causa_DS"].astype(str))
    df["AgenteGenerador_DS"] = le_agente.fit_transform(df["AgenteGenerador_DS"].astype(str))

    encoders = {
        "Causa_DS": le_causa,
        "AgenteGenerador_DS": le_agente
    }
    os.makedirs("encoders", exist_ok=True)
    joblib.dump(encoders["Causa_DS"], "encoders/encoder_Causa_DS.pkl")
    joblib.dump(encoders["AgenteGenerador_DS"], "encoders/encoder_AgenteGenerador_DS.pkl")

    return df, encoders

# 7. FUNCIÓN PRINCIPAL QUE ORDENA LAS TRANSFORMACIONES

def procesar_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = crear_features(df)
    df = crear_valor_acumulado(df)
    df = limpieza_final(df)
    df = crear_variables_temporales(df)
    df = mapear_severidad(df)
    df, encoders = codificar_categorias(df)

    return df

# EJECUCIÓN DIRECTA DEL SCRIPT

if __name__ == "__main__":
    print("Cargando dataset limpio generado por el ETL...")
    df = pd.read_excel("datos_transformados.xlsx")

    print("Creando nuevas variables y transformando dataset...")
    df = procesar_dataset(df)

    df.to_excel("datos_features.xlsx", index=False)

    print("Archivo final generado: datos_features.xlsx")
