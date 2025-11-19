import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Función para cargar los datos

def cargar_datos(ruta_archivo: str) -> pd.DataFrame:

    try:
        df = pd.read_excel(ruta_archivo)
        print("Archivo cargado correctamente.")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        raise

# Función para limpiar datos

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
   

    columnas_eliminar = [
        "Proyecto_DS", "UEN_DS", "NombreEmpleado_DS", "ApellidoEmpleado_DS",
        "Creacion_DT", "Actualizacion_DT", "CantidadSeveridad_NM", "Insercion_DT",
        "NombreRiesgo_DS", "Fuente_DS", "DescripcionRiesgo_DS", "FechaIdentificacion_DT",
        "PlanAccion_DS", "ObservacionSeguimiento_DS", "CantidadSeguimientos_NM",
        "EsfuerzoSeguimiento_NM", "Periocidad_DS", "Cliente_DS",
        "PRY_REQUIERE_RIESGOS", "Tipo_Proyecto", "Metodologia"
    ]

    df = df.drop(columns=[c for c in columnas_eliminar if c in df.columns], errors="ignore")

    # Filtrar registros vigentes
    df = df[df["Estado_DS"] != "NO VIGENTE"]

    return df

# Función para transformar datos

def transformar_datos(df: pd.DataFrame) -> pd.DataFrame:
   

    # Definir columnas categóricas
    columnas_categoricas = [
        "Severidad_DS", "Estado_DS", "Causa_DS",
        "DetalleCausa_DS", "AgenteGenerador_DS"
    ]

    for col in columnas_categoricas:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convertir FechaSeguimiento_DT a datetime
    if "FechaSeguimiento_DT" in df.columns:
        df["FechaSeguimiento_DT"] = pd.to_datetime(df["FechaSeguimiento_DT"], errors="coerce")
        df["Año"] = df["FechaSeguimiento_DT"].dt.year

        df = df[df["Año"] >= 2023]

    return df

# Función principal

def ejecutar_etl(ruta_archivo: str) -> pd.DataFrame:
    df = cargar_datos(ruta_archivo)
    df = limpiar_datos(df)
    df = transformar_datos(df)

    print("ETL ejecutado correctamente.")
    print(f"Filas finales: {df.shape[0]}")
    print(f"Columnas finales: {df.shape[1]}")

    return df


# Punto de entrada

if __name__ == "__main__":
    ruta = r"C:\Users\crisa\Documents\repositorios\Proyecto_II\datos\Bas_de_datos_riesgos_original_30.09.2025.xlsx"

    df_final = ejecutar_etl(ruta)

    # Guardar resultado opcional
    df_final.to_excel("datos_transformados.xlsx", index=False)
    print("Archivo transformado guardado como datos_transformados.xlsx")

