import logging
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw(path: Union[str, Path]) -> pd.DataFrame:
    """Cargar archivo Excel/CSV en un DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    if path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    logger.info("Datos cargados: %s (filas=%d, cols=%d)", path, df.shape[0], df.shape[1])
    return df

def drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Eliminar columnas no necesarias (ignora errores)."""
    return df.drop(columns=cols, errors="ignore")

def marcar_materializacion_30d(grupo: pd.DataFrame,
                               fecha_col: str = "FechaSeguimiento_DT",
                               interno_col: str = "ValorInterno_VR",
                               cliente_col: str = "ValorCliente_VR") -> pd.DataFrame:
    """Marca si en los 30 días siguientes a cada fila existe materialización."""
    fechas = grupo[fecha_col].values.astype("datetime64[ns]")
    interno = grupo[interno_col].fillna(0).values
    cliente = grupo[cliente_col].fillna(0).values
    resultado = []
    for i, f in enumerate(fechas):
        fecha_limite = f + np.timedelta64(30, "D")
        hay_evento = np.any(
            (fechas > f) & (fechas <= fecha_limite) & ((interno > 0) | (cliente > 0))
        )
        resultado.append(hay_evento)
    grupo = grupo.copy()
    grupo["materializado_30d"] = np.where(resultado, 1, 0)
    return grupo
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y creación de variables mínimas según notebook."""
    # Columnas a eliminar (ajustar según necesidad)
    columnas_eliminar = [
        "Proyecto_DS", "UEN_DS", "NombreEmpleado_DS", "ApellidoEmpleado_DS",
        "Creacion_DT", "Actualizacion_DT", "CantidadSeveridad_NM", "Insercion_DT",
        "NombreRiesgo_DS", "Fuente_DS", "DescripcionRiesgo_DS", "FechaIdentificacion_DT",
        "PlanAccion_DS", "ObservacionSeguimiento_DS", "CantidadSeguimientos_NM",
        "EsfuerzoSeguimiento_NM", "Periocidad_DS", "Cliente_DS",
        "PRY_REQUIERE_RIESGOS", "Tipo_Proyecto", "Metodologia"
    ]
    df = drop_columns(df, columnas_eliminar)

    # Filtrar riesgos vigentes si columna existe
    if "Estado_DS" in df.columns:
        df = df[df["Estado_DS"] != "NO VIGENTE"]

    # Fechas
    if "FechaSeguimiento_DT" in df.columns:
        df["FechaSeguimiento_DT"] = pd.to_datetime(df["FechaSeguimiento_DT"], errors="coerce")
        df["Año"] = df["FechaSeguimiento_DT"].dt.year
        # Filtrar desde 2023
        df = df[df["Año"].ge(2023)]
# Marcadores y valores
    df["ValorInterno_VR"] = df.get("ValorInterno_VR", pd.Series(dtype=float)).fillna(0)
    df["ValorCliente_VR"] = df.get("ValorCliente_VR", pd.Series(dtype=float)).fillna(0)
    df["valor_acumulado"] = df["ValorInterno_VR"] + df["ValorCliente_VR"]

    # materializado_30d por Riesgo_CD
    if {"Riesgo_CD", "FechaSeguimiento_DT", "ValorInterno_VR", "ValorCliente_VR"}.issubset(df.columns):
        df = df.groupby("Riesgo_CD", group_keys=False).apply(marcar_materializacion_30d)
    else:
        df["materializado_30d"] = 0

# Mapear severidad a numérico si existe
    if "Severidad_DS" in df.columns:
        mapa_severidad = {
            "Riesgo Bajo": 1,
            "Riesgo Medio": 2,
            "Riesgo Significativo": 3,
            "Riesgo Alto": 4,
            "Riesgo Crítico": 5
        }
        df["Severidad_NUM"] = df["Severidad_DS"].map(mapa_severidad).fillna(df.get("Severidad_NUM", np.nan))

    # Tipos categóricos
    cat_cols = [c for c in ["Severidad_DS", "Estado_DS", "Causa_DS", "DetalleCausa_DS", "AgenteGenerador_DS"] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    # Asegurar binario en materializado_30d
    if "materializado_30d" in df.columns:
        df["materializado_30d"] = df["materializado_30d"].astype(int)

    logger.info("Limpieza finalizada: filas=%d, cols=%d", df.shape[0], df.shape[1])
    return df

def save_processed(df: pd.DataFrame, out_dir: Union[str, Path]) -> None:
    """Guardar dataframe procesado únicamente en Excel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_excel = out_dir / "df_limpio_2023.xlsx"
    df.to_excel(p_excel, index=False)

    logger.info("Archivo procesado guardado en: %s", p_excel)

if __name__ == "__main__":
    # Ejecución simple del ETL (ajusta rutas según entorno)
    RAW_PATH = Path(r"C:\Users\Valentina Molina\Documents\repositorios\Proyecto_II\datos\Bas_de_datos_riesgos_original_30.09.2025.xlsx")
    OUT_DIR = Path(r"C:\Users\Valentina Molina\Documents\repositorios\Proyecto_II\datos")
    df_raw = load_raw(RAW_PATH)
    df_clean = clean_data(df_raw)
    save_processed(df_clean, OUT_DIR)
    logger.info("ETL finalizado.")