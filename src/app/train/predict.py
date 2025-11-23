import argparse
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# 1) Cargar modelo + umbral
# ===============================
def cargar_modelo_y_umbral(ruta_modelo, ruta_umbral="umbral_optimo.txt"):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"❌ No se encontró el modelo: {ruta_modelo}")

    modelo = joblib.load(ruta_modelo)

    if not os.path.exists(ruta_umbral):
        raise FileNotFoundError(f"❌ No se encontró el umbral en: {ruta_umbral}")

    with open(ruta_umbral, "r") as f:
        umbral = float(f.read().strip())

    print(f"✔ Modelo cargado: {ruta_modelo}")
    print(f"✔ Umbral óptimo cargado: {umbral:.4f}")
    return modelo, umbral

# ===============================
# 2) Transformaciones mínimas (sin tocar encoders)
# ===============================
def preparar_features(df: pd.DataFrame):
    # Convertir fecha y crear variables temporales
    df["FechaSeguimiento_DT"] = pd.to_datetime(df["FechaSeguimiento_DT"], errors="coerce", dayfirst=True)
    df["mes"] = df["FechaSeguimiento_DT"].dt.month
    df["año"] = df["FechaSeguimiento_DT"].dt.year

    # Mapear severidad a número
    mapa_severidad = {
        "Riesgo Bajo": 1,
        "Riesgo Medio": 2,
        "Riesgo Significativo": 3,
        "Riesgo Alto": 4,
        "Riesgo Crítico": 5
    }
    df["Severidad_NUM"] = df["Severidad_DS"].map(mapa_severidad)

    # Crear valor acumulado
    df["valor_acumulado"] = df["ValorInterno_VR"].fillna(0) + df["ValorCliente_VR"].fillna(0)

    return df

# ===============================
# 3) Aplicar encoders guardados (solo transform)
# ===============================
def aplicar_encoders(df: pd.DataFrame, ruta_encoders):
    # Cargar encoders
    encoder_causa = joblib.load(os.path.join(ruta_encoders, "encoder_Causa_DS.pkl"))
    encoder_agente = joblib.load(os.path.join(ruta_encoders, "encoder_AgenteGenerador_DS.pkl"))

    # Limpiar espacios
    df["Causa_DS"] = df["Causa_DS"].astype(str).str.strip()
    df["AgenteGenerador_DS"] = df["AgenteGenerador_DS"].astype(str).str.strip()

    # Verificar categorías desconocidas (LabelEncoder usa classes_)
    desconocidas_causa = set(df["Causa_DS"]) - set(encoder_causa.classes_)
    desconocidas_agente = set(df["AgenteGenerador_DS"]) - set(encoder_agente.classes_)

    if desconocidas_causa:
        print(f"⚠ Categorías desconocidas en Causa_DS: {desconocidas_causa}")
    if desconocidas_agente:
        print(f"⚠ Categorías desconocidas en AgenteGenerador_DS: {desconocidas_agente}")

    # Transformar
    df["Causa_DS"] = encoder_causa.transform(df["Causa_DS"])
    df["AgenteGenerador_DS"] = encoder_agente.transform(df["AgenteGenerador_DS"])

    return df

# ===============================
# 4) Predecir
# ===============================
def predecir(modelo, umbral, df):
    columnas = [
        "Causa_DS",
        "AgenteGenerador_DS",
        "Severidad_NUM",
        "año",
        "valor_acumulado"
    ]

    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"❌ El archivo no contiene columnas necesarias: {faltantes}")

    X = df[columnas]
    probs = modelo.predict_proba(X)[:, 1]
    df["probabilidad"] = probs
    df["prediccion"] = (probs >= umbral).astype(int)

    return df

# ===============================
# 5) MAIN
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realizar predicciones manuales")
    
    # Modelo entrenado (ruta absoluta)
    parser.add_argument(
        "--modelo",
        type=str,
        default=r"C:\Users\crisa\Documents\repositorios\Proyecto_II\modelos\modelo_gradientboosting_optimo.pkl",
        help="Archivo del modelo entrenado (.pkl)"
    )
    
    # CSV de entrada (ruta absoluta por defecto)
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\crisa\Documents\repositorios\Proyecto_II\datos\input\predicciones_input.csv",
        help="Archivo CSV con los datos nuevos"
    )
    
    # CSV de salida (ruta absoluta)
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\crisa\Documents\repositorios\Proyecto_II\datos\output\predicciones.csv",
        help="Archivo CSV donde se guardarán las predicciones"
    )
    
    # Carpeta con encoders (ruta absoluta)
    parser.add_argument(
        "--encoders",
        type=str,
        default=r"C:\Users\crisa\Documents\repositorios\Proyecto_II\encoders",
        help="Carpeta donde se encuentran los encoders guardados"
    )
    
    args = parser.parse_args()
    
    # Verificar la ruta de encoders
    print(f"Ruta de encoders usada: {args.encoders}")
    
    print("\n=== PREDICTOR MANUAL ===\n")

    # Cargar modelo y umbral
    modelo, umbral = cargar_modelo_y_umbral(args.modelo)
    
    # Cargar datos de entrada
    print(f"✔ Cargando datos desde: {args.input}")
    df_input = pd.read_csv(args.input, sep=";")  # Ajusta sep=";" si tu CSV lo requiere
    
    # Preparar features
    print("✔ Preparando features necesarias...")
    df_input = preparar_features(df_input)

    # ======== Imputar valores faltantes ========
    # Para numéricas, puedes usar 0 o la media según convenga
    df_input["Severidad_NUM"] = df_input["Severidad_NUM"].fillna(0)
    df_input["año"] = df_input["año"].fillna(0)
    df_input["valor_acumulado"] = df_input["valor_acumulado"].fillna(0)

    # Para categóricas ya codificadas con LabelEncoder, usa un valor por defecto antes de transformarlas
    df_input["Causa_DS"] = df_input["Causa_DS"].fillna("Desconocido")
    df_input["AgenteGenerador_DS"] = df_input["AgenteGenerador_DS"].fillna("Desconocido")
    # =========================================
    
    # Aplicar encoders
    print("✔ Aplicando encoders a variables categóricas...")
    df_input = aplicar_encoders(df_input, args.encoders)
    
    # Realizar predicciones
    df_pred = predecir(modelo, umbral, df_input)
    
    # Guardar resultado
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_pred.to_csv(args.output, index=False)
    print(f"\n📁 Archivo generado: {args.output}")
    print("🎉 Predicción finalizada")