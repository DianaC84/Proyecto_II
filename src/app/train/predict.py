import argparse
import pandas as pd
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
# 2) Predecir
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
# 3) MAIN
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realizar predicciones manuales")
    parser.add_argument("--modelo", type=str, default="modelo_gradientboosting_optimo.pkl")
    parser.add_argument("--input", type=str, required=True, help="Archivo CSV con datos nuevos")
    parser.add_argument("--output", type=str, default="predicciones.csv")

    args = parser.parse_args()

    print("\n=== PREDICTOR MANUAL ===\n")

    modelo, umbral = cargar_modelo_y_umbral(args.modelo)

    print(f"✔ Cargando datos desde: {args.input}")
    df_input = pd.read_csv(args.input)

    df_pred = predecir(modelo, umbral, df_input)

    df_pred.to_csv(args.output, index=False)
    print(f"\n📁 Archivo generado: {args.output}")
    print("🎉 Predicción finalizada")
