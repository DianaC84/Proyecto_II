import pandas as pd
import joblib
import argparse
import os

def cargar_modelo(ruta_modelo):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")
    print(f"✔ Cargando modelo: {ruta_modelo}")
    return joblib.load(ruta_modelo)

def cargar_datos(ruta_csv):
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    print(f"✔ Cargando datos: {ruta_csv}")
    return pd.read_csv(ruta_csv)

def preparar_features(df):
    required = ['Causa_DS', 'AgenteGenerador_DS', 'Severidad_NUM', 'año', 'valor_acumulado']

    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"❌ Faltan columnas necesarias: {missing}")

    print("✔ Features preparados")
    return df[required]

def generar_predicciones(modelo, X):
    print("✔ Generando predicciones...")
    probs = modelo.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    return preds, probs

def guardar_salida(df, preds, probs, ruta_salida):
    df_out = df.copy()
    df_out["probabilidad"] = probs
    df_out["prediccion"] = preds
    df_out.to_csv(ruta_salida, index=False)
    print(f"✔ Archivo generado: {ruta_salida}")

def main():
    parser = argparse.ArgumentParser(description="Batch prediction para riesgos")
    parser.add_argument("--modelo", type=str, default="modelo_gradientboosting_optimo.pkl")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="predicciones.csv")

    args = parser.parse_args()

    modelo = cargar_modelo(args.modelo)
    df = cargar_datos(args.input)
    X = preparar_features(df)
    preds, probs = generar_predicciones(modelo, X)
    guardar_salida(df, preds, probs, args.output)

if __name__ == "__main__":
    main()