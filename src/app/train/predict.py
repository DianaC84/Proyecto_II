import logging
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from etl import clean_data
from feature_engineer import crear_materializado_30d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(model_name: str,
                  stage: str,
                  input_path: str,
                  output_path: str):
    
    # 1️⃣ Cargar modelo desde MLflow Model Registry
    logger.info(f"Cargando modelo '{model_name}' en stage '{stage}' desde MLflow...")
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Modelo cargado correctamente.")

    # 2️⃣ Cargar datos de entrada
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_path}")

    if input_path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 3️⃣ Aplicar limpieza y feature engineering mínima
    df = clean_data(df)
    df = crear_materializado_30d(df)

    # 4️⃣ Seleccionar features utilizadas por el modelo
    features = [
        "Causa_DS",
        "AgenteGenerador_DS",
        "Severidad_NUM",
        "Año",
        "valor_acumulado"
    ]
    X = df[features]

    # 5️⃣ Predecir
    logger.info("Realizando predicciones...")
    df["materializado_pred"] = model.predict(X)
    df["materializado_proba"] = model.predict_proba(X)[:, 1]

    # 6️⃣ Guardar resultados
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    logger.info(f"Predicciones guardadas en: {output_path}")


# ---------------------------------------
# CLI
# ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicciones batch con modelo MLflow")
    parser.add_argument("input_file", type=str, help="Ruta del archivo de datos de entrada (CSV o Excel)")
    parser.add_argument("output_file", type=str, help="Ruta donde se guardarán las predicciones")
    parser.add_argument("--stage", type=str, default="Production", help="Stage del modelo en MLflow (Staging o Production)")
    parser.add_argument("--model_name", type=str, default="riesgos_materializacion_model", help="Nombre del modelo en MLflow Model Registry")
    
    args = parser.parse_args()

    predict(
        model_name=args.model_name,
        stage=args.stage,
        input_path=args.input_file,
        output_path=args.output_file
    )