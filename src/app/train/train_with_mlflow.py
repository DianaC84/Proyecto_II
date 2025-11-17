import mlflow
import mlflow.sklearn
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from etl import load_raw, clean_data
from feature_engineer import crear_materializado_30d
from train import Train

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_mlflow():
    # -------------------------------
    # 1. Cargar y preparar los datos
    # -------------------------------
    logger.info("Cargando datos procesados...")
    df = load_raw("datos/df_limpio_2023.xlsx")

    logger.info("Aplicando limpieza adicional...")
    df = clean_data(df)

    logger.info("Creando etiqueta materializado_30d...")
    df = crear_materializado_30d(df)

    target_column = "materializado_30d"

    # -------------------------------
    # 2. Definir modelos
    # -------------------------------
    modelos = {
        "RandomForest_v2": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced",
            max_depth=None, min_samples_leaf=2
        ),
        "GradientBoosting_v2": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "XGBoost_v2": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42, scale_pos_weight=10,
            eval_metric='logloss'
        )
    }

    # -------------------------------
    # 3. Entrenar cada modelo con MLflow
    # -------------------------------
    mlflow.set_experiment("riesgos_materializacion_30d")

    for nombre_modelo, modelo in modelos.items():
        logger.info(f"---- Entrenando: {nombre_modelo} ----")

        with mlflow.start_run(run_name=nombre_modelo):

            # Registrar parámetros del modelo
            mlflow.log_params(modelo.get_params())

            # Crear instancia Train
            trainer = Train(df, target_column, modelo)

            # Entrenar
            pipeline, X_train, X_test, y_train, y_test = trainer.train()

            # Predicciones
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Métricas
            auc = roc_auc_score(y_test, y_proba)
            reporte = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("precision", reporte["1"]["precision"])
            mlflow.log_metric("recall", reporte["1"]["recall"])
            mlflow.log_metric("f1", reporte["1"]["f1-score"])
            
            # Guardar modelo entrenado
            mlflow.sklearn.log_model(pipeline, artifact_path=nombre_modelo)
            print(f"Modelo guardado en MLflow bajo el run_id: {mlflow.active_run().info.run_id}")
            logger.info(f"{nombre_modelo} - AUC: {auc:.4f}")

    logger.info("----- Entrenamiento con MLflow finalizado -----")


if __name__ == "__main__":
    train_with_mlflow()