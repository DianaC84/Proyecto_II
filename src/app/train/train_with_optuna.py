import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna

from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from etl import load_raw, clean_data
from feature_engineer import crear_materializado_30d
from train import Train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================================================
# OBJETIVO DE OPTUNA
# ===============================================================
def objective(trial, df, target_column):
    """
    Función objetivo de Optuna.
    Entrena un modelo con hiperparámetros sugeridos y devuelve el AUC.
    """

    # Seleccionar modelo a optimizar
    modelo_seleccionado = trial.suggest_categorical(
        "modelo",
        ["RandomForest", "GradientBoosting", "XGBoost"]
    )

    # -----------------------------------------------------------
    # 1. Espacios de búsqueda por modelo
    # -----------------------------------------------------------
    if modelo_seleccionado == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "class_weight": "balanced",
            "random_state": 42
        }
        model = RandomForestClassifier(**params)

    elif modelo_seleccionado == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "random_state": 42
        }
        model = GradientBoostingClassifier(**params)

    else:  # XGBoost
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 5, 20),
            "eval_metric": "logloss",
            "random_state": 42
        }
        model = XGBClassifier(**params)

    # -----------------------------------------------------------
    # 2. Entrenamiento usando Train()
    # -----------------------------------------------------------
    trainer = Train(df, target_column, model)
    pipeline, X_train, X_test, y_train, y_test = trainer.train()

    # -----------------------------------------------------------
    # 3. Métricas
    # -----------------------------------------------------------
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    # -----------------------------------------------------------
    # 4. Registrar cada trial en MLflow
    # -----------------------------------------------------------
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("AUC", auc)

    return auc  # Optuna maximiza esto


# ===============================================================
# MAIN
# ===============================================================
def train_with_optuna_mlflow():
    logger.info("=== INICIANDO OPTIMIZACIÓN CON OPTUNA + MLflow ===")

    # -----------------------------------------------------------
    # 1. Cargar datos
    # -----------------------------------------------------------
    df = load_raw("datos/df_limpio_2023.xlsx")
    df = clean_data(df)
    df = crear_materializado_30d(df)

    target_column = "materializado_30d"

    # -----------------------------------------------------------
    # 2. Crear estudio Optuna
    # -----------------------------------------------------------
    mlflow.set_experiment("Optuna_Riesgos_ML")

    study = optuna.create_study(direction="maximize")

    with mlflow.start_run(run_name="Optuna_Study_Run"):
        study.optimize(
            lambda trial: objective(trial, df, target_column),
            n_trials=30,          # Ajustable
            show_progress_bar=True
        )

        # -----------------------------------------------------------
        # 3. Guardar mejores hiperparámetros en MLflow
        # -----------------------------------------------------------
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_auc", study.best_value)

        logger.info(f"Mejor AUC: {study.best_value}")
        logger.info(f"Mejores parámetros: {study.best_trial.params}")

        # -----------------------------------------------------------
        # 4. Entrenar modelo final con los mejores hiperparámetros
        # -----------------------------------------------------------
        best_params = study.best_trial.params
        modelo_name = best_params.pop("modelo")

        if modelo_name == "RandomForest":
            best_model = RandomForestClassifier(**best_params)
        elif modelo_name == "GradientBoosting":
            best_model = GradientBoostingClassifier(**best_params)
        else:
            best_params["eval_metric"] = "logloss"
            best_model = XGBClassifier(**best_params)

        trainer = Train(df, target_column, best_model)
        best_pipeline, _, _, _, _ = trainer.train()

        # Guardarlo en MLflow
        mlflow.sklearn.log_model(best_pipeline, artifact_path="best_optuna_model")

        print(f"Modelo guardado en MLflow bajo el run_id: {mlflow.active_run().info.run_id}")

    logger.info("=== OPTIMIZACIÓN FINALIZADA ===")


if __name__ == "__main__":
    train_with_optuna_mlflow()
    