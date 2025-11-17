import mlflow
import mlflow.sklearn
import logging

from etl import load_raw, clean_data
from feature_engineer import crear_materializado_30d
from train import Train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def task_train():
    
    df = load_raw("datos/df_limpio_2023.xlsx")
    df = clean_data(df)
    df = crear_materializado_30d(df)

    target_column = "materializado_30d"
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier

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
        eval_metric='logloss')
    }
     
    resultados = {}

    for nombre_modelo, modelo in modelos.items():
        logger.info(f"Entrenando el modelo: {nombre_modelo}")
        trainer = Train(df, target_column, modelo, nombre_modelo)
        trainer.preprocess_data()
        trainer.train_model()
        metrics = trainer.evaluate_model()
        resultados[nombre_modelo] = metrics

        mlflow.sklearn.log_model(modelo, artifact_path=nombre_modelo)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)