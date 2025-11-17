import logging
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

from etl import load_raw, clean_data
from feature_engineer import crear_materializado_30d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from train import Train

def run_training_with_mlflow(df_path, target_column, model_output_path):

    # ======================
    # 1. Cargar y preparar datos
    # ======================
    df = load_raw(df_path)
    df = clean_data(df)
    df = crear_materializado_30d(df)

    # ======================
    # 2. Definir modelo
    # ======================
    modelo = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    # ======================
    # 3. Entrenar usando tu clase Train
    # ======================
    trainer = Train(df=df, target_column=target_column, model=modelo)
    pipeline, X_train, X_test, y_train, y_test = trainer.train()

    # ======================
    # 4. Registrar en MLflow
    # ======================
    with mlflow.start_run():

        mlflow.log_param("modelo", "GradientBoostingClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("subsample", 0.8)

        # métricas
        from sklearn.metrics import roc_auc_score, f1_score

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("F1", f1)

        # registrar modelo en MLflow
        mlflow.sklearn.log_model(pipeline, "modelo_riesgos")

        # Capturar el run_id
        print(f"Run ID: {mlflow.active_run().info.run_id}")


    # ======================
    # 5. Guardar modelo localmente
    # ======================
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, model_output_path)
    print(f"Modelo guardado en: {model_output_path}")


if __name__ == "__main__":
    run_training_with_mlflow(
        df_path="datos/df_limpio_2023.xlsx",
        target_column="materializado_30d",
        model_output_path="modelos/modelo_riesgos.pkl"
    )